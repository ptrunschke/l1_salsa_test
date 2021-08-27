import warnings
from functools import lru_cache

import numpy as np
from scipy import linalg
from scipy.linalg.lapack import get_lapack_funcs
from bisect import insort_left


def min_pos(X):
    """
    Find the minimum value of an array over positive values
    Returns a huge value if none of the values are positive
    """
    return np.min(X[X > 0], initial=np.inf)


def rotg(x):
    """
    Compute cos and sin entries for Givens plane rotation.

    Given the Cartesian coordinates x of a point in two dimensions, the function returns the parameters c and s
    associated with the Givens rotation and modifies x accordingly. The parameters c and s define a unitary matrix such that:

        ┌      ┐      ┌   ┐
        │ c  s │      │ r │
        │-s  c │ x == │ 0 │
        └      ┘      └   ┘

    x is modified to contain x[0] = r >= 0 and x[1] = 0.
    """
    if np.all(x==0):
        return np.array([1,0])  # c, s
    else:
        r = np.linalg.norm(x)
        cs = x/r
        x[:] = (r, 0)
        return cs


def rot(xy, cs):
    """
    Perform a Givens plane rotation for the cos and sin entries c and s.

    Given two vectors x and y, each vector element of these vectors is replaced as follows:
        xₖ = c*xₖ + s*yₖ
        yₖ = c*yₖ - s*xₖ
    """
    assert xy.ndim == 2 and xy.shape[0] == 2 and cs.shape == (2,)

    c,s = cs
    x,y = xy
    tmp = c*x + s*y
    y[:] = c*y - s*x
    x[:] = tmp


def cholesky_delete(L, go_out):
    """
    Remove a row and column from the cholesky factorization.
    """
    # Delete row go_out.
    L[go_out:-1] = L[go_out+1:]
    # The resulting matrix has non-zero elements in the off-diagonal entries L[i,i+1] for i>=go_out.

    n = L.shape[0]
    for i in range(go_out, n-1):
        # Rotate the two column vectors L[i:,i] and L[i:,i+1].
        #NOTE: rotg gives the rotation that provides a positive diagonal entry.
        cs = rotg(L[i,i:i+2])
        rot(L[i+1:,i:i+2].T, cs)


def est_cond_triangular(L):
    """
    Lower bound on the condition number of a triangular matrix w.r.t. the row-sum norm.

    References:
    -----------
    .. [1] https://en.wikipedia.org/wiki/Condition_number
    .. [2] https://github.com/PetterS/SuiteSparse/blob/master/CHOLMOD/Cholesky/cholmod_rcond.c
    """
    assert L.ndim == 2 and L.shape[0] == L.shape[1]
    if L.shape[0] == 0: return 0
    diagL = abs(np.diag(L))
    return np.max(diagL) / np.min(diagL)


class ConvergenceWarning(UserWarning):
    pass


class LarsState(object):
    """
    References
    ----------
    .. [1] "Least Angle Regression", Efron et al.
           http://statweb.stanford.edu/~tibs/ftp/lars.pdf
    .. [2] `Wikipedia entry on the Least-angle regression
           <https://en.wikipedia.org/wiki/Least-angle_regression>`_
    .. [3] `Wikipedia entry on the Lasso
           <https://en.wikipedia.org/wiki/Lasso_(statistics)>`_

    """
    # def __init__(self, X, y, max_features, eps=np.finfo(np.float).eps, verbose=False, X_test=None):
    def __init__(self, X, y, max_features, verbose=False, X_test=None):
        """
        max_features <= max_iter
        """
        # self.tiny = np.finfo(np.float32).tiny  # Used to avoid division by 0.
        self.tiny = np.finfo(np.float64).tiny  # Used to avoid division by 0 without perturbing other computations too much. (Note that x+self.tiny != x only if x is tiny itself.)
        self.condition_threshold = 1e8

        n_samples, n_features = X.shape
        assert np.shape(y) == (n_samples,)
        self.max_features = min(max_features, n_features)
        self.verbose = verbose

        self.active = []  #NOTE: COEF AND ACTIVE ALWAYS HAVE THE SAME LENGTH
        self.inactive = list(range(n_features))  #NOTE: ALWAYS SORTED

        self.X = X
        self.X_test = X_test
        assert X_test is None or X.shape == X_test.shape
        self.y = y  #TODO: just relevant for tests

        self.coef = np.zeros((0,))

        self.Cov = np.asarray(self.XTy(y))      # Cov[n_active:] contains the covariances of the inactive covariates. Cov[:n_active] = 0.
        assert self.Cov.shape == (n_features,)
        self.signCov = np.empty(n_features, dtype=np.int8)  # sign_active[:n_active] holds the sign of the covariance of active covariates.
        assert self.signCov.shape == (n_features,)  #TODO: maybe signCov can be eliminated all together
        self.C = np.max(np.fabs(self.Cov))  # covariance of the active covariates
        self.CData = np.inf
        self.prevCData = np.inf

        self.L = np.zeros((max(int(0.1*max_features), 1), max(int(0.1*max_features), 1)), dtype=y.dtype)  # will hold the cholesky factorization. Only lower part is referenced.
        self.solve_cholesky, = get_lapack_funcs(('potrs',), (self.L,))
        self.G = np.zeros((max(int(0.1*max_features), 1), n_features), dtype=y.dtype)   # G[i] will hold X[:,active[i]].T @ X

        self.drop = False

    #NOTE: Methods just needed for testing and profiling.
    def XTy(self, y):
        if self.X_test is None:
            return self.X.XTy(y)
        else:
            ret = self.X.XTy(y)
            ret_test = self.X_test.XTy(y)
            assert np.allclose(ret, ret_test)
            return ret
    def XTX(self, C_idx):
        if self.X_test is None:
            return self.X.XTX(C_idx)
        else:
            ret = self.X.XTX(C_idx)
            ret_test = self.X_test.XTX(C_idx)
            assert np.allclose(ret, ret_test)
            return ret
    def add_L(self):
        if self.verbose: print("add_L")
        n_active = len(self.active)
        self.L[n_active, :n_active] = self.G[n_active,self.active]  # Y.T@y
        linalg.solve_triangular(self.L[:n_active, :n_active], self.L[n_active, :n_active], lower=1, overwrite_b=True, check_finite=False)  # Solve L@w = Y.T@y. `overwrite_b=True` implies that L[n_active, :n_active] will contain the solution.

    #TODO: write an access method for G! G[:n_active,self.active] kann sich doch keine sau merken.

    def check_condition_threshold(self):
        n_active = len(self.active)
        assert np.all(np.isfinite(self.G[:n_active]))
        assert np.all(np.isfinite(self.L[:n_active,:n_active]))
        assert est_cond_triangular(self.L[:n_active,:n_active])**2 <= self.condition_threshold

    def add_index(self):
        """
        Update the cholesky decomposition for the Gram matrix L@L.T = X.T@X

        Let Y = X[:,:n_active] and L be the Cholesky factor of Y.T@Y.
        Compute the factor L_new of Y_new = X[:,:n_active+1] as
                    ┌      ┐
                    │ L  0 │
            L_new = │      │.
                    │ w  z │
                    └      ┘
        Note that Y_new = [ Y y ] with y = X[:,n_active].
        Writing out the condition L_new@L_new.T = Y.T@Y yields the conditions

            L@w = Y.T@y    and    z**2 = y.T@y - w.T@w.

        """
        if self.verbose: print("add_index")
        if len(self.active) == self.max_features:
            raise ConvergenceWarning("Stopping the lars path, as every regressor is active.")
        assert len(self.active) < self.max_features
        self.check_condition_threshold()

        n_samples = self.X.shape[0]

        while True:
            C_idx_inactive = np.argmax(np.fabs(self.Cov[self.inactive]))
            C_idx = self.inactive[C_idx_inactive]
            if abs(self.Cov[C_idx]) == 0:
                #NOTE: This is probably an index that has been removed earlier.
                raise ConvergenceWarning("Early stopping the lars path, as every regressor with nonzero covariance would make the system unstable.")

            self.prevCData, self.CData = self.CData, abs(self.Cov[C_idx])
            #TODO: Think about using CData instead of C!
            #NOTE: lars.C can be updated in every step via self.C -= gamma * AA. This however is numerically unstable.
            #      The numerical values of the covariances (y.T @ X @ coef) should all be equal but tend to diverge.
            #      In itself this may not pose a problem. But when these covariances fall below those in self.Cov this means
            #      that the updates of Cov bring a numerical error that is greater than the remaining correlation with the regressors.
            # print(f"{self.C:.3e} == {self.CData:.3e} <= {self.prevCData:.3e}")
            if self.CData > self.prevCData:
                raise ConvergenceWarning("Early stopping the lars path, as the remaining covariances fall below the numerical error.")


            self.signCov[C_idx] = np.sign(self.Cov[C_idx])
            self.Cov[C_idx] = 0  #TODO: Anstatt das zu machen könnte ich auch einfach direkt das Element aus self.inactive entfernen, oder? Dann bräuchte man self.signCov nicht mehr...

            n_active = len(self.active)

            # After the update the new Gramian will contain the follwing new row and column (the Gramian is symmetric...):
            #     R = self.XTX(C_idx)[self.active+[C_idx]]
            # Define
            #     ER = R[-1] + np.sum(abs(R[:-1]))
            # Then the Gershgorin circle theorem gives the following estimate for the eigenvalue EV that is added to this new Gramian:
            #     EV <= ER
            # To estimate the conditon number we need to find the largest eigenvalue.
            # We can do this with Gershgorin as well (even updating the previous estimate) but for the sake of simplicity we use numpy: 
            #     EU = np.linalg.norm(self.G[:n_active,self.active], ord=2)
            # Thus if ER <= self.condition_threshold we will drop this index anyways...
            # An improvement: Let a be a real number and define
            #     R(a) = R*a**(-np.arange(n_active+1)[::-1])
            # and ER as above.
            # as described in https://math.stackexchange.com/a/177253/303111 this is just regular Gershgorin theorem applied to the matrix
            # U XTX inv(U) where U is the diagonal matrix with entries U[i,i] = a**i.
            # So in the limit a\to\infty ER goes to R[-1].
            #TODO: Check the math!
            if n_active > 0:
                # a = 10.0
                # R *= a**(-np.arange(n_active+1)[::-1])
                # ER = R[-1] + np.sum(abs(R[:-1]))
                R = self.XTX(C_idx)[self.active+[C_idx]]
                ER = R[-1]
                EU = np.linalg.norm(self.G[:n_active,self.active], ord=2)
                CG = EU/ER
                if CG >= self.condition_threshold:
                    warnings.warn(f"Regressors in active set degenerate. Skipping one with small relative eigenvalue: {{{C_idx}: {1/CG:.2e}}}.", ConvergenceWarning)
                    continue

            # if n_active > 0:
            #     L_test = np.zeros((n_active, n_active))
            #     for _ in range(n_active):
            #         L_test[_,:_+1] = self.L[_,:_+1]
            #     assert np.allclose(L_test @ L_test.T, self.G[:n_active,self.active])
            #     assert np.all(np.diag(L_test[:n_active, :n_active]) > 0)              # This is a feature of our specific construction.
            #     assert np.allclose(np.linalg.norm(L_test, ord=2), np.max(np.diag(L_test)))
            #     assert np.allclose(np.linalg.norm(L_test, ord=-2), np.min(np.diag(L_test)))

            # Ensure that G and L have the capacity for a new row.
            assert self.G.shape[0] == self.L.shape[0]
            if self.G.shape[0] <= n_active:
                n_features = self.X.shape[1]
                assert self.G.shape == (n_active, n_features)
                assert self.L.shape == (n_active, n_active)
                old_capacity = n_active
                new_capacity = min(2*n_active, self.max_features)
                #NOTE: Pad G and L with zeros. Otherwise they can contain nan's or inf's that fail the assert in check_condition_threshold.
                self.G = np.pad(self.G, ((0, new_capacity-old_capacity), (0, 0)))
                self.L = np.pad(self.L, ((0, new_capacity-old_capacity),))

            self.G[n_active] = self.XTX(C_idx)  # Y.T@y
            if n_active > 0:
                self.add_L()
            yTy = self.G[n_active, C_idx]
            wTw = np.linalg.norm(self.L[n_active,:n_active]) ** 2
            # self.L[n_active, n_active] = max(np.sqrt(max(yTy - wTw, 0)), self.eps)  #TODO: eps? The sklearn documentation states that eps regularizes the Cholesky decomposition. I am not sure of this.
            self.L[n_active, n_active] = np.sqrt(max(yTy - wTw, 0)) + self.tiny

            self.coef = np.append(self.coef, 0)
            self.active.append(C_idx)
            self.inactive.pop(C_idx_inactive)

            idcs_rem = self.remove_ill_conditioned_indices()
            if idcs_rem == [C_idx]:  # The last added regressor causes the system to become ill-conditioned.
                continue             # We can pretend like it never existed by starting over without increasing n_active.
            else:                    # In any other case we have added a new index (although `remove_ill_conditioned_indices` may have removed another).
                break

        self.check_condition_threshold()


    def remove_ill_conditioned_indices(self):
        """
        This function successively removes the regressors with smallest eigenvalue until the condition number of `self.G` deceeds `self.condition_threshold`.
        The corresponding indices have to be removed for good and can not be added back at a later time.
        This is because LARS adds regressors based on their covariance which is currently the largest among all active regressors.
        It returns a list of all indices that were removed.
        """

        #NOTE: The original idea was that if a regressor corresponds to a too small eigenvalue,
        #      that regressor is successively swapped down.
        #      In each swap the eigenvalues of the swapped regressors are displayed.
        #      If the swap results in an even lower eigenvalue for the regressor that is now on top
        #      then this regressor has to be removed first.
        #      This is cumbersome however and therefore the first regressor is simply removed (in a greedy fashion).

        idcs_rem = []
        eigs_rem = []
        n_active = len(self.active)
        while n_active > 1:
            #NOTE: The choice of the condition threshold is imporant and influences the quality of our solution. (It directly affects which regressors are added or dropped for good...)
            if est_cond_triangular(self.L[:n_active,:n_active])**2 <= self.condition_threshold:  # We solve the system L@x=y twice...
                break
            # This case happened for n_active == 1 when, in l1_salsa, self.L[0,0] = np.inf.
            ii = np.argmin(np.diag(self.L[:n_active,:n_active]))

            cholesky_delete(self.L[:n_active, :n_active], ii)
            for i in range(ii, n_active-1):
                self.G[i] = self.G[i+1]

            # #TODO: This is equivalent to the cholesky_delete block but prints the eigenvalues after each swap.
            # swap = linalg.get_blas_funcs('swap', (self.L,))
            # for i in range(ii, n_active-1):
            #     print(f"swap({i},{i+1})")
            #     prev_diag = [self.L[i,i], self.L[i+1,i+1]]
            #     swap(self.L[i], self.L[i+1])
            #     swap(self.G[i], self.G[i+1])
            #     cs = rotg(self.L[i,i:i+2])
            #     rot(self.L[i+1:,i:i+2].T, cs)
            #     print(f"    {prev_diag} -> {[L[i,i], L[i+1,i+1]]}")

            self.coef = np.delete(self.coef, ii)
            C_idx = self.active.pop(ii)
            insort_left(self.inactive, C_idx)
            self.Cov[C_idx] = 0  # Drop the index for good.

            idcs_rem.append(C_idx)
            eigs_rem.append(self.L[n_active-1, n_active-1])

            n_active -= 1

        if len(idcs_rem) > 0:
            n_active = len(self.active)
            max_eig = np.max(np.diag(self.L[:n_active, :n_active]))
            rel_eigs_rem = "{" + ",".join(f"{idx}: {eig/max_eig:.2e}" for idx,eig in zip(idcs_rem, eigs_rem)) + "}"
            warnings.warn(f"Regressors in active set degenerate. Removing those with small relative eigenvalues: {rel_eigs_rem}.", ConvergenceWarning)

        return idcs_rem

    def remove_index(self):
        if self.verbose: print("remove_index")
        self.check_condition_threshold()

        n_active = len(self.active)
        idx = np.nonzero(np.fabs(self.coef) < 1e-12)[0][::-1]
        if len(idx) == 0:
            raise ConvergenceWarning(f"Early stopping the lars path, as a regressor that ought to be removed from the active set has nonzero coefficient.")
        assert np.all(idx[:-1] > idx[1:])

        for ii in idx:
            if n_active <= 1:
                break
            cholesky_delete(self.L[:n_active, :n_active], ii)
            for i in range(ii, n_active-1):
                self.G[i] = self.G[i+1]
            self.coef = np.delete(self.coef, ii)
            C_idx = self.active.pop(ii)
            insort_left(self.inactive, C_idx)
            self.Cov[C_idx] = self.C*self.signCov[C_idx]
            n_active -= 1

        self.remove_ill_conditioned_indices()
        self.check_condition_threshold()
        self.drop = False

    def step(self):
        if self.verbose: print("step")
        self.check_condition_threshold()

        # least squares solution
        n_active = len(self.active)
        least_squares, _ = self.solve_cholesky(self.L[:n_active, :n_active], self.signCov[self.active], lower=True)
        AA = 1. / np.sqrt(np.sum(least_squares * self.signCov[self.active]))
        assert 0 < AA and np.isfinite(AA)
        least_squares *= AA  # w as defined in (2.6)

        # equiangular direction of variables in the active set
        corr_eq_dir = least_squares @ self.G[:n_active, self.inactive]  # a as defined in (2.12)

        gamma_hat = min(min_pos((self.C-self.Cov[self.inactive]) / (AA-corr_eq_dir+self.tiny)),
                        min_pos((self.C+self.Cov[self.inactive]) / (AA+corr_eq_dir+self.tiny)))  # \hat{\gamma} as defined in (2.13)
        gamma_hat = min(gamma_hat, self.C/AA)  # Stabilizes the algorithm since C/AA is the maximal step size (self.C - gamma_hat*AA == 0).

        bOd = - self.coef / (least_squares+self.tiny)  # - \hat{\beta} / \hat{d}
        gamma_tilde = min_pos(bOd)  # \tilde{\gamma} as defined in (3.5)
        gamma = min(gamma_tilde, gamma_hat)

        # update coefficient
        self.coef += gamma * least_squares
        # update correlations
        self.Cov[self.inactive] -= gamma * corr_eq_dir
        self.C -= gamma * AA

        self.drop = gamma_tilde < gamma_hat


def lasso_lars(X, y, alpha, max_iter=500, verbose=False, X_test=None):
    n_samples, n_features = X.shape
    assert y.shape == (n_samples,)
    max_features = min(max_iter, n_features)

    lars = LarsState(X, y, max_features, verbose=verbose, X_test=X_test)
    this_alpha = max(lars.C / n_samples, 0)
    coef = lars.coef.copy()
    active = lars.active[:]

    for n_iter in range(max_iter):
        if n_iter > 0 and this_alpha <= alpha:
            # In the first iteration, all alphas are zero, the formula below would make ss a NaN
            # TODO: a better reason: you want to be able to interpolate!

            # if len(active) == len(prev_active)+1:
            #     #NOTE: In the previous iteration an index was added.
            #     assert active[:-1] == prev_active
            #     prev_coef = np.append(prev_coef, 0)
            # elif len(active) == 0:
            #     #NOTE: In the previous iteration an index was removed. (special case)
            #     assert len(prev_active) == 1
            #     coef = np.insert(coef, 0, 0)
            #     active = prev_active
            # else:
            #     #NOTE: In the previous iteration an index was removed.
            #     for idx in range(len(active)):
            #         if prev_active[idx] != active[idx]: break
            #     else:
            #         idx += 1
            #     #NOTE: prev_active[idx] was removed.
            #     assert prev_active[:idx] == active[:idx]
            #     if len(active) == len(prev_active):
            #         #NOTE: prev_active[idx] was removed and another index was added.
            #         assert prev_active[idx+1:] == active[idx:-1]
            #         coef = np.insert(coef, idx, 0)
            #         prev_coef = np.append(prev_coef, 0)
            #         active = prev_active + [active[-1]]
            #     else:
            #         #NOTE: No other index was added.
            #         assert prev_active[idx+1:] == active[idx:]
            #         coef = np.insert(coef, idx, 0)
            #         active = prev_active
            

            # new_active = []
            # active contains new indices only at the end
            # the only question is where indices have been removed

            # you want all element of prev_active followed by the new elements of active (which are indeed at the end of active)


            pc = dict(zip(prev_active, prev_coef))
            cc = dict(zip(active, coef))

            active = list(sorted(set(prev_active) | set(active)))
            prev_coef = np.zeros(len(active))
            coef      = np.zeros(len(active))
            for e,idx in enumerate(active):
                if idx in pc: prev_coef[e] = pc[idx]
                if idx in cc: coef[e] = cc[idx]

            # if len(prev_active) == 0:
            #     #NOTE: In the previous iteration indices were added.
            #     assert len(active) > 0
            #     prev_active = active
            #     prev_coef = np.zeros(len(prev_active))
            # elif len(active) == 0:
            #     #NOTE: In the previous iteration indices were removed.
            #     assert len(prev_active) > 0
            #     active = prev_active
            #     coef = np.zeros(len(active))
            # else:
            #     #NOTE: Indices have been added and removed.
            #     # Find the first indices that has been added.
            #     j = len(active)-1
            #     while active[j] not in prev_active: j -= 1
            #     j += 1

            #     assert set(active[:j]) <= set(prev_active)           # The active indices that were not added are a subset of the previously active indices.
            #     assert set(active[j:]).isdisjoint(set(prev_active))  # The active indices that were added must not be contained in the previously active indices.


            #     new_active = prev_active + active[j:]
            #     assert set(new_active) == set(prev_active) + set(active)
            #     new_coef = np.empty(len(new_active))
            #     j = 0
            #     for i in range(len(new_active)):
            #         if j < len(active) and new_active[i] == active[j]:  # new_active[i] was not removed
            #             new_coef[i] = coef[j]
            #             j += 1
            #         else:                           # new_active[i] was removed
            #             new_coef[i] = 0
            #     assert j == len(active)  # ensure that every index has been considered
            #     active = new_active
            #     coef = new_coef
            #     prev_coef = np.concatenate([prev_coef, np.zeros(len(coef)-len(prev_coef))])

            coef = prev_coef + (prev_alpha - alpha) / (prev_alpha - this_alpha) * (coef - prev_coef)
            this_alpha = alpha
            break

        try:
            if not lars.drop:
                lars.add_index()
            lars.step()
            if lars.drop:
                lars.remove_index()
        except ConvergenceWarning as w:
            warnings.warn(str(w), ConvergenceWarning)
            break

        prev_alpha, this_alpha = this_alpha, lars.C / n_samples
        prev_coef, coef = coef, lars.coef.copy()
        prev_active, active = active, lars.active[:]
        coef = lars.coef
        if prev_alpha == this_alpha:
            warnings.warn("Early stopping the lars path, as correlations do not decrease.", ConvergenceWarning)
            break
        assert prev_alpha > this_alpha  # This should be guaranteed by the way lars.C is updated.
    else:
        warnings.warn("Maximum number of iterations exceeded.", ConvergenceWarning)

    class Model:
        alpha_ = this_alpha
        active_ = active
        coef_ = coef
        n_iter_ = n_iter
        state_ = lars

    return Model


def lasso_lars_cv(X, y, min_alpha=np.finfo(np.float64).eps, cv=10, max_iter=5000, overtrainingSteps=3, overtrainingFactor=2, verbose=False, X_test=None):
    """
    max_iter : maximum number of iterations across all folds
    """
    n_samples, n_features = X.shape
    assert np.shape(y) == (n_samples,)
    assert n_samples % cv == 0  #TODO: ...
    max_features = min(max_iter, n_features)

    lars = []
    alphas = []
    residuals = []
    testSetSize = n_samples // cv
    slc = lambda fold: slice(fold*testSetSize, (fold+1)*testSetSize)
    #TODO: num_samples = 250 and S = 3 == s()
    #      num_test_samples = 0.1*num_samples == 25
    #      num_training_samples = (1-0.1)*num_samples == 225 == n()
    #      --> n_samples = n() * s() == 675
    #      --> 675 / cv == 67.5
    #      --> 67 == 3*22 + 1 == s()*22 + 1
    #      --> 67 % s() != 0

    class GramedOperator(object):
        def __init__(self, _operator):
            self.operator = _operator
            self._active = dict()
            self._gramian = np.zeros((max(int(0.1*self.shape[1]), 1), self.shape[1]))
        @property
        def shape(self): return self.operator.shape
        def XTy(self, _y): return self.operator.XTy(_y)
        @lru_cache(maxsize=None)
        def XTX(self, _index):
            ret = self.operator.XTX(_index)
            n_active = len(self._active)
            if self._gramian.shape[0] <= n_active:
                assert self._gramian.shape == (n_active, self.shape[1])
                old_capacity = n_active
                new_capacity = min(2*n_active, self.shape[1])
                self._gramian = np.pad(self._gramian, ((0, new_capacity-old_capacity), (0, 0)), mode="empty")
            self._gramian[n_active] = ret
            self._active[_index] = n_active
            assert len(self._active) == n_active+1
            return ret
        def gramian(self, _indices):
            [self.XTX(idx) for idx in _indices]
            Gidcs = [self._active[idx] for idx in _indices]
            return self._gramian[Gidcs][:, _indices]
    class StackedOperator(object):
        def __init__(self, _stack):
            self.stack = _stack
            self.n_samples = sum(elem.shape[0] for elem in self.stack)
            self.n_features = self.stack[0].shape[1]
        @property
        def shape(self): return self.n_samples, self.n_features
        @lru_cache(maxsize=None)
        def XTX(self, _index):
            ret = sum(elem.XTX(_index) for elem in self.stack)
            return ret
        def XTy(self, _y):
            ret = np.zeros(self.n_features)
            a = 0
            for elem in self.stack:
                o = a + elem.shape[0]
                ret += elem.XTy(_y[a:o])
                a = o
            return ret
    takes = [GramedOperator(X.take_rows(slc(fold))) for fold in range(cv)]
    drops = [StackedOperator(takes[:fold]+takes[fold+1:]) for fold in range(cv)]
    yTys = [np.linalg.norm(y[slc(fold)])**2 for fold in range(cv)]
    yTXs = [takes[fold].XTy(y[slc(fold)]) for fold in range(cv)]

    def res(fold):
        # |y - Xc|^2 == yTy - 2yTXc + cTXTXc
        i = lars[fold].active
        c = lars[fold].coef
        G = takes[fold].gramian(i)
        ret = max(yTys[fold] - 2*yTXs[fold][i]@c + c.T@G@c, 0)
        tmp = X.take_rows(slc(fold)).error_norm(y[slc(fold)], lars[fold].active, lars[fold].coef)
        assert np.allclose(ret, tmp)
        return ret


    # print("""
    # TODO:
    #     - add a threshold parameter to LarsState
    #     - store least_squares and AA (from step()) in an attribute of LarsState
    #       -> the maximal update to coefs can be estimated as abs(coefs) + C/AA*abs(least_squares)
    #          if np.any(abs(coefs) < threshold):
    #              nt = np.count_nonzero(abs(coefs) < threshold)
    #              ntu = np.count_nonzero(abs(coefs) + C/AA*abs(least_squares) < threshold)
    #              if nt == ntu: return  # an update with maximal step size would still not make any of the lowest coefficients exceed the threshold
    #     - every time you drop an inactive index "for good" you need to update the coefs since C = max(abs(Cov)) changes
    #     - replace the update C -= gamma * AA in step() by a numerically more stable one?
    #     - add_index should add all (not only one) indices that have the same (maximum) correlation
    #     - TODO: du willst nicht einfach einen threshold, sondern einen weighted threshold machen!
    # """)



    # # res = lambda fold: np.linalg.norm(y[slc(fold)] - X.take_rows(slc(fold)).Xc(lars[fold].active, lars[fold].coef))**2
    # # res = lambda fold: X.take_rows(slc(fold)).error_norm(y[slc(fold)], lars[fold].active, lars[fold].coef)
    # #TODO: - res dauert genauso lange wie add_index
    # #      - add_index wird dominiert von XTX (XTX dauert über 117 mal so lange wie add_L)
    # #TODO: XTX kann man (maximal) um einen Faktor 10 beschleunigen indem man nicht mehr drop_rows verwendet. 
    # #      Aber res dauert dann immer noch genau so lange...
    # #TODO: Was dauert in res so lange?
    # # Implement functions to increase readability of the cProfile output.
    # def res(fold):
    #     def error_norm():
    #         if X_test is None:
    #             return X.take_rows(slc(fold)).error_norm(y[slc(fold)], lars[fold].active, lars[fold].coef)
    #         else:
    #             ret = X.take_rows(slc(fold)).error_norm(y[slc(fold)], lars[fold].active, lars[fold].coef)
    #             ret_test = X_test.take_rows(slc(fold)).error_norm(y[slc(fold)], lars[fold].active, lars[fold].coef)
    #             assert np.allclose(ret, ret_test)
    #             return ret
    #     return error_norm()

    tmpy = y[testSetSize:].copy()
    for fold in range(cv):
        if X_test is None:
            # lars.append(LarsState(X.drop_rows(slc(fold)), tmpy, max_features, verbose=verbose))
            lars.append(LarsState(drops[fold], tmpy, max_features, verbose=verbose))
        else:
            # lars.append(LarsState(X.drop_rows(slc(fold)), tmpy, max_features, verbose=verbose, X_test=X_test.drop_rows(slc(fold))))
            lars.append(LarsState(drops[fold], tmpy, max_features, verbose=verbose, X_test=X_test.drop_rows(slc(fold))))
        alphas.append([lars[-1].C / n_samples])
        residuals.append([res(fold)])
        if fold < cv-1:
            tmpy[slc(fold)] = y[slc(fold)]
    not_falling = np.full(cv, False, dtype=bool)
    minimum = np.full(cv, np.inf, dtype=float)

    for n_iter in range(max_iter):
        idx = np.argmax([a[-1] for a in alphas])
        alpha = alphas[idx][-1]
        if alpha <= min_alpha: break

        try:
            if not lars[idx].drop:
                lars[idx].add_index()
            lars[idx].step()
            if lars[idx].drop:
                lars[idx].remove_index()
        except ConvergenceWarning as w:
            warnings.warn(str(w), ConvergenceWarning)
            break  #TODO: Don't break but just don select this LarsState again.

        alphas[idx].append(max(lars[idx].C / n_samples, 0))
        residuals[idx].append(res(idx))
        if alphas[idx][-2] == alphas[idx][-1]:
            warnings.warn("Early stopping the lars path, as correlations do not decrease.", ConvergenceWarning)
            break
        assert alphas[idx][-2] > alphas[idx][-1]  # This should be guaranteed by the way lars.C is updated.

        # We want to exit early when half of the folds are in overtraining.
        # In overtraining the error does not only have to increase but it has to do so vigorously and consistently.
        # For the first condition we test if the residual exceeds the previous global minimum by more than a factor of `overtrainingFactor`.
        # For the second condition we test that this happens for more than `overtrainingSteps` steps.
        minimum[idx] = min(minimum[idx], residuals[idx][-1])
        # not_falling[idx] = residuals[idx][-1] > residuals[idx][-2]
        # not_falling[idx] = residuals[idx][-1] > overtraining_ratio*minimum[idx]
        not_falling[idx] = np.min(residuals[idx][-overtrainingSteps:]) > overtrainingFactor*minimum[idx]
        if np.count_nonzero(not_falling) > cv//2+cv%2:
            warnings.warn("Early stopping the lars path, as more than half of the folds have increasing residuals.", ConvergenceWarning)
            break
    else:
        warnings.warn("Maximum number of iterations exceeded.", ConvergenceWarning)

    all_alphas = np.concatenate(alphas)
    all_alphas = all_alphas[all_alphas >= max(alphas_idx[-1] for alphas_idx in alphas)]  # Remove all alphas that lie below the largest alpha among all folds.
    all_alphas = np.unique(all_alphas)  # np.unique also sorts
    all_residuals = [np.interp(all_alphas, a[::-1], r[::-1]) for a,r in zip(alphas, residuals)]

    # import matplotlib.pyplot as plt
    # geom_mean = np.exp(np.mean(np.log(all_residuals), axis=0))
    # minIdx_gm = np.argmin(geom_mean)
    # mean = np.mean(all_residuals, axis=0)
    # minIdx = np.argmin(mean)
    # fig, ax = plt.subplots(1,1)
    # axt = ax.twinx()
    # for idx in range(cv):
    #     # ax.plot(all_alphas, np.sqrt(all_residuals[idx]), f'C{idx}o--')
    #     ax.plot(all_alphas, all_residuals[idx], f'C{idx}o--')
    # ax.plot(all_alphas, geom_mean, color='k', ls='--', lw=2)
    # ax.plot(all_alphas, mean, color='k', lw=2)
    # ax.axvline(all_alphas[minIdx], color='k')
    # ax.axvline(all_alphas[minIdx_gm], color='k', ls='--')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # plt.show()

    minIdx = np.argmin(np.exp(np.mean(np.log(all_residuals), axis=0)))
    best_alpha = all_alphas[minIdx]
    # minIdx = np.argmin(np.mean(all_residuals, axis=0))
    # best_alpha = all_alphas[minIdx]

    ret = lasso_lars(X, y, best_alpha, max_iter=max_iter//cv, verbose=verbose, X_test=X_test)
    ret.n_iter_ = n_iter
    return ret
