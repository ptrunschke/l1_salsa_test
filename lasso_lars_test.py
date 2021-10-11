import warnings

import numpy as np

from lasso_lars import lasso_lars_cv


class SimpleOperator(object):
    def __init__(self, matrix):
        self.matrix = matrix

    @property
    def shape(self):
        """
        The shape of the operator.
        """
        return self.matrix.shape

    def take_rows(self, _slice):
        """
        Return a new SimpleOperator containing only the rows in `_slice`.
        """
        return SimpleOperator(self.matrix[_slice])

    def drop_rows(self, _slice):
        """
        Return a new SimpleOperator containing all but the rows in `_slice`.
        """
        mask = np.full(self.shape[0], True)
        mask[_slice] = False
        return SimpleOperator(self.matrix[mask])

    def Xc(self, _indices, _coefs):
        """
        Right multiplication with a sparse vector.
        """
        return self.matrix[:, _indices] @ _coefs

    def XTX(self, _index):
        """
        The `_index`-column of the Gramian `X.T@X`.
        """
        return self.matrix.T @ self.matrix[:, _index]

    def XTy(self, _y):
        """
        Left multiplication with a dense vector.
        """
        return _y @ self.matrix

    def error_norm(self, _y, _indices, _coefs):
        return np.linalg.norm(_y - self.matrix[:,_indices]@_coefs)**2


if __name__ == "__main__":
    from numpy.polynomial.legendre import legval
    from numpy.polynomial.hermite_e import hermeval
    from scipy.special import factorial
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LassoLarsCV

    import autoPDB

    # d = 20     # number of dimensions in each mode
    # N = 50     # number of samples
    # d = 20
    d = 70
    N = 200
    # d = 250
    # N = 200
    f = lambda x: 1/(1+25*x**2)

    # domain = [-1,1]
    # domain = [-2,2]
    # domain = [-3,3]
    # domain = [-5,5]
    domain = [-10,10]
    # basisval = legval
    basisval = hermeval
    estimate_optimal_weights = False
    # estimate_optimal_weights = True


    if basisval is legval:
        factors = np.sqrt(2*np.arange(d)+1)
        basisWeights = factors
    elif basisval is hermeval:
        factorials = factorial(np.arange(d, dtype=object), exact=True)
        factors = np.sqrt((1/factorials).astype(float))
        basisWeights = (factorials**3).astype(float)

    if estimate_optimal_weights:
        xs = np.linspace(*domain, num=1000)
        ys = basisval(xs, np.diag(factors)).T
        basisWeights = np.max(abs(ys), axis=0)

    samples = (domain[1]-domain[0])*np.random.rand(N) + domain[0]
    measures = basisval(samples, np.diag(factors)).T
    values = f(samples)

    assert measures.shape == (N,d)
    assert values.shape == (N,)
    assert basisWeights.shape == (d,)

    #NOTE: For large basisWeights 0/basisWeights == nan even though basisWeights != 0.
    model_skl = LassoLarsCV(normalize=False, fit_intercept=False, cv=10).fit(np.nan_to_num(measures / basisWeights[np.newaxis]), values)
    assert not model_skl.normalize and not model_skl.fit_intercept
    print("SKLearn model:")
    print("--------------")
    print(f"Optimal lambda: {model_skl.alpha_:.2e}")
    dofs = np.count_nonzero(model_skl.coef_)
    print(f"DoFs: {dofs}")
    skl_coefs = np.nan_to_num(model_skl.coef_ / basisWeights)
    xs = np.linspace(*domain, num=1000)
    ys = basisval(xs, factors*skl_coefs)
    print(f"Error: {np.linalg.norm(ys - f(xs)):.2e}  {np.max(abs(ys - f(xs))):.2e}")
    print()

    model_skl_normalized = LassoLarsCV(normalize=True, fit_intercept=False, cv=10).fit(np.nan_to_num(measures / basisWeights[np.newaxis]), values)
    assert model_skl_normalized.normalize and not model_skl_normalized.fit_intercept
    print("SKLearn model (normalized):")
    print("---------------------------")
    print(f"Optimal lambda: {model_skl_normalized.alpha_:.2e}")
    dofs = np.count_nonzero(model_skl_normalized.coef_)
    print(f"DoFs: {dofs}")
    skl_normalized_coefs = np.nan_to_num(model_skl_normalized.coef_ / basisWeights)
    xs = np.linspace(*domain, num=1000)
    ys = basisval(xs, factors*skl_normalized_coefs)
    print(f"Error: {np.linalg.norm(ys - f(xs)):.2e}  {np.max(abs(ys - f(xs))):.2e}")
    print()


    X = SimpleOperator(np.nan_to_num(measures / basisWeights[np.newaxis]))
    own_model = lasso_lars_cv(X, values, cv=10, max_iter=5000)
    print("Own model:")
    print("----------")
    print(f"Optimal lambda: {own_model.alpha_:.2e}")
    dofs = np.count_nonzero(own_model.coef_)
    print(f"DoFs: {dofs}")
    lsa = set(own_model.active_)
    skla = set(np.nonzero(skl_coefs)[0])
    print(f"Needless indices: {lsa - skla}")
    print(f"Missing indices: {skla - lsa}")
    print(f"Coefficient range: [{np.min(abs(own_model.coef_ / basisWeights[own_model.active_])):.2e}, {np.max(abs(own_model.coef_ / basisWeights[own_model.active_])):.2e}]")
    coefs = np.zeros(d)
    coefs[own_model.active_] = own_model.coef_
    coefs = np.nan_to_num(coefs / basisWeights)
    xs = np.linspace(*domain, num=1000)
    ys = basisval(xs, factors*coefs)
    print(f"Error: {np.linalg.norm(ys - f(xs)):.2e}  {np.max(abs(ys - f(xs))):.2e}")
    print()

    assert measures.shape == (N,d)
    infty_norms = np.max(abs(measures), axis=0)
    assert infty_norms.shape == (d,)
    X_norm = SimpleOperator(np.nan_to_num(measures / infty_norms[np.newaxis]))
    own_model_norm = lasso_lars_cv(X_norm, values, cv=10, max_iter=5000)
    print("Own model (normalized):")
    print("-----------------------")
    print(f"Optimal lambda: {own_model_norm.alpha_:.2e}")
    dofs = np.count_nonzero(own_model_norm.coef_)
    print(f"DoFs: {dofs}")
    lsa = set(own_model_norm.active_)
    skla = set(np.nonzero(skl_normalized_coefs)[0])
    print(f"Needless indices: {lsa - skla}")
    print(f"Missing indices: {skla - lsa}")
    print(f"Coefficient range: [{np.min(abs(own_model_norm.coef_ / infty_norms[own_model_norm.active_])):.2e}, {np.max(abs(own_model_norm.coef_ / infty_norms[own_model_norm.active_])):.2e}]")
    coefs_norm = np.zeros(d)
    coefs_norm[own_model_norm.active_] = own_model_norm.coef_
    coefs_norm = np.nan_to_num(coefs_norm / infty_norms)
    xs = np.linspace(*domain, num=1000)
    ys = basisval(xs, factors*coefs_norm)
    print(f"Error: {np.linalg.norm(ys - f(xs)):.2e}  {np.max(abs(ys - f(xs))):.2e}")

    infty_norms = np.max(abs(measures), axis=0)
    print(f"{np.linalg.norm(infty_norms):.2e} \u2192 1")
    infty_norms /= np.linalg.norm(infty_norms)
    X_norm = SimpleOperator(np.nan_to_num(measures / infty_norms[np.newaxis]))
    own_model_norm2 = lasso_lars_cv(X_norm, values, cv=10, max_iter=5000)
    print("Own model (normalized(2)):")
    print("-----------------------")
    print(f"Optimal lambda: {own_model_norm2.alpha_:.2e}")
    dofs = np.count_nonzero(own_model_norm2.coef_)
    print(f"DoFs: {dofs}")
    lsa = set(own_model_norm2.active_)
    skla = set(np.nonzero(skl_normalized_coefs)[0])
    print(f"Needless indices: {lsa - skla}")
    print(f"Missing indices: {skla - lsa}")
    print(f"Coefficient range: [{np.min(abs(own_model_norm2.coef_ / infty_norms[own_model_norm2.active_])):.2e}, {np.max(abs(own_model_norm2.coef_ / infty_norms[own_model_norm2.active_])):.2e}]")
    coefs_norm = np.zeros(d)
    coefs_norm[own_model_norm2.active_] = own_model_norm2.coef_
    coefs_norm = np.nan_to_num(coefs_norm / infty_norms)
    xs = np.linspace(*domain, num=1000)
    ys = basisval(xs, factors*coefs_norm)
    print(f"Error: {np.linalg.norm(ys - f(xs)):.2e}  {np.max(abs(ys - f(xs))):.2e}")

    ms = basisval(xs, np.diag(factors)).T
    cs = np.linalg.lstsq(measures, values, rcond=None)[0]

    tab20 = plt.cm.tab20.colors
    xs = np.linspace(*domain, num=1000)
    plt.plot(xs, f(xs), 'k', lw=2)
    plt.plot(xs, ms@cs, color='tab:red', label='ls')
    plt.plot(xs, ms@skl_coefs, color=tab20[0], label='skl')
    plt.plot(xs, ms@skl_normalized_coefs, color=tab20[1], label='skl+norm')
    plt.plot(xs, ms@coefs, color=tab20[4], label='lasso_lars_cv')
    plt.plot(xs, ms@coefs_norm, color=tab20[5], label='lasso_lars_cv(normalized)')
    plt.plot(samples, values, 'ko')
    plt.ylim(-0.5, 2.5)
    plt.legend()
    plt.show()
