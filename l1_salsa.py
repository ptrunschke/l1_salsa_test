from collections import deque
from collections.abc import Mapping
from numbers import Real
from itertools import zip_longest, takewhile

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
import xerus as xe
from colored import fg, bg, attr
from sklearn.linear_model import LassoLarsCV, RidgeCV
from .lasso_lars import lasso_lars_cv
from .lasso_lars_test import SimpleOperator

from contextlib import contextmanager


@contextmanager
def suspend_profiling(profiler):
    if profiler is not None:
        profiler.disable()
    yield
    if profiler is not None:
        profiler.enable()


import subprocess

sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
diff = subprocess.check_output(["git", "--no-pager", "diff", "--minimal", "HEAD", "--", __file__]).strip()
print("GIT COMMIT:", sha.decode("utf-8"))
if diff:
    print("GIT DIFF:\n", diff.decode("utf-8"))


tensor = lambda arr: xe.Tensor.from_buffer(arr)
is_sequence = lambda obj: hasattr(obj, "__len__") and hasattr(obj, "__getitem__")

countwhile = lambda predicate, iterable: len(list(takewhile(predicate, iterable)))


def split_int_frac(a):
    a = str(float(a))
    i = a.find(".")
    ai, af = a[:i], a[i + 1 :]
    assert a == f"{ai}.{af}"
    return ai, af

def disp_shortest_nonzero(a, min_num_digits=0):  # min_num_digits: minimal number of decimal digits to display
    ai,af = split_int_frac(a)
    if ai != "0":
        if min_num_digits == 0: return ai
        return f"{ai}.{af[:min_num_digits]}"
    num_zero_digits = countwhile(lambda d: d=="0", af)
    num_digits = max(num_zero_digits+1, min_num_digits)
    return f"{ai}.{af[:num_digits]}"

def disp_shortest_unequal(a, b):
    ai,af = split_int_frac(a)
    bi,bf = split_int_frac(b)
    if ai != bi: return disp_shortest_nonzero(a), disp_shortest_nonzero(b)
    num_equal_digits = countwhile(lambda ds: ds[0]==ds[1], zip_longest(af,bf, fillvalue="0"))
    return disp_shortest_nonzero(a, num_equal_digits+1), disp_shortest_nonzero(b, num_equal_digits+1)

def isometry_constant(M, G):
    """
    Computes the supremum sup_v max(abs(M @ v)) / sqrt(v.T @ G @ v).
    """
    es,vs = np.linalg.eigh(G)  # diagonalize G
    assert np.all(es >= 0)
    assert np.allclose(vs*es @ vs.T, G)
    M_tilde = M @ (vs / np.sqrt(es))
    x0 = np.linalg.eigh(M_tilde.T @ M_tilde)[1][:,-1]
    C0 = np.max(abs(M_tilde @ x0))
    res = minimize(lambda x: -np.max(abs(M_tilde @ x))**2/(x @ x), x0)
    x1 = res.x / np.linalg.norm(res.x)
    C1 = np.max(abs(M_tilde @ x1))
    assert C1 >= C0
    return C1


class InternalSolver(object):
    def __init__(self, dimensions, measures, values, weights=None, basisWeights=None, profiler=None):
        assert is_sequence(dimensions)
        assert is_sequence(measures)
        assert is_sequence(values)
        M = len(dimensions)
        d = list(dimensions)
        N,S = np.shape(values)  # this also ensures that values is rectangular
        assert len(measures) == M-1
        assert dimensions[0] == S
        if weights is None:
            weights = np.ones(N)
        else:
            assert np.shape(weights) == (N,)
        if basisWeights is None:
            basisWeights = [np.eye(di) for di in d]
        else:
            assert is_sequence(basisWeights) and len(basisWeights) == M
            for m in range(M):
                assert np.shape(basisWeights[m]) == (d[m],)*2
                assert np.allclose(basisWeights[m], np.transpose(basisWeights[m]))
        for m in range(1,M):
            assert isinstance(measures[m-1], xe.Tensor)
            Nm,dm = measures[m-1].dimensions
            assert dm == d[m]
            if N is None:
                N =  Nm
            else:
                assert Nm == N
        for i in range(N):
            assert is_sequence(values[i])
            for j in range(S):
                assert isinstance(values[i][j], Real)

        self.basisWeightFactors = []
        for m in range(1,M):
            # || y - measures @ x ||^2 + lmbda x.T @ basisWeights @ x
            # ==> x --> z = L.T @ x
            # ==> x = inv(L.T) @ z
            # ==> measures @ x == measures @ inv(L.T) @ z
            # basisWeights[m] = L@L.T; basisWeights[m].shape == (d,d); measures[m-1].shape == (n,d); --> (L.T @ measures[m-1].T).T == measures[m-1] @ L
            # measures @ invLT
            L = np.linalg.cholesky(basisWeights[m])
            invLT = solve_triangular(L.T, np.eye(d[m]))
            meas = measures[m-1].to_ndarray() @ invLT
            assert np.all(np.isfinite(meas))
            measures[m-1] = xe.Tensor.from_ndarray(meas)
            self.basisWeightFactors.append(invLT)
            basisWeights[m] = np.eye(d[m])  #TODO: ...

        # ADF parameters
        # minDecrease: minimum decrease of the residual relative to the highest residual in the preceding `trackingPeriodLength` iterations
        self.minDecrease = 1e-3
        # maxIterations: maximum number of iterations to perform
        self.maxIterations = 1000

        self.trackingPeriodLength = 10
        self.targetResidual = 1e-8
        self.controlSetFraction = 0.1

        # SALSA parameters
        # maxRanks: user defined list of maximal ranks
        self.maxRanks = [np.inf]*(M-1)
        # maxTheoreticalRanks: list of maximal possible ranks for a tensor with dimensions d
        self.maxTheoreticalRanks = np.minimum(np.cumprod(d[:-1], dtype=object), np.cumprod(d[:0:-1], dtype=object)[::-1])
        # kmin: number of additional ranks used
        self.kmin = 2
        # smin: factor to decide of if singular value is active or inactive
        self.smin = None  # set after everything other variables are initialized
        self.minSMinFactor = 1e-5

        # LASSO parameters
        self.microstepSolver = "lasso"  # The solver to use for the microsteps.

        self.unstableIsometryConstant = 1

        # Initialize tensor as a perturbation of the mean of the values.
        # Perturbation has a magnitude of 0.2% of the norm of mean.
        self.x = xe.TTTensor.random(d, [1]*(M-1))
        core = np.mean(values, axis=0)
        self.x.set_component(0, tensor(core.reshape(1,S,1)))
        for m in range(1,M):
            self.x.set_component(m, xe.Tensor.dirac([1,d[m],1], [0]*3))
        smin = 0.2*xe.frob_norm(self.x)
        kick = xe.TTTensor.random(self.x.dimensions, [self.kmin]*(M-1))
        kick = kick / xe.frob_norm(kick)
        self.x += 0.01*smin * kick
        self.x.canonicalize_left()
        assert np.all(self.x.ranks() == np.minimum([1+self.kmin]*(M-1), self.maxTheoreticalRanks)), f"{self.x.ranks()} != {np.minimum([1+self.kmin]*(M-1), self.maxTheoreticalRanks)}"

        self.measures = measures
        self.values = values
        self.basisWeights = basisWeights

        self.profiler = profiler

    def _leftStack_pop(self):
        self.leftStack.pop()
        self.leftRegularizationStack.pop()

    def _leftStack_push(self, pos):
        assert self.x.corePosition > pos
        xpos = self.x.get_component(pos).to_ndarray()

        lpos = self.leftStack[-1]
        if pos == 0:
            tmp = np.einsum('nl,lsr -> nrs', lpos, xpos)  # the first mode has no measure
        else:
            mpos = self.measures[pos-1].to_ndarray()  # -1 --> the first mode has no measure
            tmp = np.einsum("nls,ler,ne -> nrs", lpos, xpos, mpos)
        self.leftStack.append(tmp)

        lpos = self.leftRegularizationStack[-1]
        mpos = self.basisWeights[pos]
        tmp = np.einsum("lm,ler,ef,mfs -> rs", lpos, xpos, mpos, xpos)
        assert np.all(np.isfinite(tmp))
        self.leftRegularizationStack.append(tmp)

    def _rightStack_pop(self):
        self.rightStack.pop()
        self.rightRegularizationStack.pop()

    def _rightStack_push(self, pos):
        assert self.x.corePosition < pos
        xpos = self.x.get_component(pos).to_ndarray()

        mpos = self.measures[pos-1].to_ndarray()  # -1 --> the first mode has no measure
        rpos = self.rightStack[-1]
        tmp = np.einsum("ne,ler,nr -> nl", mpos, xpos, rpos)
        self.rightStack.append(tmp)

        rpos = self.rightRegularizationStack[-1]
        mpos = self.basisWeights[pos]
        tmp = np.einsum("ler,ef,mfs,rs -> lm", xpos, mpos, xpos, rpos)
        assert np.all(np.isfinite(tmp))
        self.rightRegularizationStack.append(tmp)

    def _adapt_rank(self, U, S, Vt):
        assert np.all(np.diag(S.to_ndarray()) > 0), np.diag(S.to_ndarray())
        i,j,k = xe.indices(3)
        eU = U.order()-1
        eV = Vt.order()-1

        Um = np.prod(U.dimensions[:-1])
        Un = U.dimensions[-1]
        Vtm = Vt.dimensions[0]
        Vtn = np.prod(Vt.dimensions[1:])
        # Check dimensions
        assert Un == S.dimensions[0] == S.dimensions[1] == Vtm
        # The rank can only be increased when the dimensions of U and Vt allow it.
        #NOTE: When called from _move_core_left or _move_core_right the condition Um > Un and Vtm < Vtn
        #      is guaranteed by the condition full_rank <= maxTheoreticalRanks.
        assert Um > Un and Vtm < Vtn

        # S (diagonal matrix)
        S_new = xe.Tensor(S.dimensions + np.array([1,1]))
        S_new.offset_add(S, [0,0])
        # At the last entry: simply set a value at 1% of the singular value threshold smin
        S_new[S.dimensions[0],S.dimensions[1]] = 0.01 * self.smin
        assert np.all(np.diag(S_new.to_ndarray()) > 0), f"{np.diag(S_new.to_ndarray())} {self.smin}"

        tmp = xe.Tensor()

        # U (left singular vectors)
        U_new = xe.Tensor(U.dimensions + np.array([0]*eU + [1]))
        U_new.offset_add(U, [0]*eU + [0])
        # create new row (U_row) in the complement of U
        # U_row = xe.Tensor.ones(U.dimensions[:eU])
        U_row = xe.Tensor.random(U.dimensions[:eU])  # == U.dimensions[:-1]
        # U_row /= U_row.frob_norm()  #TODO: bug in xerus (fixed, not compiled)
        U_row = U_row / U_row.frob_norm()
        # project U_row onto the complement of U and normalize it (twice for numerical stability)
        # U_row(i^eU) << U_row(i^eU) - U(i^eU,k) * U(j^eU,k) * U_row(j^eU)  #TODO: bug in xerus
        tmp(i^eU) << U(i^eU,k) * U(j^eU,k) * U_row(j^eU)
        U_row -= tmp
        # U_row /= U_row.frob_norm()  #TODO: bug in xerus (fixed, not compiled)
        U_row = U_row / U_row.frob_norm()
        tmp(i^eU) << U(i^eU,k) * U(j^eU,k) * U_row(j^eU)
        U_row -= tmp
        # U_row /= U_row.frob_norm()  #TODO: bug in xerus (fixed, not compiled)
        U_row = U_row / U_row.frob_norm()
        # add tensors
        U_row.reinterpret_dimensions(U_row.dimensions + [1])
        U_new.offset_add(U_row, [0]*eU + [U.dimensions[-1]])

        # Vt (right singular vectors)
        Vt_new = xe.Tensor(Vt.dimensions + np.array([1] + [0]*eV))
        Vt_new.offset_add(Vt, [0] + [0]*eV)
        # Vt_col = xe.Tensor.ones(Vt.dimensions[1:])
        Vt_col = xe.Tensor.random(Vt.dimensions[1:])
        # Vt_col /= Vt_col.frob_norm()  #TODO: bug in xerus (fixed, not compiled)
        Vt_col = Vt_col / Vt_col.frob_norm()
        # Vt_col(i^eV) << Vt_col(i^eV) - Vt(k,i^eV) * Vt(k,j^eV) * Vt_col(j^eV)  #TODO: bug in xerus
        tmp(i^eV) << Vt(k,i^eV) * Vt(k,j^eV) * Vt_col(j^eV)
        Vt_col -= tmp
        # Vt_col /= Vt_col.frob_norm()  #TODO: bug in xerus (fixed, not compiled)
        Vt_col = Vt_col / Vt_col.frob_norm()
        tmp(i^eV) << Vt(k,i^eV) * Vt(k,j^eV) * Vt_col(j^eV)
        Vt_col -= tmp
        # Vt_col /= Vt_col.frob_norm()  #TODO: bug in xerus (fixed, not compiled)
        Vt_col = Vt_col / Vt_col.frob_norm()
        Vt_col.reinterpret_dimensions([1] + Vt_col.dimensions)
        Vt_new.offset_add(Vt_col, [Vt.dimensions[0]] + [0]*eV)

        assert np.all(np.isfinite(U_new.to_ndarray())) and np.all(np.isfinite(S_new.to_ndarray())) and np.all(np.isfinite(Vt_new.to_ndarray()))
        return U_new, S_new, Vt_new

    def _move_core_left(self, adapt):
        pos = self.x.corePosition
        if not 0 < pos:
            raise ValueError(f"core at position {pos} can not move in direction 'left' in tensor of order {self.x.order()}")
        i,j,l,r = xe.indices(4)
        U,S,Vt = (xe.Tensor() for _ in range(3))

        # move core
        right = self.x.get_component(pos)    # old core
        core  = self.x.get_component(pos-1)  # new core

        right_norm = xe.frob_norm(right)
        right = right / right_norm
        (U(i,l), S(l,r), Vt(r,j^2)) << xe.SVD(right(i,j^2))
        S = S * right_norm

        SIGN = xe.Tensor.identity(S.dimensions)
        for k in range(S.dimensions[0]):
            if S[k,k] < 0:
                S[k,k] = -S[k,k]
                SIGN[k,k] = -1
        U(l,r) << U(l,i) * SIGN(i,r)
        assert np.all(np.diag(S.to_ndarray()) > 0), np.diag(S.to_ndarray())

        right = Vt
        i,j,l,r = xe.indices(4)
        core(i^2,j) << core(i^2,l) * U(l,j)

        # adapt the rank (pos-1)--(pos) i.e. self.x.ranks()[pos-1]
        rank = np.count_nonzero(np.diag(S.to_ndarray()) > self.smin)
        full_rank = min(min(rank, self.maxRanks[pos-1])+self.kmin, self.maxTheoreticalRanks[pos-1])
        if rank >= self.maxRanks[pos-1]:
            self.maxRanksReached.add(pos-1)
        else:
            self.maxRanksReached.discard(pos-1)
        if adapt:
            for r in range(S.dimensions[0], full_rank):
                core,S,right = self._adapt_rank(core,S,right)
        assert np.all(np.diag(S.to_ndarray()) > 0), np.diag(S.to_ndarray())

        i,j,l,r = xe.indices(4)
        core(i^2,j) << core(i^2,l) * S(l,j)

        assert np.all(np.isfinite(core.to_ndarray())) and np.all(np.isfinite(right.to_ndarray()))
        self.x.set_component(pos-1, core)
        self.x.set_component(pos,   right)
        self.x.assume_core_position(pos-1)

        self._rightStack_push(pos)
        self._leftStack_pop()

        # compute right singular values
        # self.rightSingularValues = np.diag(S.to_ndarray())
        self.rightSingularValues = [S[i,i] for i in range(S.dimensions[0])]
        assert np.allclose([S[i,i] for i in range(S.dimensions[0])], np.diag(S.to_ndarray()))
        assert np.all(np.asarray(self.rightSingularValues) >= 0), np.asarray(self.rightSingularValues)
        self.singularValues[pos-1] = np.array(self.rightSingularValues)
        # compute left singular values
        if 1 < pos:
            i,j,l,r = xe.indices(4)

            core_norm = xe.frob_norm(core)
            core = core / core_norm  #TODO: prevent ...
            (U(i,l), S(l,r), Vt(r,j^2)) << xe.SVD(core(i,j^2))
            S = S * core_norm

            SIGN = xe.Tensor.identity(S.dimensions)
            for k in range(S.dimensions[0]):
                if S[k,k] < 0:
                    S[k,k] = -S[k,k]
                    SIGN[k,k] = -1
            U(l,r) << U(l,i) * SIGN(i,r)
            assert np.all(np.diag(S.to_ndarray()) > 0), np.diag(S.to_ndarray())

            left = self.x.get_component(pos-2)
            left(i^2,j) << left(i^2,l) * U(l,j)
            core = Vt

            # adapt the rank (pos-2)--(pos-1) i.e. self.x.ranks()[pos-2]
            rank = np.count_nonzero(np.diag(S.to_ndarray()) > self.smin)
            full_rank = min(min(rank, self.maxRanks[pos-2])+self.kmin, self.maxTheoreticalRanks[pos-2])
            if adapt:
                for r in range(S.dimensions[0], full_rank):
                    left,S,core = self._adapt_rank(left,S,core)
            assert np.all(np.diag(S.to_ndarray()) > 0), np.diag(S.to_ndarray())

            core(i,j^2) << S(i,l) * core(l,j^2)

            assert np.all(np.isfinite(left.to_ndarray())) and np.all(np.isfinite(core.to_ndarray()))
            self.x.set_component(pos-2, left)
            self.x.set_component(pos-1, core)
            # the component left of the core (pos-2) changed and therefore the corresponding stack entry has to be recomputed
            self._leftStack_pop()
            self._leftStack_push(pos-2)
        else:
            S = xe.Tensor.from_ndarray(np.array([[xe.frob_norm(core)]]))
        # self.leftSingularValues = np.diag(S.to_ndarray())
        self.leftSingularValues = [S[i,i] for i in range(S.dimensions[0])]
        assert np.allclose([S[i,i] for i in range(S.dimensions[0])], np.diag(S.to_ndarray()))

    def _move_core_right(self, adapt):
        pos = self.x.corePosition
        if not pos < self.x.order()-1:
            raise ValueError(f"core at position {pos} can not move in direction 'right' in tensor of order {self.x.order()}")
        i,j,l,r = xe.indices(4)
        U,S,Vt = (xe.Tensor() for _ in range(3))

        # move core
        left = self.x.get_component(pos)    # old core
        core = self.x.get_component(pos+1)  # new core

        left_norm = xe.frob_norm(left)
        left = left / left_norm  # prevent rounding errors
        (U(i^2,l), S(l,r), Vt(r,j)) << xe.SVD(left(i^2,j))
        S = S * left_norm

        SIGN = xe.Tensor.identity(S.dimensions)
        for k in range(S.dimensions[0]):
            if S[k,k] < 0:
                S[k,k] = -S[k,k]
                SIGN[k,k] = -1
        Vt(l,r) << SIGN(l,i) * Vt(i,r)
        assert np.all(np.diag(S.to_ndarray()) > 0), f"{pos} {np.diag(S.to_ndarray())}"

        left = U
        i,j,l,r = xe.indices(4)  #TODO: a bug in xerus makes this somehow necessary
        core(i,j^2) << Vt(i,r) * core(r,j^2)

        # adapt the rank (pos)--(pos+1) i.e. self.x.ranks()[pos]
        rank = np.count_nonzero(np.diag(S.to_ndarray()) > self.smin)
        full_rank = min(min(rank, self.maxRanks[pos])+self.kmin, self.maxTheoreticalRanks[pos])
        if rank >= self.maxRanks[pos]:
            self.maxRanksReached.add(pos)
        else:
            self.maxRanksReached.discard(pos)
        if adapt:
            for r in range(S.dimensions[0], full_rank):
                left,S,core = self._adapt_rank(left,S,core)
        assert np.all(np.diag(S.to_ndarray()) > 0), np.diag(S.to_ndarray())

        i,j,l,r = xe.indices(4)  #TODO: a bug in xerus makes this somehow necessary
        core(i,j^2) << S(i,l) * core(l,j^2)

        assert np.all(np.isfinite(left.to_ndarray())) and np.all(np.isfinite(core.to_ndarray()))
        self.x.set_component(pos,   left)
        self.x.set_component(pos+1, core)
        self.x.assume_core_position(pos+1)

        self._leftStack_push(pos)
        self._rightStack_pop()

        # compute left singular values
        # self.leftSingularValues = np.diag(S.to_ndarray())
        self.leftSingularValues = [S[i,i] for i in range(S.dimensions[0])]
        assert np.allclose([S[i,i] for i in range(S.dimensions[0])], np.diag(S.to_ndarray()))
        assert np.all(np.asarray(self.leftSingularValues) >= 0), np.asarray(self.leftSingularValues)
        self.singularValues[pos] = np.array(self.leftSingularValues)
        # compute right singular values
        if pos < self.x.order()-2:
            i,j,l,r = xe.indices(4)

            core_norm = xe.frob_norm(core)
            core = core / core_norm  #TODO: prevent...
            (U(i^2,l), S(l,r), Vt(r,j)) << xe.SVD(core(i^2,j))
            S = S * core_norm

            SIGN = xe.Tensor.identity(S.dimensions)
            for k in range(S.dimensions[0]):
                if S[k,k] < 0:
                    S[k,k] = -S[k,k]
                    SIGN[k,k] = -1
            Vt(l,r) << SIGN(l,i) * Vt(i,r)
            assert np.all(np.diag(S.to_ndarray()) > 0), np.diag(S.to_ndarray())

            core = U
            right = self.x.get_component(pos+2)
            right(i,j^2) << Vt(i,l) * right(l,j^2)

            # adapt the rank (pos+1)--(pos+2) i.e. self.x.ranks()[pos+1]
            rank = np.count_nonzero(np.diag(S.to_ndarray()) > self.smin)
            full_rank = min(min(rank, self.maxRanks[pos+1])+self.kmin, self.maxTheoreticalRanks[pos+1])
            assert 0 <= rank <= S.dimensions[0], f"0 <= {rank} <= {S.dimensions[0]}"
            if adapt:
                for r in range(S.dimensions[0], full_rank):
                    core,S,right = self._adapt_rank(core,S,right)
            assert np.all(np.diag(S.to_ndarray()) > 0), np.diag(S.to_ndarray())

            core(i^2,j) << core(i^2,l) * S(l,j)

            assert np.all(np.isfinite(core.to_ndarray())) and np.all(np.isfinite(right.to_ndarray()))
            self.x.set_component(pos+1, core)
            self.x.set_component(pos+2, right)
            # the component right of the core (pos+2) changed and therefore the corresponding stack entry has to be recomputed
            self._rightStack_pop()
            self._rightStack_push(pos+2)
        else:
            S = xe.Tensor.from_ndarray(np.array([[xe.frob_norm(core)]]))
        # self.rightSingularValues = np.diag(S.to_ndarray())
        self.rightSingularValues = [S[i,i] for i in range(S.dimensions[0])]
        assert np.allclose([S[i,i] for i in range(S.dimensions[0])], np.diag(S.to_ndarray()))

    def residual(self, set=slice(None)):  # np.arange(len(self.values))
        if not isinstance(set, slice):
            assert np.ndim(set) == 1
        pos = self.x.corePosition

        lpos = self.leftStack[-1]
        rpos = self.rightStack[-1]
        xpos = self.x.get_component(pos).to_ndarray()

        if pos == 0:
            res = np.einsum("nl,lsr,nr -> ns", lpos, xpos, rpos)
        else:
            mpos = self.measures[pos-1].to_ndarray()  # -1 --> the first mode has no measure
            res = np.einsum("nls,ler,ne,nr -> ns", lpos, xpos, mpos, rpos)

        valueNorm = np.linalg.norm(self.values[set])
        return np.linalg.norm(res[set] - self.values[set])/valueNorm

    def solve_local(self):
        pos = self.x.corePosition
        l,e,r = self.x.get_component(pos).dimensions
        ler = l*e*r

        # create local operator and rhs
        # factors for the operator
        lOp = self.leftStack[-1][self.trainingSet]
        rOp = self.rightStack[-1][self.trainingSet]
        # factors for the weights
        lW = np.sqrt(np.diag(self.leftRegularizationStack[-1]))
        eW = np.sqrt(np.diag(self.basisWeights[pos]))
        rW = np.sqrt(np.diag(self.rightRegularizationStack[-1]))

        if pos == 0:
            assert e == np.shape(self.values)[1]
            opArr = np.einsum('nl,es,nr -> nsler', lOp, np.eye(e), rOp).reshape(-1, ler)
        else:
            eOp = self.measures[pos-1].to_ndarray()[self.trainingSet]  # -1 --> the first mode has no measure
            opArr = np.einsum('nls,ne,nr -> nsler', lOp, eOp, rOp).reshape(-1, ler)

        basisWeights = np.einsum('l,e,r -> ler', lW, eW, rW).reshape(-1)
        op_test = opArr/basisWeights[np.newaxis]
        assert np.all(np.isfinite(op_test))
        op_test = SimpleOperator(op_test)

        # use only the training set for optimization
        assert isinstance(self.trainingSet, slice)
        rhs_size = self.trainingSet_size * self.values.shape[1]
        assert op_test.shape == (rhs_size, ler), f"NOT {op_test.shape} == {(rhs_size, ler)}"
        values = self.values[self.trainingSet].reshape(rhs_size)

        core = self.x.get_component(pos).to_ndarray()
        core = np.nan_to_num(core)
        core_norm = np.max(abs(core))
        core[abs(core) < 1e-16*core_norm] = 0

        if self.microstepSolver == "lasso":
            model = lasso_lars_cv(op_test, values, cv=10)
            assert model.alpha_ >= 0
            # assert len(model.active_) > 0
            if len(model.active_) > 0:  #TODO: quick fix...
                core = xe.Tensor([ler])
                assert core.is_sparse()
                for idx,val in zip(model.active_, model.coef_):
                    core[idx] = val / basisWeights[idx]
                core.reinterpret_dimensions([l,e,r])
            else:
                model.n_iter_ = 0

            npyCore = core.to_ndarray()
            min_nonzero = np.min(np.abs(npyCore)[np.abs(npyCore) != 0])
            max_nonzero = np.max(np.abs(npyCore)[np.abs(npyCore) != 0])
            print()
            print(f"  Nonzeros: {np.count_nonzero(self.x.get_component(pos).to_ndarray())} -> {len(model.active_)}      ({np.count_nonzero(np.reshape(self.x.get_component(pos).to_ndarray(), -1)[model.active_])} stable | {npyCore.size} total)")
            print(f"  Norm:     {np.linalg.norm(npyCore):.2e}    (min: {min_nonzero:.2e}  |  max: {max_nonzero:.2e}  |  lambda: {model.alpha_:.2e})")
            print(f"  LARS iterations: {model.n_iter_}")

        elif self.microstepSolver == "sklearn.lasso":
            #NOTE: We can not use `fit_intercept` since we dont know if the local basis contains a constant function.
            model = LassoLarsCV(normalize=False, fit_intercept=False, cv=10).fit(opArr/basisWeights[np.newaxis], values)
            assert not model.normalize and not model.fit_intercept
            if len(model.active_) != 0:  #TODO: quick fix...
                coefs = model.coef_ / basisWeights
                core = xe.Tensor.from_buffer(coefs)
                core.reinterpret_dimensions([l,e,r])
                core.use_sparse_representation()
            else:
                model.n_iter_ = 0

            npyCore = core.to_ndarray()
            min_nonzero = np.min(np.abs(npyCore)[np.abs(npyCore) != 0])
            max_nonzero = np.max(np.abs(npyCore)[np.abs(npyCore) != 0])
            print()
            print(f"  Nonzeros: {np.count_nonzero(self.x.get_component(pos).to_ndarray())} -> {len(model.active_)}      ({np.count_nonzero(self.x.get_component(pos).to_ndarray().reshape(-1)[model.active_])} stable | {npyCore.size} total)")
            print(f"  Norm:     {np.linalg.norm(npyCore):.2e}    (min: {min_nonzero:.2e}  |  max: {max_nonzero:.2e}  |  lambda: {model.alpha_:.2e})")
            print(f"  LARS iterations: {model.n_iter_}")

        elif self.microstepSolver == "sklearn.ridge":
            gram = np.einsum('lk,ef,rs -> lerkfs', self.leftRegularizationStack[-1], self.basisWeights[pos], self.rightRegularizationStack[-1])
            gram /= len(self.basisWeights[pos])
            gram.shape = ler,ler
            es,vs = np.linalg.eigh(gram)
            es[es < 1e-16] = 1e-16
            # v.T @ gram @ v = v.T @ vs @ np.diag(es) @ vs.T @ v = w.T @ w    (w = np.diag(np.sqrt(es)) @ vs.T @ v --> v = vs @ np.diag(1/np.sqrt(es)) @ w)
            # L @ v = L @ vs / np.sqrt(es) @ w = L @ scale @ w
            scale = vs / np.sqrt(es)
            model = RidgeCV(alphas=np.logspace(-10,1,23), normalize=False, fit_intercept=False, cv=10).fit(opArr @ scale, values)
            assert not model.normalize and not model.fit_intercept
            coefs = scale @ model.coef_
            model.active_ = np.nonzero(coefs)[0]
            core = xe.Tensor.from_buffer(coefs)
            core.reinterpret_dimensions([l,e,r])

            npyCore = core.to_ndarray()
            min_nonzero = np.min(np.abs(npyCore)[np.abs(npyCore) != 0])
            max_nonzero = np.max(np.abs(npyCore)[np.abs(npyCore) != 0])
            print()
            print(f"  Relative Gram decomposition error: {np.linalg.norm(vs*es @ vs.T - gram)/np.linalg.norm(gram):.2e}")
            print(f"  Regularization norm: {np.sqrt(coefs.T @ gram @ coefs):.2e}")
            print(f"  Norm:     {np.linalg.norm(npyCore):.2e}    (min: {min_nonzero:.2e}  |  max: {max_nonzero:.2e}  |  lambda: {model.alpha_:.2e})")

        else:
            raise NotImplementedError(f"Microstep solver {self.microstepSolver} is not defined")

        self.lambdas[pos] = model.alpha_
        self.weightedNorms[pos] = len(model.active_)/ler

        # set component
        self.x.set_component(pos, core)


    def print_parameters(self):
        parameters = ["minDecrease", "maxIterations", "trackingPeriodLength", "targetResidual", "controlSetFraction",
                      "maxRanks", "kmin", "minSMinFactor", "unstableIsometryConstant", "microstepSolver"]
        max_param_len = max(max(len(p) for p in parameters), 18)  # 18 == len("isometry_constants")
        tab = " "*2
        print("-"*125)
        print(f"{tab}dimensions = {' '*(max_param_len-10)}{self.x.dimensions}")
        print(f"{tab}initial_ranks = {' '*(max_param_len-13)}{self.x.ranks()}")
        print(f"{tab}num_samples = {' '*(max_param_len-11)}{len(self.values)}")
        print("-"*125)
        values = [getattr(self, p) for p in parameters]
        for param,value in zip(parameters, values):
            space = " "*(max_param_len-len(param))
            print(f"{tab}{param} = {space}{value}")
        print("-"*125)
        Cs = []
        for pos in range(1, self.x.order()):
            Cs.append(isometry_constant(self.measures[pos-1].to_ndarray(), self.basisWeights[pos]))
        alerted = lambda C: f"{C:.2e}" if C < self.unstableIsometryConstant else f"{fg('dark_red_2')}{C:.2e}{attr('reset')}"
        Cs_dump = "[" + ", ".join(alerted(C) for C in Cs) + "]"
        print(f"{tab}isometry_constants = {' '*(max_param_len-18)}"+Cs_dump)
        print("-"*125)

    def initialize(self):
        M = self.x.order()
        N = np.shape(self.values)[0]

        # build stacks
        #NOTE: The stacks need to be build here since they could be corrupted otherwise.
        self.leftStack = [ np.ones([N,1]) ]
        self.rightStack = [ np.ones([N,1]) ]
        self.leftRegularizationStack = [ np.ones([1,1]) ]
        self.rightRegularizationStack = [ np.ones([1,1]) ]
        for pos in reversed(range(1,M)):
            self._rightStack_push(pos)

        # compute left and right singular value arrays
        i,j,l,r = xe.indices(4)
        U,S,Vt = (xe.Tensor() for _  in range(3))
        core = self.x.get_component(0)
        self.leftSingularValues = [xe.frob_norm(core)]
        (U(i^2,l), S(l,r), Vt(r,j)) << xe.SVD(core(i^2,j))
        self.rightSingularValues = [S[i,i] for i in range(S.dimensions[0])]

        # compute SALSA parameters
        valueNorm = np.linalg.norm(self.values)
        maxResSqrtRes = (lambda res: max(res, np.sqrt(res))) (self.residual())
        # self.omega = maxResSqrtRes
        self.smin = 0.2*min(maxResSqrtRes, self.residual())
        # In this way the residual on the active part is equivalent to the entire residual.
        #     norm(A*round(x)-b)/norm(b) <= norm(A*x-b)/norm(b) + norm(A*dx)/norm(b) = residual + norm(A*dx)/norm(b)
        #                                <= residual + norm(A)/norm(b)*norm(dx)
        #                                <= residual + norm(A)/norm(b)*(kmin*0.2*residual) = (1+0.2*kmin*norm(A)/norm(b))*residual

        trainingSet_size = int((1-self.controlSetFraction)*N)
        validationSet_size = N - trainingSet_size
        assert trainingSet_size > 0 and validationSet_size > 0
        self.trainingSet = slice(None,trainingSet_size)    # [:trainingSet_size]
        self.trainingSet_size = trainingSet_size
        self.validationSet = slice(trainingSet_size,None)  # [trainingSet_size:]
        self.validationSet_size = validationSet_size

        self.maxRanksReached = set()
        self.lambdas = [None]*M
        self.weightedNorms = [None]*M  #TODO: rename
        self.singularValues = [None]*(M-1)

    def check(self, pos):
        # check internal state
        assert pos == self.x.corePosition, f"{pos} != {self.x.corePosition}"
        assert len(self.leftStack) == pos+1
        assert len(self.rightStack) == self.x.order()-pos

    def frac_ranks(self):
        assert len(self.singularValues) == self.x.order()-1
        for i in range(self.x.order()-1):
            assert len(self.singularValues[i]) == self.x.rank(i)
        ranks = []
        for ss in self.singularValues:
            assert np.all(ss >= 0), ss
            rank = np.count_nonzero(ss >= self.smin)
            if rank < len(ss):
                inactive = ss[rank]/self.smin
                assert 0 <= inactive < 1
                rank += inactive
            ranks.append(rank)
        xranks = []
        for r,xr in zip(ranks, self.x.ranks()):
            if xr <= int(r)+self.kmin:
                xranks.append(f"{fg('grey_30')}{xr}{attr('reset')}")
            else:
                xranks.append(f"{fg('dark_red_2')}{xr}{attr('reset')}")
        return "[" + ", ".join(f"{int(r)}{fg('grey_50')}.{int(10*(r-int(r)))}{attr('reset')}/{xr}" for r,xr in zip(ranks, xranks)) + "]"

    def lambda_list(self):
        def disp_float(flt):
            r = f"{flt:.0f}"
            if r[-3:] == "inf":
                r = r[:-3] + "\u221e"
            return r
        with np.errstate(divide='ignore'):
            return "10^[" + ", ".join(disp_float(l) for l in np.rint(np.log10(self.lambdas))) + "]"

    def density(self):
        return "[" + ", ".join(f"{int(100*d+0.5):2d}" for d in self.weightedNorms) + "]%"

    def solve(self):
        print("="*125)
        print(" "*55 + "Running VReSALSA")  # Vectorized Regularized Stabilized Alternating Linear Scheme
        self.print_parameters()

        self.initialize()
        M = self.x.order()
        trainingResiduals = deque(maxlen=self.trackingPeriodLength)  # stores the last X residuals
        validationResiduals = []
        initialResidual = self.residual(self.trainingSet)
        prev_bestValidationResidual = self.residual(self.validationSet)

        itr_str = f"[{{0:{len(str(self.maxIterations))}d}}]  ".format

        adapt = False
        # sweep left -> right
        for pos in range(M-1):
            print('\r' + ' '*(32+2*len(str(M-1))) + '\r', end='')
            print(f"\rSweep left -> right. Position: {pos}/{M-1}", end='')
            self.check(pos)
            self.solve_local()
            self._move_core_right(adapt)
        # sweep right -> left
        for pos in reversed(range(1,M)):
            print('\r' + ' '*(32+2*len(str(M-1))) + '\r', end='')
            print(f"\rSweep right -> left. Position: {pos}/{M-1}", end='')
            self.check(pos)
            self.solve_local()
            self._move_core_left(adapt)
        print('\r' + ' '*(32+2*len(str(M-1))) + '\r', end='')

        trainingResiduals.append(self.residual(self.trainingSet))
        validationResiduals.append(self.residual(self.validationSet))

        bestIteration = 0
        bestX = self.x
        bestValidationResidual = validationResiduals[-1]
        bestValidationResidual_cycle = bestValidationResidual
        nonImprovementCounter = 0

        update_str = lambda prev, new: f"{fg('dark_sea_green_2') if new <= prev+1e-8 else fg('misty_rose_3')}{new:.2e}{attr('reset')}"
        bestTrainingResidual = trainingResiduals[-1]
        trn_str = lambda: update_str(bestTrainingResidual , trainingResiduals[-1])
        val_str = lambda: update_str(bestValidationResidual , validationResiduals[-1])

        minActiveSV = None
        maxInactiveSV = None

        def disp_smin():
            if minActiveSV is None:
                assert maxInactiveSV is None
                return f"{fg('misty_rose_3')}{self.smin:.2e}{attr('reset')}"
            return f"{minActiveSV:.2e} >= {fg('dark_sea_green_2')}{self.smin:.2e}{attr('reset')} >= {maxInactiveSV:.2e}"

        iteration = 0  #NOTE: `iteration` is needed at the end of the method.
        print(itr_str(iteration) + f"Residuals: trn\u2295\u03C9\u2295\u03B1={trn_str()}, val={val_str()}  |  Lambdas: {self.lambda_list()}  |  Density: {self.density()}  |  SMin: {disp_smin()}  |  Ranks: {self.frac_ranks()}")

        adapt = True
        # Keep adapting the ranks during the optimization process.
        # Adding inactive singular values reduces the chance of the algorithm to run into local minima.
        for iteration in range(1,self.maxIterations+1):
            # assert self.omega == None

            # sweep left -> right
            for pos in range(M-1):
                print('\r' + ' '*(32+2*len(str(M-1))) + '\r', end='')
                print(f"\rSweep left -> right. Position: {pos}/{M-1}", end='')
                self.check(pos)
                self.solve_local()
                self._move_core_right(adapt)
            # sweep right -> left
            for pos in reversed(range(1,M)):
                print('\r' + ' '*(32+2*len(str(M-1))) + '\r', end='')
                print(f"\rSweep right -> left. Position: {pos}/{M-1}", end='')
                self.check(pos)
                self.solve_local()
                self._move_core_left(adapt)
            print('\r' + ' '*(32+2*len(str(M-1))) + '\r', end='')

            trainingResiduals.append(self.residual(self.trainingSet))
            validationResiduals.append(self.residual(self.validationSet))

            print(itr_str(iteration) + f"Residuals: trn\u2295\u03C9\u2295\u03B1={trn_str()}, val={val_str()}  |  Lambdas: {self.lambda_list()}  |  Density: {self.density()}  |  SMin: {disp_smin()}  |  Ranks: {self.frac_ranks()}")
            bestTrainingResidual = min(trainingResiduals[-1], bestTrainingResidual)

            # minDecrease should typically be 1e-2: an increase that you care about!
            if validationResiduals[-1] < (1-self.minDecrease)*bestValidationResidual:
                bestIteration = iteration
                bestX = self.x
                prev_bestValidationResidual = bestValidationResidual
                bestValidationResidual = validationResiduals[-1]
                bestTrainingResidual = trainingResiduals[-1]

            if validationResiduals[-1] < self.targetResidual:
                print("Terminating: Minimum residual reached.")
                break

            sweepsPerRank = 3
            minSV = min(np.min(svs) for svs in self.singularValues)
            maxSV = max(np.max(svs) for svs in self.singularValues)
            if self.smin > maxSV:
                minActiveSV = None
                maxInactiveSV = None
                self.smin = maxSV
                assert self.smin > 0
            elif self.smin > minSV:
                minActiveSV = min(np.min(svs[svs>=self.smin], initial=np.inf) for svs in self.singularValues)
                maxInactiveSV = max(np.max(svs[svs<self.smin], initial=-np.inf) for svs in self.singularValues)
                gap = minActiveSV - maxInactiveSV
                assert np.isfinite(gap) and gap > 0
                self.smin = max(self.smin - gap/sweepsPerRank, maxInactiveSV)
                assert self.smin > 0
            else:  # self.smin <= minSV
                self.smin = minSV  # possibility 1
                # self.smin += (minSV - self.smin)/2  # possibility 2

            if self.smin < self.minSMinFactor * bestValidationResidual:
                #TODO: Truncate singular values that are below this threshold! (so that they can be added again by move_core)
                #      But this requires a working case for self.smin < minSV.
                print(f"Terminating: SMin deceeds presumed lower influence limit.")
                break

            maxUnsuccessfulRankIncreases = 3
            if np.max(np.array(bestX.ranks()) - np.array(self.x.ranks())) > maxUnsuccessfulRankIncreases:
                print(f"Terminating: Residual not affected by rank increase.")
                break

        else:
            print("Terminating: Maximum iterations reached.")

        for rk_idx in sorted(self.maxRanksReached):
            print(f"WARNING: maxRanks[{rk_idx}] = {self.maxRanks[rk_idx]} was reached during optimization.")

        print("-"*125)
        print(f"Best validation residual in iteration {bestIteration}.")
        self.x = bestX

        self.finalize()

        print(f"Residual decreased from {initialResidual:.2e} to {bestTrainingResidual:.2e} in {iteration} iterations.")
        print("="*125)

    def finalize(self):
        M = self.x.order()
        for m in range(1,M):
            core = self.x.get_component(m).to_ndarray()
            invLT = self.basisWeightFactors[m-1]
            core = np.einsum('ed,ldr -> ler', invLT, core)
            assert np.all(np.isfinite(core))
            self.x.set_component(m, xe.Tensor.from_ndarray(core))
