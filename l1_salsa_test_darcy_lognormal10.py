import autoPDB

import sys
if not sys.warnoptions:
    import os, warnings
    warnings.filterwarnings("ignore")        # Change the filter in this process
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses
from math import factorial

import numpy as np
from numpy.polynomial.hermite_e import hermeval
import xerus as xe
from rich.console import Console
from rich.table import Table

from l1_salsa import InternalSolver
from misc import (
    timeit,
    tensor,
    HkinnerLegendre as Hkinner,
    Gramian,
    evaluate,
    tn2tt,
    save_to_file_verbose,
    get_profiler
)

M = 20        # number of modes
d = 7         # number of dimensions in each mode
S = 1

# profiling = True
# maxIterations = 3
profiling = False
maxIterations = 100

# fileName = ".cache/darcy_lognormal_mean.npz"
fileName = ".cache/darcy_lognormal10_mean.npz"
microstepSolverList = ["lasso", "sklearn.ridge"]
# numTrainingSamplesList = [45, 100, 500]
# numTrainingSamplesList = [45, 100, 500, 1000]
numTrainingSamplesList = [45, 100, 500, 1000, 9000]
numTestSamples = 1000


with timeit("Load samples: {:.2f} s"):
    z = np.load(fileName)
    samples = z['samples']
    values = z['values']
    assert samples.ndim == 2 and values.ndim == 1 and samples.shape[0] == values.shape[0]
    assert M <= samples.shape[1]
    samples = samples[:,:M]
assert len(samples) >= max(numTrainingSamplesList) + numTestSamples
values = values[:, np.newaxis]
assert values.shape == (len(samples),S)
digits = int(np.ceil(np.log10(len(samples))))+1
print(f"Number of samples:               {len(samples): >{digits}d}")
print(f"Max. number of training samples: {max(numTrainingSamplesList): >{digits}d}")
print(f"Number of test samples:          {numTestSamples: >{digits}d}")
print("-"*80)
print(f"Mean:     {np.mean(values):.2e}")
print(f"Variance: {np.var(values):.2e}")


with timeit("Compute measures: {:.2f} s"):
    factorials = np.array([factorial(k) for k in range(d)], dtype=object)
    factors = np.sqrt((1/factorials).astype(float))
    measures = hermeval(samples, np.diag(factors)).T
    assert measures.shape == (M,len(samples),d)


# G = np.eye(d)  # standard l1...
G = np.diag(factorials**3).astype(float)
basisWeights = [np.eye(S)] + [G]*M


for numTrainingSamples in numTrainingSamplesList:
    for microstepSolver in microstepSolverList:
        l1_solution_dest = f".cache/{__file__[:-3]}_{microstepSolver}_{d}x{M}_{numTrainingSamples}.xrs"
        if not os.path.exists(l1_solution_dest):
            valuesLoc   = values[:numTrainingSamples]
            measuresLoc = measures[:,:numTrainingSamples,:]
            measuresLoc = [tensor(m) for m in measuresLoc]
            with get_profiler(f"profiles/{__file__[:-3]}.cprof", profiling) as pr:
                solver = InternalSolver([S]+[d]*M, measuresLoc, valuesLoc, basisWeights=basisWeights, profiler=pr)
                solver.minDecrease = 1e-2
                solver.maxIterations = maxIterations
                solver.targetResidual = 1e-6
                solver.trackingPeriodLength = 10
                solver.maxRanks = [S] + [5]*(M-1)
                solver.minSMinFactor = 1e-10
                solver.controlSetFraction = 0.1
                solver.unstableIsometryConstant = 10
                solver.microstepSolver = microstepSolver
                solver.solve()
            save_to_file_verbose(solver.x, l1_solution_dest)

    uq_solution_dest = f".cache/{__file__[:-3]}_{d}x{M}_{numTrainingSamples}_uq_ra_adf.xrs"
    if not os.path.exists(uq_solution_dest):
        valuesLoc   = values[:numTrainingSamples]
        valuesLoc   = [tensor(val) for val in valuesLoc]
        measuresLoc = np.transpose(measures[:,:numTrainingSamples,:], axes=(1,0,2))
        assert measuresLoc.shape == (numTrainingSamples,M,d)
        measuresLoc = [[tensor(cmp_m) for cmp_m in m] for m in measuresLoc]
        x = xe.uq_ra_adf(measuresLoc, valuesLoc, [S]+[d]*M, targeteps=1e-6, maxitr=2000)
        save_to_file_verbose(x, uq_solution_dest)

    salsa_solution_dest = f".cache/{__file__[:-3]}_{d}x{M}_{numTrainingSamples}_salsa.xrs"
    if not os.path.exists(salsa_solution_dest):
        valuesLoc = values[:numTrainingSamples]
        init = np.mean(valuesLoc, axis=0)
        assert init.shape == (S,)
        init = xe.TTTensor(xe.Tensor.from_ndarray(init))
        init = xe.dyadic_product([init, xe.TTTensor.dirac([d]*M, [0]*M)])
        valuesLoc = xe.Tensor.from_ndarray(valuesLoc)
        measuresLoc = measures[:,:numTrainingSamples,:]
        measuresLoc = [xe.Tensor.from_buffer(np.array(measuresLoc[m])) for m in range(M)]

        solver = xe.uqSALSA(init, measuresLoc, valuesLoc)
        solver.maxSweeps = maxIterations
        solver.targetResidual = 1e-6
        solver.trackingPeriodLength = 10
        solver.maxStagnatingEpochs = maxIterations//10
        solver.maxRanks = [S] + [5]*(M-1)
        solver.basisWeights = [xe.Tensor.from_buffer(w) for w in basisWeights]
        solver.alphaFactor = 0
        solver.maxIRsteps = 0
        solver.fomega = 1.1
        solver.run()
        save_to_file_verbose(solver.bestState.x, salsa_solution_dest)


def test(tt, header, verbose=True):
    assert 0 < len(header) <= 50
    header = f" {header} "
    density = lambda core: np.count_nonzero(core)/core.size
    densities = "[" + ", ".join(f"{100*density(tt.get_component(pos)):.0f}" for pos in range(tt.order())) + "]%"
    dofs = sum(np.count_nonzero(tt.get_component(pos)) for pos in range(tt.order()))
    valuesLoc = values[-numTestSamples:,:]
    measuresLoc = measures[:,-numTestSamples:,:]
    res = valuesLoc - evaluate(tt, measuresLoc)
    err = np.linalg.norm(res)/np.linalg.norm(valuesLoc)
    if verbose:
        print("-"*25 + header + "-"*(55-len(header)))
        print(f"  Ranks:     {tt.ranks()}")
        print(f"  Densities: {densities}")
        print(f"  Dofs:      {dofs}")
        print(f"  Residual:  {err:.2e}")
    return dofs, err


def hard_threshold(_tt, _threshold=1e-6):
    tt = xe.TTTensor(_tt)
    for pos in range(tt.order()):
        tt.move_core(pos)
        core = tt.get_component(pos).to_ndarray()
        core[abs(core) < _threshold] = 0
        core = xe.Tensor.from_buffer(core)
        tt.set_component(pos, core)
    tt.move_core(0)
    return tt


dofss = np.empty((len(numTrainingSamplesList), len(microstepSolverList)+2), dtype=int)
errors = np.empty((len(numTrainingSamplesList), len(microstepSolverList)+2))

verbose = False
for i,numTrainingSamples in enumerate(numTrainingSamplesList):
    header = f" numTrainingSamples = {numTrainingSamples} "
    if verbose: print("\n" + "="*25 + header + "="*(55-len(header)))
    for j,microstepSolver in enumerate(microstepSolverList):
        l1_solution_dest = f".cache/{__file__[:-3]}_{microstepSolver}_{d}x{M}_{numTrainingSamples}.xrs"
        l1_x = tn2tt(xe.load_from_file(l1_solution_dest))
        errors[i,j] = test(l1_x, f"l1-SALSA ({microstepSolver})", verbose)[1]
        dofss[i,j] = test(hard_threshold(l1_x), f"l1-SALSA ({microstepSolver}; thresholded)", verbose)[0]

    salsa_solution_dest = f".cache/{__file__[:-3]}_{d}x{M}_{numTrainingSamples}_salsa.xrs"
    salsa_x = tn2tt(xe.load_from_file(salsa_solution_dest))
    errors[i,j+1] = test(salsa_x, "SALSA", verbose)[1]
    dofss[i,j+1] = test(hard_threshold(salsa_x), "SALSA (thresholded)", verbose)[0]

    uq_solution_dest = f".cache/{__file__[:-3]}_{d}x{M}_{numTrainingSamples}_uq_ra_adf.xrs"
    uq_x = tn2tt(xe.load_from_file(uq_solution_dest))
    errors[i,j+2] = test(uq_x, "uq_ra_adf", verbose)[1]
    dofss[i,j+2] = test(hard_threshold(uq_x), "uq_ra_adf (thresholded)", verbose)[0]
if verbose: print("="*80)


console = Console()

table = Table(title="Error for lognormal Darcy", title_style="bold", show_header=True, header_style="dim")
table.add_column(style="dim")  # Algorithm
for numTrainingSamples in numTrainingSamplesList:
    table.add_column(f"N = {numTrainingSamples}", justify="right")

algMin = np.argmin(errors, axis=1)
assert microstepSolverList == ["lasso", "sklearn.ridge"]
for e,algorithm in enumerate(["l1-SALSA", "l2-SALSA", "SALSA", "uq_ra_adf"]):
    algErrors = [f"{err:.2e}" for err in errors[:, e]]
    for k in range(len(numTrainingSamplesList)):
        if algMin[k] == e:
            algErrors[k] = f"[bold]{algErrors[k]}[/bold]"
    table.add_row(algorithm, *algErrors)

print()
console.print(table)

table = Table(title="Dofs for lognormal Darcy", title_style="bold", show_header=True, header_style="dim")
table.add_column(style="dim")  # Algorithm
for numTrainingSamples in numTrainingSamplesList:
    table.add_column(f"N = {numTrainingSamples}", justify="right")

algMin = np.argmin(dofss, axis=1)
assert microstepSolverList == ["lasso", "sklearn.ridge"]
for e,algorithm in enumerate(["l1-SALSA", "l2-SALSA", "SALSA", "uq_ra_adf"]):
    algDofs = [f"{dofs:d}" for dofs in dofss[:, e]]
    for k in range(len(numTrainingSamplesList)):
        if algMin[k] == e:
            algDofs[k] = f"[bold]{algDofs[k]}[/bold]"
    table.add_row(algorithm, *algDofs)

print()
console.print(table)
