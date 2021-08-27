import autoPDB

import sys
if not sys.warnoptions:
    import os, warnings
    warnings.filterwarnings("ignore")        # Change the filter in this process
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

import numpy as np
from numpy.polynomial.legendre import legval
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
d = 50        # number of dimensions in each mode
S = 1

# profiling = True
# maxIterations = 3
profiling = False
maxIterations = 100

fileName = ".cache/darcy_uniform_mean.npz"
microstepSolverList = ["lasso", "sklearn.ridge"]
numTrainingSamplesList = [45, 100, 500, 1000]
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


with timeit("Compute measures: {:.2f} s"):
    factors = np.sqrt(2*np.arange(d)+1)
    measures = legval(samples, np.diag(factors)).T
    assert measures.shape == (M,len(samples),d)


gramian_dest = f".cache/{__file__[:-3]}_gramian_{d}.npy"
try:
    G = np.load(gramian_dest)
    assert G.shape == (d,d)
except:
    G = Gramian(d, Hkinner(1))
    np.save(gramian_dest, G)
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
                solver.unstableIsometryConstant = 1.7
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


def test(tt, header):
    assert 0 < len(header) <= 50
    header = f" {header} "
    print("-"*25 + header + "-"*(55-len(header)))
    print(f"  Ranks:     {tt.ranks()}")
    density = lambda core: np.count_nonzero(core)/core.size
    densities = "[" + ", ".join(f"{100*density(tt.get_component(pos)):.0f}" for pos in range(tt.order())) + "]%"
    print(f"  Densities: {densities}")
    dofs = sum(np.count_nonzero(tt.get_component(pos)) for pos in range(tt.order()))
    print(f"  Dofs:      {dofs}")
    valuesLoc = values[-numTestSamples:,:]
    measuresLoc = measures[:,-numTestSamples:,:]
    res = valuesLoc - evaluate(tt, measuresLoc)
    err = np.linalg.norm(res)/np.linalg.norm(valuesLoc)
    print(f"  Residual:  {err:.2e}")
    return err


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


errors = np.empty((len(numTrainingSamplesList), len(microstepSolverList)+2))

for i,numTrainingSamples in enumerate(numTrainingSamplesList):
    header = f" numTrainingSamples = {numTrainingSamples} "
    print("\n" + "="*25 + header + "="*(55-len(header)))
    for j,microstepSolver in enumerate(microstepSolverList):
        l1_solution_dest = f".cache/{__file__[:-3]}_{microstepSolver}_{d}x{M}_{numTrainingSamples}.xrs"
        l1_x = tn2tt(xe.load_from_file(l1_solution_dest))
        errors[i,j] = test(l1_x, f"l1-SALSA ({microstepSolver})")
        test(hard_threshold(l1_x), f"l1-SALSA ({microstepSolver}; thresholded)")

    salsa_solution_dest = f".cache/{__file__[:-3]}_{d}x{M}_{numTrainingSamples}_salsa.xrs"
    salsa_x = tn2tt(xe.load_from_file(salsa_solution_dest))
    errors[i,j+1] = test(salsa_x, "SALSA")
    test(hard_threshold(salsa_x), "SALSA (thresholded)")
    
    uq_solution_dest = f".cache/{__file__[:-3]}_{d}x{M}_{numTrainingSamples}_uq_ra_adf.xrs"
    uq_x = tn2tt(xe.load_from_file(uq_solution_dest))
    errors[i,j+2] = test(uq_x, "uq_ra_adf")
    test(hard_threshold(uq_x), "uq_ra_adf (thresholded)")
print("="*80 + "\n")


console = Console()

table = Table(title="Darcy (uniform)", title_style="bold", show_header=True, header_style="dim")
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

console.print(table)
