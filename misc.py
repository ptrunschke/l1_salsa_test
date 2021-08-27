import cProfile
import time
from contextlib import contextmanager

import numpy as np
from numpy.polynomial.legendre import legval, legmul, legint, legder
import xerus as xe


@contextmanager
def timeit(title):
    try:
        t0 = time.process_time()
        yield
    finally:
        print(title.format(time.process_time()-t0))


@contextmanager
def get_profiler(_dest, _active=True):
    if _active:
        print("="*80)
        print(f"Profiling l1-SALSA. Destination: {_dest}")
        print("="*80)
        pr = cProfile.Profile()
        pr.enable()
        yield pr
        pr.disable()
        pr.dump_stats(_dest)
    else:
        yield None


def save_to_file_verbose(_tt, _file):
    print("="*80)
    print(f"Saving to File: {_file}")
    print("="*80)
    xe.save_to_file(_tt, _file)


tensor = lambda arr: xe.Tensor.from_buffer(arr)


def evaluate(tt, measures):
    assert isinstance(tt, xe.TTTensor) and isinstance(measures, np.ndarray)
    assert tt.order() == measures.shape[0]+1
    ret = np.ones((measures.shape[1],1))
    for pos in reversed(range(1, tt.order())):
        cmp = tt.get_component(pos).to_ndarray()
        ret = np.einsum('ler, ne, nr -> nl', cmp, measures[pos-1], ret)
    cmp = tt.get_component(0).to_ndarray()[0]
    return np.einsum('er,nr -> ne', cmp, ret)


def L2innerLegendre(c1, c2):
    i = legint(legmul(c1, c2))
    return legval(1, i) - legval(-1, i)


def HkinnerLegendre(k):
    assert isinstance(k, int) and k >= 0
    def inner(c1, c2):
        ret = L2innerLegendre(c1, c2)
        for j in range(k):
            c1 = legder(c1)
            c2 = legder(c2)
            ret += L2innerLegendre(c1, c2)
        return ret
    return inner


def Gramian(d, inner):
    matrix = np.empty((d,d))
    e = lambda k: np.eye(1,d,k)[0]
    for i in range(d):
        ei = e(i)
        for j in range(i+1):
            ej = e(j)
            matrix[i,j] = matrix[j,i] = inner(ei,ej)
    return matrix


def check_tt(tn):
    if not isinstance(tn, xe.TensorNetwork):
        return False
    left_link_to  = lambda node, pos: not node.neighbors[0].external and node.neighbors[0].other == pos
    right_link_to = lambda node, pos: not node.neighbors[-1].external and node.neighbors[-1].other == pos
    tt_component  = lambda node, pos: len(node.neighbors) == 3 and left_link_to(node, pos-1) and node.neighbors[1].external and right_link_to(node, pos+1)
    end_component = lambda node: node.tensorObject.dimensions == [1] and node.tensorObject[0] == 1
    if not end_component(tn.nodes[0]) and right_link_to(tn.nodes[0], 1):
        return False
    for pos,node in enumerate(tn.nodes[1:-1], start=1):
        if not tt_component(node, pos):
            return False
    return end_component(tn.nodes[-1]) and left_link_to(tn.nodes[-1], pos)


def tn2tt(tn):
    assert check_tt(tn)
    tt_list = [ttn.tensorObject for ttn in tn.nodes[1:-1]]
    tt = xe.TTTensor([t.dimensions[1] for t in tt_list])
    for pos in range(len(tt_list)):
        tt.set_component(pos, tt_list[pos])
    def inner(t1, t2):
        i, = xe.indices(1)
        return float(t1(i&0) * t2(i&0))
    tn_norm = xe.frob_norm(tn)**2
    error = abs(tn_norm + xe.frob_norm(tt)**2 - 2*inner(tn, tt))
    if not error <= 1e-12*tn_norm:
        raise argparse.ArgumentTypeError(f"could not convert tensor network to tensor train network (NOT {error:.2e} <= 1e-12*{tn_norm:.2e})")
    return tt


def cartesian_product(x, y):
    return np.dstack(np.meshgrid(x, y)).reshape(-1, 2)
