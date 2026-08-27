"""Run every GPU experiment's CONTROL FLOW on CPU with a stubbed kernel.

Two exp8 jobs -- the parameter-free falsifier and the three cells, the two most
valuable experiments in the set -- died on the GPU with

    IndexError: tuple index out of range      (best[3] on a 3-tuple)

That is a pure control-flow bug. It syntax-checked clean, `py_compile` passed,
and nothing local could reach it because the file imports torch and triton at
module scope. A queue slot found it.

This stubs torch/triton/CUDA hard enough that main() runs end to end on this
machine: every branch is taken, every record is packed and unpacked, every format
string is evaluated. It does NOT check numerics -- the timings are fabricated --
so it cannot replace the GPU. It catches exactly the class of bug that cost those
two jobs, and it runs in seconds.

    .venv/bin/python tools/dryrun_experiments.py
"""
import pathlib
import runpy
import sys
import types

REPO = pathlib.Path(__file__).resolve().parents[1]
GPU = REPO / "gpu"


def _stub_torch():
    import numpy as np

    class _T:                                   # a tensor-ish thing
        def __init__(self, shape):
            self.shape = tuple(shape)
            self.dtype = "float16"
            self.device = "cuda"
        def __getattr__(self, k):
            if k in ("half", "float", "contiguous", "cuda", "clone", "t", "zero_",
                     "transpose", "permute", "view", "detach", "flatten"):
                return lambda *a, **kw: self
            raise AttributeError(k)
        def stride(self, i=None):
            s = [1] * len(self.shape)
            for j in range(len(self.shape) - 2, -1, -1):
                s[j] = s[j + 1] * self.shape[j + 1]
            return s if i is None else s[i]
        def reshape(self, *sh):
            sh = sh[0] if len(sh) == 1 and isinstance(sh[0], (tuple, list)) else sh
            return _T(sh)
        def index_select(self, dim, idx):
            return self
        def numel(self):
            return int(np.prod(self.shape))
        def __getitem__(self, k):  return self
        def __setitem__(self, k, v):  pass
        def __sub__(self, o):  return self
        def __rsub__(self, o):  return self
        def __mul__(self, o):  return self
        def __rmul__(self, o):  return self
        def __add__(self, o):  return self
        def __radd__(self, o):  return self
        def __truediv__(self, o):  return self
        def __rtruediv__(self, o):  return self
        def __neg__(self):  return self
        def __mod__(self, o):  return self
        def __matmul__(self, o):  return self
        def __rmatmul__(self, o):  return self
        def __rmod__(self, o):  return self
        def __floordiv__(self, o):  return self
        def __ge__(self, o):  return _T((1,))
        def __le__(self, o):  return _T((1,))
        def __lt__(self, o):  return _T((1,))
        def __gt__(self, o):  return _T((1,))
        def __eq__(self, o):  return _T((1,))
        def __hash__(self):  return id(self)
        def __bool__(self):  return True
        def __and__(self, o):  return self
        def __invert__(self):  return self
        def masked_fill(self, *a, **kw):  return self
        def any(self, *a, **kw):  return self
        def all(self, *a, **kw):  return self
        def sum(self, *a, **kw):  return self
        def abs(self):  return self
        def max(self):  return self
        def item(self):  return 1e-4
        def to(self, *a, **kw):  return self
        def long(self):  return self
        def int(self):  return self

    t = types.ModuleType("torch")
    t.float16 = "float16"; t.int32 = "int32"; t.int = "int32"
    t.bool = "bool"; t.float32 = "float32"; t.int64 = "int64"
    t.softmax = lambda x, **kw: x
    t.nan_to_num = lambda x, **kw: x
    t.equal = lambda a, b: True
    t.where = lambda c, a, b: a
    t.randn = lambda *sh, **kw: _T(sh if not isinstance(sh[0], (tuple, list)) else sh[0])
    t.empty_like = t.zeros_like = lambda x, **kw: x
    t.zeros = lambda *sh, **kw: _T(sh if not isinstance(sh[0], (tuple, list)) else sh[0])
    t.arange = lambda n, **kw: _T((n,))
    t.empty = lambda *sh, **kw: _T(sh if not isinstance(sh[0], (tuple,list)) else sh[0])
    t.from_numpy = lambda a: _T(a.shape)
    t.manual_seed = lambda s: None
    t.compile = lambda f, **kw: f
    t.cat = lambda xs, **kw: xs[0]
    t.cos = t.sin = t.exp = lambda x: x
    t.nn = types.SimpleNamespace(
        functional=types.SimpleNamespace(gelu=lambda x: x),
        attention=types.SimpleNamespace(flex_attention=types.SimpleNamespace(
            create_block_mask=lambda *a, **kw: object(),
            flex_attention=lambda *a, **kw: _T((1, 1, 8, 8)))))
    t.cuda = types.SimpleNamespace(
        is_available=lambda: True, synchronize=lambda: None,
        current_device=lambda: 0, get_device_capability=lambda i=0: (8, 0),
        get_device_properties=lambda i=0: types.SimpleNamespace(
            name="stub", total_memory=2**35, multi_processor_count=108),
        Event=lambda *a, **kw: types.SimpleNamespace(
            record=lambda: None, elapsed_time=lambda o: 0.1 + 0.01 * _tick()))
    t._dynamo = types.SimpleNamespace(reset=lambda: None)
    t.version = types.SimpleNamespace(cuda="12.4")
    t.__version__ = "stub"
    return t


_N = [0]
def _tick():
    _N[0] += 1
    return (_N[0] % 7) / 7.0


def _stub_triton():
    tr = types.ModuleType("triton")
    tr.__version__ = "stub"
    tr.jit = lambda f: f
    lang = types.ModuleType("triton.language")
    for n in ("constexpr", "program_id", "arange", "load", "store", "zeros",
              "full", "dot", "trans", "where", "maximum", "exp", "sum", "max",
              "static_range"):
        setattr(lang, n, lambda *a, **kw: None)
    lang.constexpr = int
    tr.language = lang
    return tr, lang


def main():
    sys.path.insert(0, str(GPU))
    sys.modules["torch"] = _stub_torch()
    tr, lang = _stub_triton()
    sys.modules["triton"] = tr
    sys.modules["triton.language"] = lang

    # the kernel entry point is replaced wholesale -- we are testing the
    # experiment's control flow, not the kernel
    ta = types.ModuleType("triton_attn")
    ta.block_sparse_attention = lambda q, *a, **kw: q
    sys.modules["triton_attn"] = ta

    bad = []
    # TWO PASSES. The happy path is not enough: exp8 died on the GPU because
    # every launch config raised, `except Exception: continue` swallowed all 12,
    # and the missing config surfaced two frames later as int('None'). A stub
    # kernel that always succeeds can never reach that. The second pass makes the
    # kernel always raise, so the "nothing ran" path is exercised too -- an
    # experiment must report WHY, not crash on its own bookkeeping.
    for label, boom in (("happy path      ", False), ("kernel always fails", True)):
        print(f"--- {label} ---")
        ta.block_sparse_attention = (
            (lambda *a, **kw: (_ for _ in ()).throw(
                RuntimeError("stub: kernel refused to compile")))
            if boom else (lambda q, *a, **kw: q))
        for f in sorted(GPU.glob("exp*.py")):
            sys.argv = [str(f)]
            try:
                runpy.run_path(str(f), run_name="__main__")
                print(f"  OK        {f.name}")
            except SystemExit as e:
                print(f"  OK (exit {e.code})  {f.name}")
            except Exception as e:
                # In the failure pass, PROPAGATING the kernel's own error is
                # correct -- the reason reaches the log. What is not correct is
                # dying with a DIFFERENT error, which means the experiment lost
                # the reason and reported its own bookkeeping instead. That is
                # exactly what exp8 did: 12 swallowed compile errors surfaced as
                # int('None') two frames away.
                lost = boom and "stub: kernel refused to compile" not in str(e)
                if lost or not boom:
                    bad.append((f"{f.name} [{label.strip()}]",
                                f"{type(e).__name__}: {e}"))
                    print(f"  BROKEN    {f.name}   {type(e).__name__}: {e}")
                else:
                    print(f"  ok, reason preserved  {f.name}")
        print()
    if bad:
        print(f"{len(bad)} experiment(s) would die on the GPU:")
        for n, e in bad:
            print(f"  {n}: {e}")
        return 1
    print("all experiments' control flow runs end to end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
