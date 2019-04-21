import collections.abc
import typing as t
import copy
import functools

import edn_format as edn
import toolz.itertoolz as tzi
import toolz.dicttoolz as tzd
import toolz.functoolz as tzf
import toolz.curried as tzc
import atomos.atom


"""Port of data structures and predicates from Clojure.

Uses data structures from edn_format, but provides indirection layer in case the library should be replaced.
"""


Keyword = edn.Keyword
Symbol = edn.Symbol

# common usage aliases
K = Keyword
S = Symbol


def str2edn(form: str):
    return edn.loads(form)


def name(form: t.Union[Keyword, Symbol]) -> str:
    return form.name


def is_keyword(obj) -> bool:
    return isinstance(obj, Keyword)


def is_symbol(obj) -> bool:
    return isinstance(obj, Symbol)


def is_sequential(obj) -> bool:
    return isinstance(obj, collections.abc.Sequence)


def is_seq(obj):
    return is_sequential(obj)


def is_nil(obj):
    return (obj is None)


def is_map(obj):
    return isinstance(obj, t.Mapping)


def is_instance(klass, obj):
    return isinstance(obj, klass)


def satisfies(protocol, x):
    return isinstance(x, protocol)


def next_(seq):
    return list(tzi.drop(1, seq)) or None


def get(seq, ind, default=None):
    e = tzi.get(ind, seq, default)
    if e == default:
        try:
            return getattr(seq, ind)
        except (AttributeError, TypeError):
            pass
    return e


def into(to, from_):
    return to + from_


def is_empty(coll):
    return not coll


def not_empty(coll):
    if coll:
        return coll
    else:
        return None


def is_distinct(seq):
    return tzi.isdistinct(seq)


def some_fn(*fns):
    assert len(fns) == 2, 'some_fn is temporarily defined only for two fns'

    def _some(*args):
        for arg in args:
            for fn in fns:
                result = fn(arg)
                if result is not False and result is not None:
                    return result

    return _some


def with_meta(obj, m):
    copied = copy.copy(obj)
    copied.clj_meta = m
    return copied


class MetaMixin:
    clj_meta = {}


def extract_seq(seq, n_first):
    """My destructuring implementation"""
    firsts = []
    for i in range(n_first):
        val = get(seq, i)
        firsts.append(val)

    rest = seq[n_first:]
    if is_empty(rest):
        rest = None

    return tuple(firsts) + (rest,)


def first(seq):
    return get(seq, 0)


def last(seq):
    return get(seq, -1)


def reduce(f, val, coll):
    # note the argument ordering
    return functools.reduce(f, liberal_iter(coll), val)


def conj(coll, x):
    if coll is None:
        return [x]

    conjed = list(coll) + [x]
    return type(coll)(conjed)


def concat(*seqs):
    return tzf.thread_last(seqs,
                           tzc.map(lambda s: s or []),
                           tzi.concat,
                           list)


def set_(seq):
    return set(seq or [])


def fnil(f, x):
    def _patched(*args):
        a, *args_rest = args
        if a is None:
            called_args = [x] + args_rest
            return f(*called_args)
        else:
            return f(*args)
    return _patched


def count(seq):
    return len(seq or [])


def liberal_iter(seq_or_nil):
    if isinstance(seq_or_nil, t.Mapping):
        return seq_or_nil.items()

    return iter(seq_or_nil or [])


def merge(*maps):
    sanitized = [m for m in maps if not is_nil(m)]
    return tzd.merge(*sanitized)


def atom(x):
    return atomos.atom.Atom(x)


def mapv(f, *colls):
    iters = [liberal_iter(c) for c in colls]
    return [f(*args) for args in zip(*iters)]


def mapcat(f, *colls):
    for coll in map(f, *colls):
        for e in coll:
            yield e


def zipmap(keys, vals):
    return dict(zip(keys, vals))


def some(pred, coll):
    for e in coll:
        val = pred(e)
        if val is not None and val is not False:
            return e


def is_every(pred, coll):
    for e in coll:
        if not pred(e):
            return False
    return True


def complement(f):
    return tzf.complement(f)


def is_pos(num):
    return num > 0


def vec(coll):
    return tuple(coll)


def vector(*args):
    return vec(args)


def butlast(coll):
    return coll[:-1]


def is_string(x):
    return is_instance(str, x)


def compare(a, b):
    return (a > b) - (a < b)


def select_keys(m, keyseq):
    return {k: m[k] for k in keyseq}


def to_array(coll):
    return list(coll)


def aclone(array):
    return list(array)


def aget(array, idx):
    return array[idx]


def aset(array, idx, val):
    array[idx] = val
    return val
