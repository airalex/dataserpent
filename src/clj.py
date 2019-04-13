import collections.abc
import typing as t
import copy
import functools

import edn_format as edn
import toolz.itertoolz as tzi
import toolz.functoolz as tzf
import toolz.curried as tzc


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


def is_map(obj):
    return isinstance(obj, t.Mapping)


def satisfies(protocol, x):
    return isinstance(x, protocol)


def next_(seq):
    return list(tzi.drop(1, seq)) or None


def get(seq, ind, default=None):
    return tzi.get(ind, seq, default)


def into(to, from_):
    return to + from_


def is_empty(coll):
    return not coll


def not_empty(coll):
    return not is_empty(coll)


def is_distinct(seq):
    return tzi.isdistinct(seq)


def mapv(f, *colls):
    assert len(colls) == 1, 'mapv is temporarily defined only for single coll'
    return list(map(f, colls[0]))


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


def reduce(f, val, coll):
    # note the argument ordering
    return functools.reduce(f, coll, val)


def conj(coll, x):
    return coll + [x]


def concat(*seqs):
    return tzf.thread_last(seqs,
                           tzc.map(lambda s: s or []),
                           tzi.concat,
                           list)

def set_(seq):
    return set(seq or [])
