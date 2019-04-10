import collections.abc
import typing as t
import copy

import edn_format as edn
import toolz.itertoolz as tzi


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


def next_(seq):
    return list(tzi.drop(1, seq))


def into(to, from_):
    return to + from_


def with_meta(obj, m):
    copied = copy.copy(obj)
    copied.clj_meta = m
    return copied


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
        return fns[0](args) or fns[1](args)

    return _some
