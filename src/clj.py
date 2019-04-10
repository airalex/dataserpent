import collections.abc
import typing as t

import edn_format as edn


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
