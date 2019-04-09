import collections.abc


"""Port of data structures and predicates from Clojure"""


class Keyword:
    def __init__(self, name):
        self._name = name

    def __str__(self):
        return ':' + self._name

    def __repr__(self):
        return "Keyword('{}')".format(self._name)

    def __hash__(self):
        return hash(self._name)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self._name == other._name

    def name(self):
        return self._name


class Symbol:
    def __init__(self, name):
        self._name = name

    def __str__(self):
        return "'{}".format(self._name)

    def __repr__(self):
        return "Symbol('{}')".format(self._name)

    def __hash__(self):
        return hash(self._name)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self._name == other._name

    def name(self):
        return self._name


K = Keyword
S = Symbol


def is_keyword(obj) -> bool:
    return isinstance(obj, Keyword)


def is_symbol(obj) -> bool:
    return isinstance(obj, Symbol)


def is_sequential(obj) -> bool:
    return isinstance(obj, collections.abc.Sequence)
