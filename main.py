from pprint import pprint as pp
import typing as t
import collections

import toolz.itertoolz as tzi
import toolz.dicttoolz as tzd


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



K = Keyword
S = Symbol


def query2map(query: list) -> dict:
    def _loop(parsed: dict, key, qs: list):
        q = tzi.get(0, qs, None)
        if q is not None:
            if isinstance(q, Keyword):
                return _loop(parsed,
                             q,
                             list(tzi.drop(1, qs)))
            else:
                return _loop(tzd.update_in(parsed, [key], lambda prev: prev + q if prev else [q]),
                             key,
                             list(tzi.drop(1, qs)))
        else:
            return parsed

    return _loop({}, None, query)


Query = collections.namedtuple('Query', ['qfind', 'qwith', 'qin', 'qwhere'])


def parse_find(qfind: [S]):
    pass

def parse_with(qwith):
    pass

def parse_in(qin):
    pass

def parse_where(qwhere: [[t.Union[S, K]]]):
    pass


def parse_query(query: t.Union[list, dict]):
    qm = query2map(query)

    qfind = parse_find(qm[K('find')])
    if K('with') in qm:
        qwith = parse_with(qm[L('with')])
    else:
        qwith = None
    qin = parse_in(qm.get(K('in'), [S('$')]))
    qwhere = parse_where(qm.get(K('where'), []))

    res = Query(qfind, qwith, qin, qwhere)

    return res


def main():
    # k = Keyword("dataserpent")
    # print(k)
    # print(repr(k))

    query = [K('find'), S('?e'),
             K('where'), [S('?e'), K('name')]]
    # pp(query2map(query))
    # => {Keyword('find'): [Symbol('?e')],
    #     Keyword('where'): [[Symbol('?e'), Keyword('name')]]}
    parsed_query = parse_query(query)
    pp(parsed_query)


if __name__ == '__main__':
    main()
