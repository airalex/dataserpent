from pprint import pprint as pp
import typing as t
import collections
import collections.abc

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

# implementation based on https://github.com/tonsky/datascript/blob/aa67e7a4d99b954a357b0c6533bd7039f5d99e54/src/datascript/parser.cljc

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


# --- clojure proxy

def is_sequential(obj) -> bool:
    return isinstance(obj, collections.abc.Sequence)

# ---

Query = collections.namedtuple('Query', ['qfind', 'qwith', 'qin', 'qwhere'])

FindRel = collections.namedtuple('FindRel', ['elements'])

def parse_seq(parse_el, form):
    if isinstance(form, collections.abc.Iterable):
        acc = []
        for e in form:
            parsed = parse_el(e)
            if parsed is not None:
                acc.append(parsed)
            else:
                break
        return acc

Variable = collections.namedtuple('Variable', ['symbol'])

def parse_variable(form):
    if isinstance(form, Symbol) and form.name()[0] == "?":
        return Variable(form)

def parse_pull_expr(form):
    # Not implemented yet
    pass

def parse_aggregate_custom(form):
    # Not implemented yet
    pass

def parse_aggregate(form):
    # Not implemented yet
    pass


def parse_find_elem(form):
    return \
        parse_variable(form) or \
        parse_pull_expr(form) or \
        parse_aggregate_custom(form) or \
        parse_aggregate(form)


def parse_find_rel(form):
    elements = parse_seq(parse_find_elem, form)
    if elements is not None:
        return FindRel(elements)

FindColl = collections.namedtuple('FindColl', ['element'])
FindScalar = collections.namedtuple('FindScalar', ['element'])
FindTuple = collections.namedtuple('FindTuple', ['element'])

def parse_find_coll(form):
    if is_sequential(form) and len(form) == 1:
        inner = form[0]
        if is_sequential(inner) and len(inner) == 2 and inner[1] == S('...'):
            element = parse_find_elem(inner[0])
            if element is not None:
                return FindColl(element)

def parse_find_scalar(form):
    if is_sequential(form) and len(form) == 2 and form[1] == S('.'):
        element = parse_find_elem(form[0])
        if element is not None:
            return FindScalar(element)

def parse_find_tuple(form):
    if is_sequential(form) and len(form) == 1:
        inner = form[0]
        element = parse_find_elem(inner)
        if element is not None:
            return FindTuple(element)

def parse_find(form: [S]):
    result = \
        parse_find_rel(form) or \
        parse_find_coll(form) or \
        parse_find_scalar(form) or \
        parse_find_tuple(form)
    assert result is not None, 'Cannot parse :find, expected: (find-rel | find-coll | find-tuple | find-scalar)'
    return result


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
