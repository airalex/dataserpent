from pprint import pprint as pp
import typing as t
import collections
import collections.abc
import functools
import abc

import toolz.itertoolz as tzi
import toolz.dicttoolz as tzd
import toolz.functoolz as tzf

import src.clj as clj
from src.clj import K, S


# implementation based on
# https://github.com/tonsky/datascript/blob/aa67e7a4d99b954a357b0c6533bd7039f5d99e54/src/datascript/parser.cljc


class ITraversable(abc.ABC):
    def _collect(self, pred, acc):
        # TODO
        pass

    def _collect_vars(self, acc):
        # TODO
        pass

    def _postwalk(self, f):
        # TODO
        pass


def query2map(query: list) -> dict:
    def _loop(parsed: dict, key, qs: list):
        q = tzi.get(0, qs, None)
        if q is not None:
            if clj.is_keyword(q):
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

FindRel = collections.namedtuple('FindRel', ['elements'])


def parse_seq(parse_el, form):
    if clj.is_sequential(form):
        acc = []
        for e in form:
            parsed = parse_el(e)
            if parsed is not None:
                acc.append(parsed)
            else:
                acc = None
                break
        return acc


def is_distinct(coll):
    return clj.is_empty(coll) or clj.is_distinct(coll)


def with_source(obj, source):
    return clj.with_meta(obj, {'source': source})


Variable = collections.namedtuple('Variable', ['symbol'])
SrcVar = collections.namedtuple('SrcVar', ['symbol'])
DefaultSrc = collections.namedtuple('DefaultSrc', [])


def parse_variable(form):
    if clj.is_symbol(form) and clj.name(form)[0] == "?":
        return Variable(form)


def parse_src_var(form):
    if clj.is_symbol(form) and clj.name(form)[0] == '$':
        return SrcVar(form)


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
    if clj.is_sequential(form) and len(form) == 1:
        inner = form[0]
        if clj.is_sequential(inner) and len(inner) == 2 and inner[1] == S('...'):
            element = parse_find_elem(inner[0])
            if element is not None:
                return FindColl(element)


def parse_find_scalar(form):
    if clj.is_sequential(form) and len(form) == 2 and form[1] == S('.'):
        element = parse_find_elem(form[0])
        if element is not None:
            return FindScalar(element)


def parse_find_tuple(form):
    if clj.is_sequential(form) and len(form) == 1:
        inner = form[0]
        element = parse_seq(parse_find_elem, inner)
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


Pattern = collections.namedtuple('Pattern', ['source', 'pattern'])
Predicate = collections.namedtuple('Predicate', ['fn', 'args'])
Function = collections.namedtuple('Function', ['fn', 'args', 'binding'])
RuleExpr = collections.namedtuple('RuleExpr', ['source', 'name', 'args'])
Not = collections.namedtuple('Not', ['source', 'vars_', 'clauses'])
Or = collections.namedtuple('Or', ['source', 'rule_vars', 'clauses'])
And = collections.namedtuple('And', ['clauses'])


def take_source(form):
    if clj.is_sequential(form):
        source = parse_src_var(form[0])
        if source is not None:
            return source, clj.next_(form)
        else:
            return DefaultSrc(), form


def _collect_vars_acc(acc, form):
    if isinstance(form, Variable):
        return acc + form
    if isinstance(form, Not):
        return clj.into(acc, form.vars_)
    if isinstance(form, Or):
        return _collect_vars_acc(acc, form.rule_vars)
    if clj.satisfies(form, ITraversable):
        return _collect_vars(form, acc)
    if clj.is_sequential(form):
        return list(functools.reduce(_collect_vars_acc, initial=acc, sequence=form))
    return acc


def _collect_vars(form):
    return _collect_vars_acc([], form)


def collect_vars_distinct(form):
    return list(set(_collect_vars(form)))


def _validate_join_vars(vars_, clauses, form):
    undeclared = set(vars_).difference(set(_collect_vars(clauses)))
    if not clj.is_empty(undeclared):
        assert False, "Join variable not declared inside clauses: " + clj.mapv(lambda v: v.symbol, undeclared)

    if clj.is_empty(vars_):
        assert False, "Join variables should not be empty"


def _validate_not(clause, form):
    _validate_join_vars(clause.vars_, clause.clauses, form)
    return clause


def parse_not(form):
    source_star_next_form = take_source(form)
    if source_star_next_form is not None:
        source_star, next_form = source_star_next_form
        sym = next_form[0]
        clauses = clj.next_(next_form)
        if sym == S('not'):
            clauses_star = parse_clauses(clauses)
            if clauses_star is not None:
                return tzf.thread_first(Not(source_star,
                                            collect_vars_distinct(clauses_star),
                                            clauses_star),
                                        (with_source, form),
                                        (_validate_not, form))
            else:
                assert False, "Cannot parse 'not' clause, expected [ src-var? 'not' clause+ ]"

    # TODO


def parse_clause(form):
    result = \
        parse_not(form) or \
        parse_not_join(form) or \
        parse_or(form) or \
        parse_or_join(form) or \
        parse_pred(form) or \
        parse_fn(form) or \
        parse_rule_expr(form) or \
        parse_pattern(form)
    assert result is not None, \
        'Cannot parse clause, expected' + \
        ' (data-pattern | pred-expr | fn-expr | rule-expr | not-clause | not-join-clause | or-clause | or-join-clause)'


def parse_clauses(clauses):
    return parse_seq(parse_clause, clauses)


def parse_where(qwhere: [[t.Union[S, K]]]):
    pass


def parse_query(query: t.Union[list, dict]):
    qm = query2map(query)

    qfind = parse_find(qm[K('find')])
    if K('with') in qm:
        qwith = parse_with(qm[K('with')])
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
