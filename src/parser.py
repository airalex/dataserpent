from pprint import pprint as pp
import typing as t
import collections
import collections.abc
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
        q = clj.get(qs, 0)
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


class Query(collections.namedtuple('Query', ['qfind', 'qwith', 'qin', 'qwhere']), clj.MetaMixin):
    pass


class FindRel(collections.namedtuple('FindRel', ['elements']), clj.MetaMixin):
    pass


def is_of_size(form, size):
    return clj.is_sequential(form) and len(form) == size


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


# 'unused_placeholder' is a hack used to make the named tuples's objects truthy when comparing with "or" / "and".


class Placeholder(collections.namedtuple('Placeholder', ['unused_placeholder']), clj.MetaMixin):
    pass


class Variable(collections.namedtuple('Variable', ['symbol']), clj.MetaMixin):
    pass


class SrcVar(collections.namedtuple('SrcVar', ['symbol']), clj.MetaMixin):
    pass


class DefaultSrc(collections.namedtuple('DefaultSrc', ['unused_placeholder']), clj.MetaMixin):
    pass


class RulesVar(collections.namedtuple('RulesVar', ['unused_placeholder']), clj.MetaMixin):
    pass


class Constant(collections.namedtuple('Constant', ['value']), clj.MetaMixin):
    pass


class PlainSymbol(collections.namedtuple('Constant', ['symbol']), clj.MetaMixin):
    pass


def parse_placeholder(form):
    if S('_') == form:
        return Placeholder(None)


def parse_variable(form):
    if clj.is_symbol(form) and clj.name(form)[0] == "?":
        return Variable(form)


def parse_src_var(form):
    if clj.is_symbol(form) and clj.name(form)[0] == '$':
        return SrcVar(form)


def parse_rules_var(form):
    if S('%') == form:
        return RulesVar(None)


def parse_constant(form):
    if not clj.is_symbol(form):
        return Constant(form)


def parse_plain_symbol(form):
    if clj.is_symbol(form) and \
       not parse_variable(form) and \
       not parse_src_var(form) and \
       not parse_rules_var(form) and \
       not parse_placeholder(form):
        return PlainSymbol(form)


# fn-arg = (variable | constant | src-var)

def parse_fn_arg(form):
    return parse_variable(form) or \
        parse_constant(form) or \
        parse_src_var(form)


class RuleVars(collections.namedtuple('RuleVars', ['required', 'free']), clj.MetaMixin):
    pass


def parse_rule_vars(form):
    if clj.is_sequential(form):
        if clj.is_sequential(clj.get(form, 0)):
            required = clj.get(form, 0)
            rest = clj.next_(form)
        else:
            required = None
            rest = form
        required_star = parse_seq(parse_variable, required)
        free_star = parse_seq(parse_variable, rest)
        if clj.is_empty(required_star) and clj.is_empty(free_star):
            assert False, "Cannot parse rule-vars, expected [ variable+ | ([ variable+ ] variable*) ]"
        if not is_distinct(required_star + free_star):
            assert False, "Rule variables should be distinct"
        return RuleVars(required_star, free_star)
    assert False, "Cannot parse rule-vars, expected [ variable+ | ([ variable+ ] variable*) ]"

# binding        = (bind-scalar | bind-tuple | bind-coll | bind-rel)
# bind-scalar    = variable
# bind-tuple     = [ (binding | '_')+ ]
# bind-coll      = [ binding '...' ]
# bind-rel       = [ [ (binding | '_')+ ] ]


class BindIgnore(collections.namedtuple('BindIgnore', []), clj.MetaMixin):
    pass


class BindScalar(collections.namedtuple('BindScalar', ['variable']), clj.MetaMixin):
    pass


class BindTuple(collections.namedtuple('BindTuple', ['bindings']), clj.MetaMixin):
    pass


class BindColl(collections.namedtuple('BindColl', ['binding']), clj.MetaMixin):
    pass


def parse_bind_ignore(form):
    if S('_') == form:
        return with_source(BindIgnore(), form)


def parse_bind_scalar(form):
    var = parse_variable(form)
    if var is not None:
        return with_source(BindScalar(var), form)


def parse_bind_coll(form):
    if is_of_size(form, 2) and form[1] == S('...'):
        sub_bind = parse_binding(clj.get(form, 0))
        if sub_bind is not None:
            return with_source(BindColl(sub_bind), form)
        else:
            assert False, "Cannot parse collection binding"


def parse_tuple_el(form):
    return parse_bind_ignore(form) or parse_binding(form)


def parse_bind_tuple(form):
    sub_bindings = parse_seq(parse_tuple_el, form)
    if sub_bindings is not None:
        if not clj.is_empty(sub_bindings):
            return with_source(BindTuple(sub_bindings), form)
        else:
            assert False, "Tuple binding cannot be empty"


def parse_bind_rel(form):
    if is_of_size(form, 1) and clj.is_sequential(clj.get(form, 0)):
        return with_source(BindColl(parse_bind_tuple(clj.get(form, 0))),
                           form)


def parse_binding(form):
    result = \
        parse_bind_coll(form) or \
        parse_bind_rel(form) or \
        parse_bind_tuple(form) or \
        parse_bind_ignore(form) or \
        parse_bind_scalar(form)
    assert result is not None, "Cannot parse binding, expected (bind-scalar | bind-tuple | bind-coll | bind-rel)"
    return result


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


class FindColl(collections.namedtuple('FindColl', ['element']), clj.MetaMixin):
    pass


class FindScalar(collections.namedtuple('FindScalar', ['element']), clj.MetaMixin):
    pass


class FindTuple(collections.namedtuple('FindTuple', ['element']), clj.MetaMixin):
    pass


def parse_find_coll(form):
    if clj.is_sequential(form) and len(form) == 1:
        inner = clj.get(form, 0)
        if clj.is_sequential(inner) and len(inner) == 2 and inner[1] == S('...'):
            element = parse_find_elem(clj.get(inner, 0))
            if element is not None:
                return FindColl(element)


def parse_find_scalar(form):
    if clj.is_sequential(form) and len(form) == 2 and form[1] == S('.'):
        element = parse_find_elem(clj.first(form))
        if element is not None:
            return FindScalar(element)


def parse_find_tuple(form):
    if clj.is_sequential(form) and len(form) == 1:
        inner = clj.first(form)
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


class Pattern(collections.namedtuple('Pattern', ['source', 'pattern']), clj.MetaMixin):
    pass


class Predicate(collections.namedtuple('Predicate', ['fn', 'args']), clj.MetaMixin):
    pass


class Function(collections.namedtuple('Function', ['fn', 'args', 'binding']), clj.MetaMixin):
    pass


class RuleExpr(collections.namedtuple('RuleExpr', ['source', 'name', 'args']), clj.MetaMixin):
    pass


class Not(collections.namedtuple('Not', ['source', 'vars_', 'clauses']), clj.MetaMixin):
    pass


class Or(collections.namedtuple('Or', ['source', 'rule_vars', 'clauses']), clj.MetaMixin):
    pass


class And(collections.namedtuple('And', ['clauses']), clj.MetaMixin):
    pass


def parse_pattern_el(form):
    return parse_placeholder(form) or \
        parse_variable(form) or \
        parse_constant(form)


def take_source(form):
    if clj.is_sequential(form):
        source_star = parse_src_var(clj.first(form))
        if source_star is not None:
            return source_star, clj.next_(form)
        else:
            return DefaultSrc(None), form


def parse_pattern(form):
    source_star_next_form = take_source(form)
    if source_star_next_form is not None:
        source_star, next_form = source_star_next_form
        pattern_star = parse_seq(parse_pattern_el, next_form)
        if pattern_star is not None:
            if not clj.is_empty(pattern_star):
                return with_source(Pattern(source_star, pattern_star), form)
            else:
                assert False, "Pattern could not be empty"


def parse_call(form):
    if clj.is_sequential(form):
        fn, args = clj.extract_seq(form, n_first=1)
        if args is None:
            args = []
        fn_star = parse_plain_symbol(fn) or parse_variable(fn)
        args_star = parse_seq(parse_fn_arg, args)
        if fn_star is not None and args_star is not None:
            return fn_star, args_star


def parse_pred(form):
    if is_of_size(form, 1):
        fn_star_args_star = parse_call(clj.first(form))
        if fn_star_args_star is not None:
            fn_star, args_star = fn_star_args_star
            return tzf.thread_first(Predicate(fn_star, args_star),
                                    (with_source, form))


def parse_fn(form):
    if is_of_size(form, 2):
        call, binding = form
        fn_star_args_star = parse_call(call)
        if fn_star_args_star is not None:
            fn_star, args_star = fn_star_args_star
            binding_star = parse_binding(binding)
            if binding_star is not None:
                return tzf.thread_first(Function(fn_star, args_star, binding_star),
                                        (with_source, form))


def parse_rule_expr(form):
    source_star_next_form = take_source(form)
    if source_star_next_form is not None:
        source_star, next_form = source_star_next_form
        name, args = clj.extract_seq(next_form, n_first=1)
        name_star = parse_plain_symbol(name)
        args_star = parse_seq(parse_pattern_el, args)
        if name_star is not None:
            if clj.is_empty(args):
                assert False, "rule-expr requires at least one argument"
            if args_star is None:
                assert False, "Cannot parse rule-expr arguments, expected [ (variable | constant | '_')+ ]"
            return RuleExpr(source_star, name_star, args_star)


def _collect_vars_acc(acc, form):
    if isinstance(form, Variable):
        return clj.conj(acc, form)
    if isinstance(form, Not):
        return clj.into(acc, form.vars_)
    if isinstance(form, Or):
        return _collect_vars_acc(acc, form.rule_vars)
    if clj.satisfies(ITraversable, form):
        return _collect_vars(form, acc)
    if clj.is_sequential(form):
        return list(clj.reduce(_collect_vars_acc, acc, form))
    return acc


def _collect_vars(form):
    return _collect_vars_acc([], form)


def collect_vars_distinct(form):
    # We cannot use simple list(set(...)), because the order is non-deterministic between interpreter runs
    # (see https://stackoverflow.com/a/21919429/6093101).
    # We can't use sorted(set(...)) either, as the elements aren't always comparable given current implementation
    # of Symbol and Keyword.
    # The hack is to use dict as the insertion order is preserved in Python >=3.7.
    collected = _collect_vars(form)
    collected_dict = {v: None for v in collected}
    return list(collected_dict.keys())


def _validate_join_vars(vars_, clauses, form):
    undeclared = set(vars_).difference(set(_collect_vars(clauses)))
    if not clj.is_empty(undeclared):
        assert False, "Join variable not declared inside clauses: " + repr(clj.mapv(lambda v: v.symbol, undeclared))

    if clj.is_empty(vars_):
        assert False, "Join variables should not be empty"


def _validate_not(clause, form):
    _validate_join_vars(clause.vars_, clause.clauses, form)
    return clause


def parse_not(form):
    source_star_next_form = take_source(form)
    if source_star_next_form is not None:
        source_star, next_form = source_star_next_form
        sym, clauses = clj.extract_seq(next_form, 1)
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


def parse_not_join(form):
    source_star_next_form = take_source(form)
    if source_star_next_form is not None:
        source_star, next_form = source_star_next_form
        sym, vars_, clauses = clj.extract_seq(next_form, n_first=2)
        if sym == S('not-join'):
            vars_star = parse_seq(parse_variable, vars_)
            clauses_star = parse_clauses(clauses)
            if vars_star is not None and clauses_star is not None:
                return tzf.thread_first(Not(source_star, vars_star, clauses_star),
                                        (with_source, form),
                                        (_validate_not, form))
            else:
                assert False, "Cannot parse 'not-join' clause, expected [ src-var? 'not-join' [variable+] clause+ ]"


def validate_or(clause, form):
    required = clause.rule_vars.required
    free = clause.rule_vars.free
    clauses = clause.clauses
    vars_ = clj.concat(required, free)
    for a_clause in clauses:
        _validate_join_vars(vars_, [a_clause], form)
    return clause


def parse_and(form):
    if clj.is_sequential(form) and S('and') == clj.first(form):
        clauses_star = parse_clauses(clj.next_(form))
        if clj.not_empty(clauses_star):
            return And(clauses_star)
        else:
            assert False, "Cannot parse 'and' clause, expected [ 'and' clause+ ]"


def parse_or(form):
    source_star_next_form = take_source(form)
    if source_star_next_form is not None:
        source_star, next_form = source_star_next_form
        # sym = next_form[0]
        # clauses = clj.next_(next_form)
        sym, clauses = clj.extract_seq(next_form, 1)
        if S('or') == sym:
            clauses_star = parse_seq(clj.some_fn(parse_and, parse_clause), clauses)
            if clauses_star is not None:
                return tzf.thread_first(Or(source_star,
                                           RuleVars(None, collect_vars_distinct(clauses_star)),
                                           clauses_star),
                                        (with_source, form),
                                        (validate_or, form))
            else:
                assert False, "Cannot parse 'or' clause, expected [ src-var? 'or' clause+ ]"


def parse_or_join(form):
    source_star_next_form = take_source(form)
    if source_star_next_form is not None:
        source_star, next_form = source_star_next_form
        sym, vars_, clauses = clj.extract_seq(next_form, n_first=2)
        if S('or-join') == sym:
            vars_star = parse_rule_vars(vars_)
            clauses_star = parse_seq(clj.some_fn(parse_and, parse_clause), clauses)
            if vars_star is not None and clauses_star is not None:
                return tzf.thread_first(Or(source_star, vars_star, clauses_star),
                                        (with_source, form),
                                        (validate_or, form))
            else:
                assert False, "Cannot parse 'or-join' clause, expected [ src-var? 'or-join' [variable+] clause+ ]"


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
    return result


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
