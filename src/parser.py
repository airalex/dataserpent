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
import src.db as db


# implementation based on
# https://github.com/tonsky/datascript/blob/aa67e7a4d99b954a357b0c6533bd7039f5d99e54/src/datascript/parser.cljc


class ITraversable(abc.ABC):
    @abc.abstractmethod
    def _collect(self, pred, acc):
        pass

    @abc.abstractmethod
    def _collect_vars(self, acc):
        pass

    @abc.abstractmethod
    def _postwalk(self, f):
        pass


def validate_arity(name, branches):
    vars0 = clj.first(branches).vars_
    arity0 = rule_vars_arity(vars0)
    for b in clj.next_(branches) or []:
        vars_ = b.vars_
        if not arity0 == rule_vars_arity(vars_):
            assert False, "Arity mismatch for rule {}: {} vs. {}'".format(name.symbol,
                                                                          flatten_rule_vars(vars0),
                                                                          flatten_rule_vars(vars_))


def parse_rules(form):
    rules = []
    for name, branches in tzi.groupby(lambda n: n['name'], parse_seq(parse_rule, form)).items():
        branches_star = [RuleBranch(b['vars'], b['clauses']) for b in branches]
        validate_arity(name, branches_star)
        rules.append(Rule(name, branches_star))
    return rules


class Query(collections.namedtuple('Query', ['qfind', 'qwith', 'qin', 'qwhere']), clj.MetaMixin):
    pass


def query2map(query: list) -> dict:
    def _loop(parsed: dict, key, qs: list):
        q = clj.first(qs)
        if q is not None:
            if clj.is_keyword(q):
                return _loop(parsed, q, clj.next_(qs))
            else:
                return _loop(tzd.update_in(parsed,
                                           [key.name],
                                           clj.fnil(lambda l: clj.conj(l, q), [])),
                             key,
                             clj.next_(qs))
        else:
            return parsed

    return _loop({}, None, query)


def validate_query(q: Query, form):
    def _validate_unknown_shared():
        find_vars = clj.set_(_collect_vars(q.qfind))
        with_vars = clj.set_(q.qwith)
        in_vars = clj.set_(_collect_vars(q.qin))
        where_vars = clj.set_(_collect_vars(q.qwhere))
        unknown = find_vars.union(with_vars).difference(where_vars.union(in_vars))
        shared = find_vars.intersection(with_vars)

        if not clj.is_empty(unknown):
            assert False, "Query for unknown vars: {}".format([v.symbol for v in unknown])

        if not clj.is_empty(shared):
            assert False, ":find and :with should not use same variables: {}".format([v.symbol for v in shared])
    _validate_unknown_shared()

    def _validate_distinct_in():
        in_vars = _collect_vars(q.qin)
        in_sources = collect(lambda e: isinstance(e, SrcVar), q.qin)
        in_rules = collect(lambda e: isinstance(e, RulesVar), q.qin)
        if not (is_distinct(in_vars) and
                is_distinct(in_sources) and
                is_distinct(in_rules)):
            assert False, "Vars used in :in should be distinct"
    _validate_distinct_in()

    def _validate_distinct_with():
        with_vars = _collect_vars(q.qwith)
        if not is_distinct(with_vars):
            assert False, "Vars used in :with should be distinct"
    _validate_distinct_with()

    def _validate_sources():
        in_sources = collect(lambda e: isinstance(e, SrcVar), q.qin, set())
        where_sources = collect(lambda e: isinstance(e, SrcVar), q.qwhere, set())
        unknown = where_sources.difference(in_sources)
        if not clj.is_empty(unknown):
            assert False, "Where uses unknown source vars: {}".format([v.symbol for v in unknown])
    _validate_sources()

    def _validate_rules_vars():
        rule_exprs = collect(lambda e: isinstance(e, RuleExpr), q.qwhere)
        rules_vars = collect(lambda e: isinstance(e, RulesVar), q.qin)
        if (not clj.is_empty(rule_exprs)) and clj.is_empty(rules_vars):
            assert False, "Missing rules var '%' in :in"
    _validate_rules_vars()


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


def collect(pred, form, acc=[]):
    if pred(form):
        return clj.conj(acc, form)
    elif clj.satisfies(ITraversable, form):
        return form._collect(pred, acc)
    elif db.is_seqable(form):
        return clj.reduce(lambda acc, form: collect(pred, form, acc),
                          acc,
                          form)
    else:
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


def parse_plain_variable(form):
    if parse_plain_symbol(form) is not None:
        return Variable(form)


# fn-arg = (variable | constant | src-var)

def parse_fn_arg(form):
    return parse_variable(form) or \
        parse_constant(form) or \
        parse_src_var(form)


class RuleVars(collections.namedtuple('RuleVars', ['required', 'free']), clj.MetaMixin):
    pass


def parse_rule_vars(form):
    if clj.is_sequential(form):
        if clj.is_sequential(clj.first(form)):
            required = clj.first(form)
            rest = clj.next_(form)
        else:
            required = None
            rest = form
        required_star = parse_seq(parse_variable, required)
        free_star = parse_seq(parse_variable, rest)
        if clj.is_empty(required_star) and clj.is_empty(free_star):
            assert False, "Cannot parse rule-vars, expected [ variable+ | ([ variable+ ] variable*) ]"
        if not is_distinct(clj.concat(required_star, free_star)):
            assert False, "Rule variables should be distinct"
        return RuleVars(required_star, free_star)
    assert False, "Cannot parse rule-vars, expected [ variable+ | ([ variable+ ] variable*) ]"


def flatten_rule_vars(rule_vars):
    if rule_vars.required is not None:
        req_symbols = [[v.symbol for v in rule_vars.required]]
    else:
        req_symbols = None
    return clj.concat(req_symbols, [v.symbol for v in clj.liberal_iter(rule_vars.free)])


def rule_vars_arity(rule_vars):
    return clj.count(rule_vars.required), clj.count(rule_vars.free)


# binding        = (bind-scalar | bind-tuple | bind-coll | bind-rel)
# bind-scalar    = variable
# bind-tuple     = [ (binding | '_')+ ]
# bind-coll      = [ binding '...' ]
# bind-rel       = [ [ (binding | '_')+ ] ]


class BindIgnore(collections.namedtuple('BindIgnore', ['unused_placeholder']), clj.MetaMixin):
    pass


class BindScalar(collections.namedtuple('BindScalar', ['variable']), clj.MetaMixin):
    pass


class BindTuple(collections.namedtuple('BindTuple', ['bindings']), clj.MetaMixin):
    pass


class BindColl(collections.namedtuple('BindColl', ['binding']), clj.MetaMixin):
    pass


def parse_bind_ignore(form):
    if S('_') == form:
        return with_source(BindIgnore(None), form)


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


class Aggregate(collections.namedtuple('Aggregate', ['fn', 'args']), clj.MetaMixin):
    pass


class Pull(collections.namedtuple('Pull', ['source', 'variable', 'pattern']), clj.MetaMixin):
    pass


class FindRel(collections.namedtuple('FindRel', ['elements']), clj.MetaMixin):
    pass


class FindColl(collections.namedtuple('FindColl', ['element']), clj.MetaMixin):
    pass


class FindScalar(collections.namedtuple('FindScalar', ['element']), clj.MetaMixin):
    pass


class FindTuple(collections.namedtuple('FindTuple', ['element']), clj.MetaMixin):
    pass


def parse_aggregate(form):
    if clj.is_sequential(form) and (len(form) >= 2):
        fn, args = clj.extract_seq(form, 1)
        fn_star = parse_plain_symbol(fn)
        args_star = parse_seq(parse_fn_arg, args)
        if fn_star is not None and args_star is not None:
            return Aggregate(fn_star, args_star)


def parse_aggregate_custom(form):
    if clj.is_sequential(form) and clj.first(form) == clj.S('aggregate'):
        if len(form) >= 3:
            _, fn, args = clj.extract_seq(form, 2)
            fn_star = parse_variable(fn)
            args_star = parse_seq(parse_fn_arg, args)
            if fn_star is not None and args_star is not None:
                return Aggregate(fn_star, args_star)
            else:
                assert False, "Cannot parse custom aggregate call, expect ['aggregate' variable fn-arg+]"
        else:
            assert False, "Cannot parse custom aggregate call, expect ['aggregate' variable fn-arg+]"


def parse_pull_expr(form):
    if clj.is_sequential(form) and clj.first(form) == clj.S('pull'):
        if 3 <= len(form) <= 4:
            is_long = (len(form) == 4)
            if is_long:
                src = form[1]
                var, pattern = clj.nnext(form)
            else:
                src = clj.S('$')
                var, pattern = clj.next_(form)
            src_star = parse_src_var(src)
            var_star = parse_variable(var)
            pattern_star = parse_variable(pattern) or \
                parse_plain_variable(pattern) or \
                parse_constant(pattern)

            if src_star and var_star and pattern_star:
                return Pull(src_star, var_star, pattern_star)
            else:
                assert False, "Cannot parse pull expression, expect" \
                    "['pull' src-var? variable (constant | variable | plain-symbol)]"
        else:
            assert False, "Cannot parse pull expression, expect ['pull' src-var? variable" \
                "(constant | variable | plain-symbol)]"


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


def parse_find(form):
    result = \
        parse_find_rel(form) or \
        parse_find_coll(form) or \
        parse_find_scalar(form) or \
        parse_find_tuple(form)
    assert result is not None, 'Cannot parse :find, expected: (find-rel | find-coll | find-tuple | find-scalar)'
    return result


def parse_with(form):
    result = parse_seq(parse_variable, form)
    assert result is not None, "Cannot parse :with clause, expected [ variable+ ]"
    return result


def _parse_in_binding(form):
    var = parse_src_var(form) or \
        parse_rules_var(form) or \
        parse_plain_variable(form)
    if var is not None:
        return with_source(BindScalar(var), form)
    else:
        return parse_binding(form)


def parse_in(form):
    result = parse_seq(_parse_in_binding, form)
    assert result is not None, \
        "Cannot parse :in clause, expected " \
        "(src-var | % | plain-symbol | bind-scalar | bind-tuple | bind-coll | bind-rel)"
    return result


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


def parse_where(form):
    result = parse_clauses(form)
    assert result is not None, "Cannot parse :where clause, expected [clause+]"
    return result


class RuleBranch(collections.namedtuple('RuleBranch', 'vars_ clauses'), clj.MetaMixin):
    pass


class Rule(collections.namedtuple('Rule', 'name clauses'), clj.MetaMixin):
    pass


def validate_vars(vars_, clauses, form):
    declared_vars = collect(lambda o: clj.is_instance(Variable, o), vars_, set())
    used_vars = collect(lambda o: clj.is_instance(Variable, o), clauses, set())
    undeclared_vars = used_vars.difference(declared_vars)
    if not clj.is_empty(undeclared_vars):
        assert False, "Reference to the unknown variables: {}".format([v.symbol for v in undeclared_vars])


def parse_rule(form):
    if clj.is_sequential(form):
        head, clauses = clj.extract_seq(form, 1)
        if clj.is_sequential(head):
            name, vars_ = clj.extract_seq(head, 1)
            name_star = parse_plain_symbol(name)
            assert name_star is not None, "Cannot parse rule name, expected plain-symbol"
            vars_star = parse_rule_vars(vars_)
            clauses_star = clj.not_empty(parse_clauses(clauses))
            assert clauses_star is not None, "Rule branch should have clauses"

            validate_vars(vars_star, clauses_star, form)
            return {'name': name_star,
                    'vars': vars_star,
                    'clauses': clauses_star}
        else:
            assert False, "Cannot parse rule head, expected [rule-name rule-vars]"
    else:
        assert False, "Cannot parse rule, expected [rule-head clause+]"


def parse_query(q: t.Union[list, dict]):
    if clj.is_map(q):
        qm = q
    elif clj.is_sequential(q):
        qm = query2map(q)
    else:
        assert False, "Query should be a vector or a map"

    if qm.get('with') is not None:
        with_ = parse_with(qm['with'])
    else:
        with_ = None

    res = Query(qfind=parse_find(qm['find']),
                qwith=with_,
                qin=parse_in(clj.get(qm, 'in', [clj.S('$')])),
                qwhere=parse_where(clj.get(qm, 'where', [])))
    validate_query(res, q)
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
