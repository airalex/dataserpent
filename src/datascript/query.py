import collections
import itertools
import importlib

import toolz.functoolz as tzf
import toolz.dicttoolz as tzd
import toolz.itertoolz as tzi

import src.clj as clj
import src.datascript.parser as dp


def parse_rules(rules):
    # No need to parse edn str here, assuming we do it sooner.
    return tzi.groupby(clj.ffirst, rules)




Context = collections.namedtuple('Context', 'rels sources rules')

def looks_like(pattern, form):
    if clj.S('_') == pattern:
        return True
    elif [clj.S('*')] == pattern:
        return clj.is_sequential(form)
    elif clj.is_symbol(pattern):
        return form == pattern
    elif clj.is_sequential(pattern):
        if clj.last(pattern) == clj.S('*'):
            if not clj.is_sequential(form):
                return False
            return clj.is_every(lambda pattern_el_form_el: looks_like(*pattern_el_form_el),
                                clj.mapv(clj.vector, clj.butlast(pattern), form))
        else:
            if not clj.is_sequential(form):
                return False
            if not clj.count(form) == clj.count(pattern):
                return False
            return clj.is_every(lambda pattern_el_form_el: looks_like(*pattern_el_form_el),
                                clj.mapv(clj.vector, pattern, form))
    else:
        return pattern(form)


def is_source(sym):
    return clj.is_symbol(sym) and clj.S('$') == clj.first(sym.name)

def is_rule(context, clause):
    if clj.is_sequential(clause):
        if is_source(clj.first(clause)):
            e = clj.second(clause)
        else:
            e = clj.first(clause)
        return (e in context.rules)
    else:
        return False



# TODO: fill
built_in_aggregates = {}


def resolve_in(context, binding_value):
    binding, value = binding_value
    if clj.is_instance(dp.BindScalar, binding) and \
       clj.is_instance(dp.SrcVar, binding.variable):
        new_sources = tzd.assoc(context.sources, binding.variable.symbol, value)
        return context._replace(sources=new_sources)
    elif clj.is_instance(dp.BindScalar, binding) and \
         clj.is_instance(dp.RulesVar, binding.variable):
        return context._replace(rules=parse_rules(value))
    else:
        new_rels = clj.conj(context.rels, binding.in2rel(value))
        return context._replace(rels=new_rels)


def resolve_ins(context, bindings, values):
    return clj.reduce(resolve_in, context, clj.zipmap(bindings, values))


def _rel_with_attr(context, sym):
    def _pred(r):
        if sym in r.attrs:
            return r

    return clj.some(_pred, context.rels)


def _resolve_sym(sym):
    # adaptation for Python
    fully_qualified = sym.name
    name_elements = fully_qualified.split('.')
    mod_name = '.'.join(name_elements[:-1])
    fn_name = name_elements[-1]
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    return fn


def _context_resolve_val(context, sym):
    rel = _rel_with_attr(context, sym)
    if rel is not None:
        tuple_ = clj.first(rel.tuples)
        if tuple is not None:
            return clj.get(tuple_, clj.get(sym, rel.attrs))

# temporarily not thread-safe
implicit_source_stack = []

def current_implicit_source():
    return clj.get(implicit_source_stack, -1)

def push_implicit_source(new_source):
    implicit_source_stack.append(new_source)

def pop_implicit_source():
    return implicit_source_stack.pop()

def _resolve_clause(context, clause, orig_clause=None):
    if orig_clause is None:
        return _resolve_clause(context, clause, clause)

    if looks_like([[clj.is_symbol, clj.S('*')]], clause):
        return filter_by_pred(context, clause)
    elif looks_like([[clj.is_symbol, clj.S('*')], clj.S('_')]):
        return bind_by_fn(context, clause)
    elif looks_like([is_source, clj.S('*')]):
        source_sym, rest = clj.extract_seq(clause, 1)
        push_implicit_source(clj.get(context.source, source_sym))
        result = _resolve_clause(context, rest, clause)
        pop_implicit_source()
        return result
    elif looks_like([clj.S('or'), clj.S('*')]):
        _, branches = clj.extract_seq(clause, 1)
        contexts = clj.mapv(lambda b: resolve_clause(context, b), branches)
        rels = clj.mapv(lambda c: clj.reduce2(hash_join, c.rels), contexts)
        return clj.first(contexts)._replace(rels=[clj.reduce2(sum_rel, rels)])
    elif looks_like([clj.S('or-join'), [[clj.S('*')], clj.S('*')], clj.S('*')]):
        _, req_vars_vars, branches = clj.extract_seq(clause, 2)
        req_vars, vars_ = clj.extract_seq(req_vars_vars, 1)
        check_bound(context, req_vars, orig_clause)
        return _resolve_clause(context,
                               clj.list_star(clj.S('or-join'), clj.concat(req_vars, vars_), branches),
                                             clause)
    elif looks_like([clj.S('or-join'), [clj.S('*')], clj.S('*')]):
        _, vars_, branches = clj.extract_seq(clause, 2)
        vars_ = clj.set_(vars_)
        join_context = limit_context(context, vars)
        contexts = clj.mapv(lambda b: tzf.thread_first(join_context,
                                                       (resolve_clause, b),
                                                       (limit_context, vars_)), branches)
        rels = clj.mapv(lambda c: clj.reduce2(hash_join, c.rels), contexts)
        sum_rel = clj.reduce2(sum_rel, rels)
        return context._replace(rels=collapse_rels(context.rels, sum_rel))
    elif looks_like([clj.S('and'), clj.S('*')]):
        _, clauses = clj.extract_seq(clause, 1)
        return clj.reduce(resolve_clause, context, clauses)
    elif looks_like([clj.S('not'), clj.S('*')]):
        _, clauses = clj.extract_seq(clause, 1)
        bound_vars = clj.set_(clj.mapcat(lambda c: list(c.attrs.keys()), context.rels))
        negation_vars = collect_vars(clauses)
        if clj.is_empty(bound_vars.intersection(negation_vars)):
            assert False, "Insufficient bindings: none of {} is bound in {}".format(negation_vars, orig_clause)
        context1 = context._replace(rels=[clj.reduce2(hash_join, context.rels)])
        negation_context = clj.reduce(resolve_clause, context1, clauses)
        negation = substract_rel(single(context1.rels),
                                 clj.reduce2(hash_join, negation_context.rels))
        return context1._replace(rels=[negation])
    elif looks_like([clj.S('not-join'), [clj.S('*')], clj.S('*')]):
        _, vars_, clauses = clj.extract_seq(clause, 2)
        check_bound(context, vars_, orig_clause)
        context1 = context._replace(rels=[clj.reduce2(hash_join, context.rels)])
        join_context = limit_context(context1, vars_)
        negation_context = tzf.thread_first(clj.reduce(resolve_clause, join_context, clauses),
                                            (limit_context, vars))
        negation = substract_rel(single(context1.rels),
                                 clj.reduce2(hash_join(negation_context.rels)))
        return context1._replace(rels=[negation])
    elif looks_like([clj.S('*')]):
        source = current_implicit_source()
        pattern = resolve_pattern_lookup_refs(source, clause)
        relation = lookup_pattern(source, pattern)
        if clj.satisfies(db.IDB, source):
            push_lookup_attrs(dynamic_lookup_attrs(source, pattern))
        else:
            push_lookup_attrs(current_lookup_attrs())
        result = context._replace(rels=collapse_rels(context.rels, relation))
        pop_lookup_attrs()
        return result
    else:
        assert False, 'No match found for "looks_like"'


def resolve_clause(context, clause):
    if is_rule(context, clause):
        if is_source(clj.first(clause)):
            push_implicit_source(clj.get(context.sources, clj.first(clause)))
            result = resolve_clause(context, clj.next_(clause))
            pop_implicit_source()
            return result
        else:
            return context._replace(rels=collapse_rels(context.rels, solve_rule(context, clause)))
    else:
        return _resolve_clause(context, clause)


def _q(context, clauses):
    implicit_source = clj.get(context.sources, clj.S('$'))
    push_implicit_source(implicit_source)
    result = clj.reduce(resolve_clause, context, clauses)
    pop_implicit_source()
    return result


def _collect2(context, symbols):
    rels = context.rels if context is not None else None
    # Temporarily using a plain list for acc instead of custom tonsky's array
    array = []
    return _collect3([array], rels, symbols)


def _collect3(acc, rels, symbols):
    if clj.first(rels) is not None:
        rel = clj.first(rels)
        keep_attrs = clj.select_keys(rel.attrs, symbols)
        if clj.is_empty(keep_attrs):
            return _collect3(acc, clj.next_(rels), symbols)
        else:
            copy_map = list(map(lambda s: clj.get(keep_attrs, s), symbols))
            len_ = clj.count(symbols)

            results = []
            for t1 in acc:
                for t2 in rel.tuples:
                    res = list(t1)
                    for i in len_:
                        if clj.get(copy_map, i) is not None:
                            idx = clj.get(copy_map, i)
                            res[i] = clj.get(t2, idx)
                    results.append(res)
            return _collect3(results, clj.next_(rels), symbols)


def collect(context, symbols):
    return tzf.thread_last(_collect2(context, symbols),
                           clj.liberal_iter,
                           (map, clj.vec),
                           clj.set_)


def _context_resolve_Variable(var, context):
    return _context_resolve_val(context, var.symbol)


def _context_resolve_SrcVar(var, context):
    return getattr(context.sources, var.symbol)


def _context_resolve_PlainSymbol(var, _):
    return clj.get(built_in_aggregates, var.symbol) or \
        _resolve_sym(var.symbol)


def _context_resolve_Constant(var, _):
    return var.value


dp.Variable.context_resolve = _context_resolve_Variable
dp.SrcVar.context_resolve = _context_resolve_SrcVar
dp.PlainSymbol.context_resolve = _context_resolve_PlainSymbol
dp.Constant.context_resolve = _context_resolve_Constant


def _aggregate(find_elements, context, tuples):
    def _mapper(element, fixed_value, i):
        if dp.is_aggregate(element):
            f = element.fn.context_resolve(context)
            args = map(lambda e: e.context_resolve(context), clj.butlast(element.args))
            vals = map(lambda t: t[i], tuples)
            return f(clj.concat(args, [vals]))
        else:
            return fixed_value

    return clj.mapv(_mapper,
                    find_elements,
                    clj.first(tuples),
                    itertools.count())


def _idxs_of(pred, coll):
    return tzf.thread_last(zip(coll, itertools.count()),
                           (map, lambda args: args[1] if pred(args[0]) else None),
                           (tzi.remove, clj.is_nil),
                           list)


def aggregate(find_elements, context, resultset):
    group_idxs = _idxs_of(clj.complement(dp.is_aggregate), find_elements)

    def group_fn(tuple_):
        return map(lambda idx: tuple_[idx], group_idxs)

    grouped = tzi.groupby(group_fn, resultset)
    return [
        _aggregate(find_elements, context, tuples) for _, tuples in grouped
    ]


def memoized_parse_query(q):
    # TODO: memoize
    return dp.parse_query(q)


def q(q, *inputs):
    parsed_q = memoized_parse_query(q)
    find = parsed_q.qfind
    find_elements = find.find_elements()
    find_vars = dp.find_vars(find)
    result_arity = clj.count(find_elements)
    with_ = parsed_q.qwith
    all_vars = clj.concat(find_vars, clj.mapv(dp.Variable.symbol, with_))
    q0 = q
    if clj.is_sequential(q):
        q = dp.query2map(q)
    wheres = q['where']
    context = tzf.thread_first(Context([], {}, {}),
                               (resolve_ins, parsed_q.qin, inputs))
    resultset = tzf.thread_first(context,
                                 (_q, wheres),
                                 (collect, all_vars))

    result = resultset
    if 'with' in q:
        result = [clj.subvec(e, 0, result_arity) for e in result]
    if clj.some(dp.is_aggregate, find_elements):
        result = aggregate(find_elements, context, result)
    # TODO
    # if clj.some(dp.is_pull, find_elements):
    #     result = pull(find_elements, context, result)
    # result = _post_process(find, result)
    return result
