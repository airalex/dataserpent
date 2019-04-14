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


def _q(context, clauses):
    pass


def _collect2(context, symbols):
    rels = context.rels
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
    find_elements = dp.find_elements(find)
    find_vars = dp.find_vars(find)
    result_arity = clj.count(find_elements)
    with_ = parsed_q.qwith
    all_vars = clj.concat(find_vars, map(dp.Variable.symbol, with_))
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
    if q['with'] is not None:
        result = [clj.subvec(e, 0, result_arity) for e in result]
    if clj.some(dp.is_aggregate, find_elements):
        result = aggregate(find_elements, context, result)
    # TODO
    # if clj.some(dp.is_pull, find_elements):
    #     result = pull(find_elements, context, result)
    # result = _post_process(find, result)
    return result
