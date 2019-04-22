import collections
import abc

import toolz.dicttoolz as tzd
import toolz.functoolz as tzf
import toolz.curried as tzc
import sortedcontainers as sc

import src.clj as clj


E0 = 0
TX0 = 0x20000000
EMAX = 0x7FFFFFFF
TMAX = 0x7FFFFFFF
IMPLICIT_SCHEMA = {'db/ident': {'db/unique': 'db.unique/identity'}}


def is_seqable(x):
    if isinstance(x, str):
        return False

    # is_nil() isn't included here, as Python generally doesn't allow iterating over None
    return clj.is_seq(x)


class Datom(collections.namedtuple('Datom', 'e a v tx added')):
    def datom_tx(self):
        if clj.is_pos(self.tx):
            return self.tx
        else:
            return -self.tx


class ISearch(abc.ABC):
    @abc.abstractmethod
    def search(self, pattern):
        pass


# Uncomment when needs to be used
# class IIndexAccesss(abc.ABC):
#     @abc.abstractmethod
#     def datoms(self, index, components):
#         pass

#     @abc.abstractmethod
#     def seek_datoms(self, index, components):
#         pass

#     @abc.abstractmethod
#     def rseek_datoms(self, index, components):
#         pass

#     @abc.abstractmethod
#     def index_range(self, attr, start, end):
#         pass


class IDB(abc.ABC):
    @abc.abstractmethod
    def schema(self):
        pass

    @abc.abstractmethod
    def attrs_by(self, property_):
        pass


class DB(collections.namedtuple('DB', 'schema_ eavt aevt avet max_eid max_tx rschema hash_'),
         IDB, ISearch):
    def schema(self):
        return self.schema_

    def attrs_by(self, property_):
        return clj.get(self.rschema, property_)

    def search(self, pattern):
        e, a, v, tx, _ = clj.extract_seq(pattern, 4)

        filters = []
        if e is not None:
            filters.append(tzc.filter(lambda d: d.e == e))
        if a is not None:
            filters.append(tzc.filter(lambda d: d.a == a))
        if v is not None:
            filters.append(tzc.filter(lambda d: d.v == v))
        if tx is not None:
            filters.append(tzc.filter(lambda d: d.tx == tx))

        datoms = list(tzf.thread_last(self.eavt, *filters))
        return datoms

        # eavt = self.eavt
        # aevt = self.aevt
        # avet = self.avet
        # Very naive implementation for now.
        # Datascript uses `case-tree` here for searching indices.
        #
        # if e is not None:
        #     if a is not None:
        #         if v is not None:
        #             if tx is not None:
        #                 pass
        #             else:
        #                 pass
        #         else:
        #             pass
        #     else:
        #         pass
        # else:
        #     if a is not None:
        #         if v is not None:
        #             if tx is not None:
        #                 pass
        #             else:
        #                 pass
        #         else:
        #             if tx is not None:
        #                 pass
        #             else:
        #                 # TODO: fix *** TypeError: '>' not supported between instances of 'Keyword' and 'Keyword'
        #                 # return filter(tzf.curry(cmp_datoms_aevt,
        #                 #                         Datom(e=None, a=a, v=None, tx=None, added=None)),
        #                 #               aevt)
        #                 return [d for d in aevt if d.a == a]


FilteredDB = collections.namedtuple('FilteredDB', 'unfiltered_db pred hash_')
TxReport = collections.namedtuple('TxReport', 'db_before db_after tx_data tempids tx_meta')


def is_attr(db, attr, property_):
    return clj.contains(db.attrs_by(property_), attr)


def is_ref(db, attr):
    return is_attr(db, attr, 'db.type/ref')


def combine_cmp(*comps):
    def _loop(comps, res):
        if clj.not_empty(comps):
            c = clj.first(comps)
            if c == 0:
                next_res = res
            else:
                next_res = c
            return _loop(clj.next_(comps), next_res)
    return _loop(clj.reverse(comps), 0)


def cmp_(o1, o2):
    if clj.is_nil(o1):
        return 0
    else:
        if clj.is_nil(o2):
            return 0
        else:
            return clj.compare(o1, o2)


def cmp_datoms_eavt(d1, d2):
    return combine_cmp(
        clj.integer_compare(d1.e, d2.e),
        cmp_(d1.a, d2.a),
        cmp_(d1.v, d2.v),
        clj.integer_compare(d1.datom_tx(), d2.datom_tx()))


def cmp_datoms_aevt(d1, d2):
    return combine_cmp(
        cmp_(d1.a, d2.a),
        clj.integer_compare(d1.e, d2.e),
        cmp_(d1.v, d2.v),
        clj.integer_compare(d1.datom_tx(), d2.datom_tx()))


def cmp_datoms_avet(d1, d2):
    return combine_cmp(
        cmp_(d1.a, d2.a),
        cmp_(d1.v, d2.v),
        clj.integer_compare(d1.e, d2.e),
        clj.integer_compare(d1.datom_tx(), d2.datom_tx()))


def datom_cmp_eavt(datom):
    return [datom.e, datom.a, datom.v, datom.datom_tx()]


def datom_cmp_aevt(datom):
    return [datom.a, datom.e, datom.v, datom.datom_tx()]


def datom_cmp_avet(datom):
    return [datom.a, datom.v, datom.e, datom.datom_tx()]


def attr2properties(k, v):
    if v == 'db.unique/identity':
        return ['db/unique', 'db.unique/identity', 'db/index']
    elif v == 'db.unique/value':
        return ['db/unique', 'db.unique/value', 'db/index']
    elif v == 'db.cardinality/many':
        return ['db.cardinality/many']
    elif v == 'db.type/ref':
        return ['db.type/ref', 'db/index']
    elif v is True:
        if k == 'db/isComponent':
            return ['db/isComponent']
        elif k == 'db/index':
            return ['db/index']
        else:
            return []


def _rschema(schema):
    def _reducer1(m, attr_keys2values):
        attr, keys2values = attr_keys2values

        def _reducer2(m, key_value):
            key, value = key_value

            def _reducer3(m, prop):
                return tzd.assoc(m, prop, clj.conj(clj.get(m, prop, set()), attr))

            return clj.reduce(_reducer3, m, attr2properties(key, value))

        return clj.reduce(_reducer2, m, keys2values)

    return clj.reduce(_reducer1, {}, schema)


def _validate_schema_key(a, k, v, expected):
    if not (clj.is_nil(v) or v in expected):
        assert False, "Bad attribute specification for {}, expected one of {}".format({a: {k: v}}, expected)


def _validate_schema(schema):
    for a, kv in clj.liberal_iter(schema):
        is_comp = clj.get(kv, 'db/isComponent', False)
        _validate_schema_key(a, 'db/isComponent', clj.get(kv, 'db/isComponent'), {True, False})
        if is_comp and not clj.get(kv, 'db/valueType') == 'db.type/ref':
            assert False, "Bad attribute specification for {}:" \
                " {:db/isComponent true} should also have {:db/valueType :db.type/ref}".format(a)
        _validate_schema_key(a, 'db/unique', clj.get(kv, 'db/unique'), {'db.unique/value', 'db.unique/identity'})
        _validate_schema_key(a, 'db/valueType', clj.get(kv, 'db/valueType'), {'db.type/ref'})
        _validate_schema_key(a, 'db/cardinality', clj.get(kv, 'db/cardinality'), {'db.cardinality/one',
                                                                                  'db.cardinality/many'})


def empty_db(schema=None):
    """Creates an empty database with an optional schema."""
    assert clj.is_nil(schema) or clj.is_map(schema)
    _validate_schema(schema)
    # return DB(schema=schema,
    #           rschema=_rschema(clj.merge(IMPLICIT_SCHEMA, schema)),
    #           eavt=sc.SortedSet(key=datom_cmp_eavt),
    #           aevt=sc.SortedSet(key=datom_cmp_aevt),
    #           avet=sc.SortedSet(key=datom_cmp_avet),
    #           max_eid=E0,
    #           max_tx=TX0,
    #           hash_=clj.atom(0))
    return DB(schema=schema,
              rschema=_rschema(clj.merge(IMPLICIT_SCHEMA, schema)),
              eavt=[],
              aevt=[],
              avet=[],
              max_eid=E0,
              max_tx=TX0,
              hash_=clj.atom(0))


def init_db(datoms, schema=None):
    _validate_schema(schema)
    rschema = _rschema(clj.merge(IMPLICIT_SCHEMA, schema))
    # indexed = rschema['db/index']
    # eavt = sc.SortedSet(datoms, datom_cmp_eavt)
    # aevt = sc.SortedSet(datoms, datom_cmp_aevt)
    # avet_datoms = list(filter(lambda d: d.a in indexed, datoms))
    # avet = sc.SortedSet(avet_datoms, datom_cmp_avet)
    max_eid = 100  # TODO init_max_eid(eavt)
    max_tx = 1000*1000  # TODO
    # return DB(schema_=schema,
    #           rschema=rschema,
    #           eavt=eavt,
    #           aevt=aevt,
    #           avet=avet,
    #           max_eid=max_eid,
    #           max_tx=max_tx,
    #           hash_=clj.atom(0))
    return DB(schema_=schema,
              rschema=rschema,
              eavt=list(datoms),
              aevt=list(datoms),
              avet=list(datoms),
              max_eid=max_eid,
              max_tx=max_tx,
              hash_=clj.atom(0))


def is_db(x):
    # TODO: use more polimorphic introspection
    return clj.is_instance(DB, x)


def transact_tx_data(initial_report, initial_es):
    raise NotImplementedError()
