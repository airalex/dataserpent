import toolz.functoolz as tzf

import src.datascript.core as d
import src.datascript.query as dq
import src.datascript.db as ddb
import src.clj as clj


# deftest test-joins

def test_joins():
    # TODO set proper db
    # db = tzf.thread_first(d.empty_db(),
    #                       (d.db_with, [{clj.K('db/id'): 1, clj.K('name'): 'Ivan', clj.K('age'): 15},
    #                                    {clj.K('db/id'): 2, clj.K('name'): 'Petr', clj.K('age'): 37},
    #                                    {clj.K('db/id'): 3, clj.K('name'): 'Ivan', clj.K('age'): 37},
    #                                    {clj.K('db/id'): 4, clj.K('age'): 15}]))
    # db = ddb.init_db([ddb.Datom(1, 'name', 'Asia', 1, True),
    #                   ddb.Datom(2, 'name', 'Alex', 1, True)])
    db = ddb.init_db([ddb.Datom(1, clj.K('name'), 'Asia', 1, True),
                      ddb.Datom(2, clj.K('name'), 'Alex', 1, True)])
    form = clj.str2edn('[:find ?e :where [?e :name]]')
    result = dq.q(form, db)
    # assert {1, 2} == result
    assert {(1,), (2,)} == result


class TestQuerying():
    def test_returning_v(self):
        db = ddb.init_db([ddb.Datom(1, clj.K('name'), 'Asia', 1, True),
                          ddb.Datom(2, clj.K('name'), 'Alex', 1, True)])
        form = clj.str2edn('[:find ?n :where [_ :name ?n]]')
        result = dq.q(form, db)
        assert {('Asia',), ('Alex',)} == result

    def test_e_contrained_to_v(self):
        db = ddb.init_db([ddb.Datom(1, clj.K('name'), 'Asia', 1, True),
                          ddb.Datom(2, clj.K('name'), 'Alex', 1, True)])
        form = clj.str2edn('[:find ?e :where [?e :name "Alex"]]')
        result = dq.q(form, db)
        assert {(2,)} == result

    def test_v_joined_by_e(self):
        db = ddb.init_db([ddb.Datom(1, clj.K('name'), 'Asia', 1, True),
                          ddb.Datom(2, clj.K('name'), 'Alex', 1, True),
                          ddb.Datom(2, clj.K('friend'), 1, 1, True)])
        form = clj.str2edn('''
        [:find ?f
         :where [?e :name "Alex"]
                [?e :friend ?f]]
        ''')
        result = dq.q(form, db)
        assert {(1,)} == result

    def test_v_of_join_joined_constrained(self):
        db = ddb.init_db([ddb.Datom(1, clj.K('name'), 'Asia', 1, True),
                          ddb.Datom(1, clj.K('gender'), 'f', 1, True),
                          ddb.Datom(2, clj.K('name'), 'Alex', 1, True),
                          ddb.Datom(2, clj.K('gender'), 'm', 1, True),
                          ddb.Datom(3, clj.K('name'), 'Artur', 1, True),
                          ddb.Datom(3, clj.K('gender'), 'm', 1, True),
                          ddb.Datom(1, clj.K('friends'), 2, 1, True),
                          ddb.Datom(2, clj.K('friends'), 1, 1, True),
                          ddb.Datom(2, clj.K('friends'), 3, 1, True),
                          ddb.Datom(3, clj.K('friends'), 2, 1, True),
                          ddb.Datom(4, clj.K('name'), 'Jane', 1, True)])
        assert {('Alex',), ('Artur',), ('Asia',)} == \
            dq.q(clj.str2edn('''
            [:find ?n
             :where
            [?e :gender "m"]
            [?e :friends ?f]
            [?f :name ?n]]
            '''), db)


class TestLooksLike:
    def test_is_symbol_star(self):
        clause = [[clj.S('?e'), clj.K('name')]]
        res = dq.looks_like([dq.is_source, clj.S('*')], clause)
        assert res is False

    def test_star(self):
        clause = [clj.S('*')]
        res = dq.looks_like([clj.S('_')], clause)
        assert res is True
