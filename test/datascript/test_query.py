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
    assert {1, 2} == result


class TestLooksLike:
    def test_is_symbol_star(self):
        clause = [[clj.S('?e'), clj.K('name')]]
        res = dq.looks_like([dq.is_source, clj.S('*')], clause)
        assert res is False

    def test_star(self):
        clause = [clj.S('*')]
        res = dq.looks_like([clj.S('_')], clause)
        assert res is True
