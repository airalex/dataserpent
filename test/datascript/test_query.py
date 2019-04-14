import toolz.functoolz as tzf

import src.datascript.core as d
import src.datascript.db as db
import src.clj as clj


# deftest test-joins

# TODO
# def test_joins():
#     db = tzf.thread_first(d.empty_db(),
#                           (d.db_with, [{clj.K('db/id'): 1, clj.K('name'): 'Ivan', clj.K('age'): 15},
#                                        {clj.K('db/id'): 2, clj.K('name'): 'Petr', clj.K('age'): 37},
#                                        {clj.K('db/id'): 3, clj.K('name'): 'Ivan', clj.K('age'): 37},
#                                        {clj.K('db/id'): 4, clj.K('age'): 15}]))
