import pytest

import src.parser as dp
import src.clj as clj


@pytest.mark.parametrize('q,msg', [('[:find ?e :where [?x]]',
                                    "Query for unknown vars: [Symbol(?e)]"),
                                   ('[:find ?e :with ?f :where [?e]]',
                                    "Query for unknown vars: [Symbol(?f)]"),
                                   ('[:find ?e ?x ?t :in ?x :where [?e]]',
                                    "Query for unknown vars: [Symbol(?t)]"),
                                   ('[:find ?x ?e :with ?y ?e :where [?x ?e ?y]]',
                                    ":find and :with should not use same variables: [Symbol(?e)]"),
                                   ('[:find ?e :in $ $ ?x :where [?e]]',
                                    "Vars used in :in should be distinct")])
def test_validation(q, msg):
    query = clj.str2edn(q)
    with pytest.raises(AssertionError) as excinfo:
        dp.parse_query(query)
    assert str(excinfo.value) == msg
