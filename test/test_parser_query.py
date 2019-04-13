import pytest

import src.parser as dp
import src.clj as clj


@pytest.mark.parametrize('q,msg', [('[:find ?e :where [?x]]',
                                    "Query for unknown vars: [Symbol(?e)]"),
                                   ('[:find ?e :with ?f :where [?e]]',
                                    "Query for unknown vars: [Symbol(?f)]")])
def test_validation(q, msg):
    query = clj.str2edn(q)
    with pytest.raises(AssertionError) as excinfo:
        dp.parse_query(query)
    assert str(excinfo.value) == msg
