import src.parser as dp
import src.clj as clj


# test-parse-find

def test_parse_find_rel():
    assert dp.parse_find(clj.str2edn('[?a ?b]')) \
        == dp.FindRel([dp.Variable(clj.S('?a')), dp.Variable(clj.S('?b'))])


def test_parse_find_coll():
    assert dp.parse_find(clj.str2edn('[[?a ...]]')) \
        == dp.FindColl(dp.Variable(clj.S('?a')))


def test_parse_find_scalar():
    assert dp.parse_find(clj.str2edn('[?a .]')) \
        == dp.FindScalar(dp.Variable(clj.S('?a')))


def test_parse_find_tuple():
    assert dp.parse_find(clj.str2edn('[[?a ?b]]')) \
        == dp.FindTuple([dp.Variable(clj.S('?a')), dp.Variable((clj.S('?b')))])
