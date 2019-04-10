import src.parser as dp
import src.clj as clj


def test_parse_find1():
    assert dp.parse_find(clj.str2edn('[?a ?b]')) \
        == dp.FindRel([dp.Variable(clj.S('?a')), dp.Variable(clj.S('?b'))])


def test_parse_find2():
    assert dp.parse_find(clj.str2edn('[[?a ...]]')) \
        == dp.FindColl(dp.Variable(clj.S('?a')))
