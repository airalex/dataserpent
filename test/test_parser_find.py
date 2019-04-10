import src.parser as dp
import src.clj as clj


def test_parse_find():
    # assert dp.parse_find([S('?a'), S('?b')]) \
    #     == dp.FindRel([dp.Variable(S('?a')), dp.Variable(S('?b'))])

    assert dp.parse_find(clj.str2edn('[?a ?b]')) \
        == dp.FindRel([dp.Variable(clj.S('?a')), dp.Variable(clj.S('?b'))])

    # TODO: fix
    # assert dp.parse_find([[S('?a'), S('...')]]) \
    #     == dp.FindColl(dp.Variable(S('?a')))
