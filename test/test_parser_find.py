import src.parser as dp
from src.clj import S


def test_parse_find():
    assert dp.parse_find([S('?a'), S('?b')]) \
        == dp.FindRel([dp.Variable(S('?a')), dp.Variable(S('?b'))])

    # TODO: fix
    # assert dp.parse_find([[S('?a'), S('...')]]) \
    #     == dp.FindColl(dp.Variable(S('?a')))
