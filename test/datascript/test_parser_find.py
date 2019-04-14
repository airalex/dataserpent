import src.datascript.parser as dp
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


# deftest test-parse-aggregate

def test_parse_aggregate_a_count_b():
    assert dp.parse_find(clj.str2edn('[?a (count ?b)]')) \
        == dp.FindRel([dp.Variable(clj.S('?a')),
                       dp.Aggregate(dp.PlainSymbol(clj.S('count')),
                                    [dp.Variable(clj.S('?b'))])])


def test_parse_aggregate_count_a_ellipsis():
    assert dp.parse_find(clj.str2edn('[[(count ?a) ...]]')) \
        == dp.FindColl(dp.Aggregate(dp.PlainSymbol(clj.S('count')),
                                    [dp.Variable(clj.S('?a'))]))


def test_parse_aggregate_count_a_dot():
    assert dp.parse_find(clj.str2edn('[(count ?a) .]')) \
        == dp.FindScalar(dp.Aggregate(dp.PlainSymbol(clj.S('count')),
                                      [dp.Variable(clj.S('?a'))]))


def test_parse_aggregate_count_a_b():
    assert dp.parse_find(clj.str2edn('[[(count ?a) ?b]]')) \
        == dp.FindTuple([dp.Aggregate(dp.PlainSymbol(clj.S('count')), [dp.Variable(clj.S('?a'))]),
                         dp.Variable(clj.S('?b'))])


# deftest test-parse-custom-aggregates

def test_parse_custom_agg_f_a():
    assert dp.parse_find(clj.str2edn('[(aggregate ?f ?a)]')) \
        == dp.FindRel([dp.Aggregate(dp.Variable(clj.S('?f')),
                                    [dp.Variable(clj.S('?a'))])])


def test_parse_custom_a_agg_f_b():
    assert dp.parse_find(clj.str2edn('[?a (aggregate ?f ?b)]')) \
        == dp.FindRel([dp.Variable(clj.S('?a')),
                       dp.Aggregate(dp.Variable(clj.S('?f')),
                                    [dp.Variable(clj.S('?b'))])])


def test_parse_custom_agg_f_a_ellipsis():
    assert dp.parse_find(clj.str2edn('[[(aggregate ?f ?a) ...]]')) \
        == dp.FindColl(dp.Aggregate(dp.Variable(clj.S('?f')),
                                    [dp.Variable(clj.S('?a'))]))


def test_parse_custom_agg_f_a_dot():
    assert dp.parse_find(clj.str2edn('[(aggregate ?f ?a) .]')) \
        == dp.FindScalar(dp.Aggregate(dp.Variable(clj.S('?f')),
                                      [dp.Variable(clj.S('?a'))]))


def test_parse_custom_agg_f_a_b():
    assert dp.parse_find(clj.str2edn('[[(aggregate ?f ?a) ?b]]')) \
        == dp.FindTuple([dp.Aggregate(dp.Variable(clj.S('?f')),
                                      [dp.Variable(clj.S('?a'))]),
                         dp.Variable(clj.S('?b'))])


# deftest test-parse-find-element

def test_parse_find_elems_count_b_1_x_dot():
    assert dp.parse_find(clj.str2edn('[(count ?b 1 $x) .]')) \
        == dp.FindScalar(dp.Aggregate(dp.PlainSymbol(clj.S('count')),
                                      [dp.Variable(clj.S('?b')),
                                       dp.Constant(1),
                                       dp.SrcVar(clj.S('$x'))]))
