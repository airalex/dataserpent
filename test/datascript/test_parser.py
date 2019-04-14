import pytest

import src.datascript.parser as dp
import src.clj as clj


# deftest bindings

def test_bindings_x():
    assert dp.parse_binding(clj.str2edn('?x')) \
        == dp.BindScalar(dp.Variable(clj.S('?x')))


def test_bindings__():
    assert dp.parse_binding(clj.str2edn('_')) \
        == dp.BindIgnore(None)


def test_bindings_x_ellipsis():
    assert dp.parse_binding(clj.str2edn('[?x ...]')) \
        == dp.BindColl(dp.BindScalar(dp.Variable(clj.S('?x'))))


def test_bindings_vec_x():
    assert dp.parse_binding(clj.str2edn('[?x]')) \
        == dp.BindTuple([dp.BindScalar(dp.Variable(clj.S('?x')))])


def test_bindings_vec_x_y():
    assert dp.parse_binding(clj.str2edn('[?x ?y]')) \
        == dp.BindTuple([dp.BindScalar(dp.Variable(clj.S('?x'))),
                         dp.BindScalar(dp.Variable(clj.S('?y')))])


def test_bindings_vec___y():
    assert dp.parse_binding(clj.str2edn('[_ ?y]')) \
        == dp.BindTuple([dp.BindIgnore(None),
                         dp.BindScalar(dp.Variable(clj.S('?y')))])


def test_bindings_vec___x_ellipsis():
    assert dp.parse_binding(clj.str2edn('[[_ [?x ...]] ...]')) \
        == dp.BindColl(
            dp.BindTuple([dp.BindIgnore(None),
                          dp.BindColl(dp.BindScalar(dp.Variable(clj.S('?x'))))]))


def test_bindings_vec_vec_a_b_c():
    assert dp.parse_binding(clj.str2edn('[[?a ?b ?c]]')) \
        == dp.BindColl(
            dp.BindTuple([dp.BindScalar(dp.Variable(clj.S('?a'))),
                          dp.BindScalar(dp.Variable(clj.S('?b'))),
                          dp.BindScalar(dp.Variable(clj.S('?c')))]))


# deftest in

def test_in_x():
    assert dp.parse_in(clj.str2edn('[?x]')) \
        == [dp.BindScalar(dp.Variable(clj.S('?x')))]


def test_in_scalars():
    assert dp.parse_in(clj.str2edn('[$ $1 % _ ?x]')) \
        == [dp.BindScalar(dp.SrcVar(clj.S('$'))),
            dp.BindScalar(dp.SrcVar(clj.S('$1'))),
            dp.BindScalar(dp.RulesVar(None)),
            dp.BindIgnore(None),
            dp.BindScalar(dp.Variable(clj.S('?x')))]


def test_in_coll_scalars():
    assert dp.parse_in(clj.str2edn('[$ [[_ [?x ...]] ...]]')) \
        == [dp.BindScalar(dp.SrcVar(clj.S('$'))),
            dp.BindColl(dp.BindTuple([dp.BindIgnore(None),
                                      dp.BindColl(
                                          dp.BindScalar(dp.Variable(clj.S('?x'))))]))]


def test_in_x_key():
    clause = clj.str2edn('[?x :key]')
    with pytest.raises(AssertionError) as excinfo:
        dp.parse_in(clause)
    assert str(excinfo.value) == "Cannot parse binding, expected (bind-scalar | bind-tuple | bind-coll | bind-rel)"


# deftest with


def test_with_x_y():
    assert dp.parse_with(clj.str2edn('[?x ?y]')) \
        == [dp.Variable(clj.S('?x')),
            dp.Variable(clj.S('?y'))]


def test_with_x__():
    clause = clj.str2edn('[?x _]')
    with pytest.raises(AssertionError) as excinfo:
        dp.parse_with(clause)
    assert str(excinfo.value) == "Cannot parse :with clause, expected [ variable+ ]"
