import pytest

import src.parser as dp
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
