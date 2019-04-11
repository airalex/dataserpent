import pytest

import src.parser as dp
import src.clj as clj


# deftest pattern

def test_pattern_eav():
    assert dp.parse_clause(clj.str2edn('[?e ?a ?v]')) \
        == dp.Pattern(dp.DefaultSrc(None),
                      [dp.Variable(clj.S('?e')),
                       dp.Variable(clj.S('?a')),
                       dp.Variable(clj.S('?v'))])


def test_pattern_a():
    assert dp.parse_clause(clj.str2edn('[_ ?a _ _]')) \
        == dp.Pattern(dp.DefaultSrc(None),
                      [dp.Placeholder(None),
                       dp.Variable(clj.S('?a')),
                       dp.Placeholder(None),
                       dp.Placeholder(None)])


def test_pattern_x_a():
    assert dp.parse_clause(clj.str2edn('[$x _ ?a _ _]')) \
        == dp.Pattern(dp.SrcVar(clj.S('$x')),
                      [dp.Placeholder(None),
                       dp.Variable(clj.S('?a')),
                       dp.Placeholder(None),
                       dp.Placeholder(None)])


def test_pattern_x_name_v():
    assert dp.parse_clause(clj.str2edn('[$x _ :name ?v]')) \
        == dp.Pattern(dp.SrcVar(clj.S('$x')),
                      [dp.Placeholder(None),
                       dp.Constant(clj.K('name')),
                       dp.Variable(clj.S('?v'))])


# deftest test-pred

def test_pred_a_1():
    clause = clj.str2edn('[(pred ?a 1)]')
    expected = dp.Predicate(dp.PlainSymbol(clj.S('pred')),
                            [dp.Variable(clj.S('?a')), dp.Constant(1)])
    assert dp.parse_clause(clause) == expected


def test_pred():
    clause = clj.str2edn('[(pred)]')
    expected = dp.Predicate(dp.PlainSymbol(clj.S('pred')), [])
    assert dp.parse_clause(clause) == expected


def test_pred_custom():
    clause = clj.str2edn('[(?custom-pred ?a)]')
    expected = dp.Predicate(dp.PlainSymbol(clj.S('?custom-pred')),
                            [dp.Variable(clj.S('?a'))])
    assert dp.parse_clause(clause) == expected


# deftest test-fn

def test_fn_a_1_x():
    clause = clj.str2edn('[(fn ?a 1) ?x]')
    expected = dp.Function(dp.PlainSymbol(clj.S('fn')),
                           [dp.Variable(clj.S('?a')), dp.Constant(1)],
                           dp.BindScalar(dp.Variable(clj.S('?x'))))
    assert dp.parse_clause(clause) == expected


def test_fn_x():
    clause = clj.str2edn('[(fn) ?x]')
    expected = dp.Function(dp.PlainSymbol(clj.S('fn')),
                           [],
                           dp.BindScalar(dp.Variable(clj.S('?x'))))
    assert dp.parse_clause(clause) == expected


def test_fn_custom_fn_x():
    clause = clj.str2edn('[(?custom-fn) ?x]')
    expected = dp.Function(dp.Variable(clj.S('?custom-fn')),
                           [],
                           dp.BindScalar(dp.Variable(clj.S('?x'))))
    assert dp.parse_clause(clause) == expected


def test_fn_custom_fn_arg_x():
    clause = clj.str2edn('[(?custom-fn ?arg) ?x]')
    expected = dp.Function(dp.Variable(clj.S('?custom-fn')),
                           [dp.Variable(clj.S('?arg'))],
                           dp.BindScalar(dp.Variable(clj.S('?x'))))
    assert dp.parse_clause(clause) == expected


# deftest rule-expr

def test_rule_expr_friends_x_y():
    clause = clj.str2edn('(friends ?x ?y)')
    expected = dp.RuleExpr(dp.DefaultSrc(None),
                           dp.PlainSymbol(clj.S('friends')),
                           [dp.Variable(clj.S('?x')), dp.Variable(clj.S('?y'))])
    assert dp.parse_clause(clause) == expected


def test_rule_expr_friends_ivan():
    clause = clj.str2edn('(friends "Ivan" _)')
    expected = dp.RuleExpr(dp.DefaultSrc(None),
                           dp.PlainSymbol(clj.S('friends')),
                           [dp.Constant("Ivan"), dp.Placeholder(None)])
    assert dp.parse_clause(clause) == expected


def test_rule_expr_1_friends_x_y():
    clause = clj.str2edn('($1 friends ?x ?y)')
    expected = dp.RuleExpr(dp.SrcVar(clj.S('$1')),
                           dp.PlainSymbol(clj.S('friends')),
                           [dp.Variable(clj.S('?x')), dp.Variable(clj.S('?y'))])
    assert dp.parse_clause(clause) == expected


def test_rule_expr_friends():
    clause = clj.str2edn('(friends)')
    with pytest.raises(AssertionError) as excinfo:
        dp.parse_clause(clause)
    assert "rule-expr requires at least one argument" in str(excinfo.value)


def test_rule_expr_friends_something():
    clause = clj.str2edn('(friends something)')
    with pytest.raises(AssertionError) as excinfo:
        dp.parse_clause(clause)
    assert "Cannot parse rule-expr arguments" in str(excinfo.value)
