import pytest

import src.parser as dp
import src.clj as clj


# deftest clauses

def test_clauses_valid():
    form = clj.str2edn('[[(rule ?x) [?x :name _]]]')
    result = dp.parse_rules(form)
    expected = [
        dp.Rule(dp.PlainSymbol(clj.S('rule')),
                [dp.RuleBranch(
                    dp.RuleVars(None, [dp.Variable(clj.S('?x'))]),
                    [dp.Pattern(
                        dp.DefaultSrc(None),
                        [dp.Variable(clj.S('?x')), dp.Constant(clj.K('name')), dp.Placeholder(None)])])])]
    assert result == expected


def test_clauses_unknown_var():
    form = clj.str2edn('[[(rule ?x) [?x :name ?y]]]')
    with pytest.raises(AssertionError) as excinfo:
        dp.parse_rules(form)
    assert str(excinfo.value) == "Reference to the unknown variables: [Symbol(?y)]"


# deftest rule-vars

@pytest.mark.parametrize('form_edn,expected',
                         [('[[(rule [?x] ?y) [_]]]',
                           [dp.Rule(
                               dp.PlainSymbol(clj.S('rule')),
                               [dp.RuleBranch(
                                   dp.RuleVars([dp.Variable(clj.S('?x'))], [dp.Variable(clj.S('?y'))]),
                                   [dp.Pattern(dp.DefaultSrc(None), [dp.Placeholder(None)])])])]),
                          ('[[(rule [?x ?y] ?a ?b) [_]]]',
                           [dp.Rule(
                               dp.PlainSymbol(clj.S('rule')),
                               [dp.RuleBranch(
                                   dp.RuleVars([dp.Variable(clj.S('?x')), dp.Variable(clj.S('?y'))],
                                               [dp.Variable(clj.S('?a')), dp.Variable(clj.S('?b'))]),
                                   [dp.Pattern(dp.DefaultSrc(None), [dp.Placeholder(None)])])])]),
                          ('[[(rule [?x]) [_]]]',
                           [dp.Rule(dp.PlainSymbol(clj.S('rule')),
                                    [dp.RuleBranch(
                                        dp.RuleVars([dp.Variable(clj.S('?x'))], None),
                                        [dp.Pattern(dp.DefaultSrc(None), [dp.Placeholder(None)])])])])])
def test_rule_vars_valid(form_edn, expected):
    form = clj.str2edn(form_edn)
    result = dp.parse_rules(form)
    assert result == expected
