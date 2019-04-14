import pytest

import src.parser as dp
import src.clj as clj


# deftest clauses

def test_clauses():
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
