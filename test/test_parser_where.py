import src.parser as dp
import src.clj as clj


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
