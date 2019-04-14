import hypothesis as h
import hypothesis.strategies as st

import src.clj as clj


class TestExtractSeq:
    @h.given(seq=st.lists(st.uuids()),
             n_first=st.integers(min_value=0, max_value=100))
    def test_extract_seq_return_len(self, seq, n_first):
        extracted = clj.extract_seq(seq, n_first)
        assert n_first + 1 == len(extracted)

    def test_extract_seq_1_empty(self):
        val, rest = clj.extract_seq([], 1)
        assert val is None
        assert rest is None

    def test_extract_seq_2_empty(self):
        val1, val2, rest = clj.extract_seq([], 2)
        assert val1 is None
        assert val2 is None
        assert rest is None

    @h.given(seq=st.lists(st.uuids(), min_size=1))
    def test_extract_seq_1(self, seq):
        val, rest = clj.extract_seq(seq, 1)
        assert seq[0] == val
        assert rest == (seq[1:] or None)

    @h.given(seq=st.lists(st.uuids(), min_size=2))
    def test_extract_seq_2(self, seq):
        val1, val2, rest = clj.extract_seq(seq, 2)
        assert seq[0] == val1
        assert seq[1] == val2
        assert rest == (seq[2:] or None)


class TestConj:
    @h.given(seq=st.one_of(st.lists(st.integers(min_value=0)),
                           st.sets(st.integers(min_value=0)),
                           st.none()))
    def test_iterables(self, seq):
        element = -42
        result = clj.conj(seq, element)

        seq_list = list(seq or [])
        result_list = list(result)

        # assert len(list(seq)) + 1 == len(list(result))
        assert len(seq_list) + 1 == len(result_list)
        assert element in result_list
