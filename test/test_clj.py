import hypothesis as h
import hypothesis.strategies as st

import src.clj as clj


@h.given(seq=st.lists(st.uuids()),
         n_first=st.integers(min_value=0, max_value=100))
def test_extract_seq_return_len(seq, n_first):
    extracted = clj.extract_seq(seq, n_first)
    assert n_first + 1 == len(extracted)


def test_extract_seq_1_empty():
    val, rest = clj.extract_seq([], 1)
    assert val is None
    assert rest is None


def test_extract_seq_2_empty():
    val1, val2, rest = clj.extract_seq([], 2)
    assert val1 is None
    assert val2 is None
    assert rest is None


@h.given(seq=st.lists(st.uuids(), min_size=1))
def test_extract_seq_1(seq):
    val, rest = clj.extract_seq(seq, 1)
    assert seq[0] == val
    assert rest == (seq[1:] or None)


@h.given(seq=st.lists(st.uuids(), min_size=2))
def test_extract_seq_2(seq):
    val1, val2, rest = clj.extract_seq(seq, 2)
    assert seq[0] == val1
    assert seq[1] == val2
    assert rest == (seq[2:] or None)
