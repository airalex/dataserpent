import src.clj as clj


def is_seqable(x):
    if isinstance(x, str):
        return False

    return (clj.is_seq(x) or
            clj.is_nil(x))
