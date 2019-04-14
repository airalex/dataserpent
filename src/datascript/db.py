import src.clj as clj


def is_seqable(x):
    if isinstance(x, str):
        return False

    # is_nil() isn't included here, as Python generally doesn't allow iterating over None
    return clj.is_seq(x)
