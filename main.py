import toolz.itertoolz as tzi
import toolz.dicttoolz as tzd


class Keyword:
    def __init__(self, name):
        self._name = name

    def __str__(self):
        return ':' + self._name

    def __repr__(self):
        return "Keyword('{}')".format(self._name)


class Symbol:
    def __init__(self, name):
        self._name = name

    def __str__(self):
        return "'{}".format(self._name)

    def __repr__(self):
        return "Symbol('{}')".format(self._name)


K = Keyword
S = Symbol


def query2map(query: list) -> dict:
    def _loop(parsed: dict, key, qs: list):
        q = tzi.get(0, qs, None)
        if q is not None:
            if isinstance(q, Keyword):
                return _loop(parsed,
                             q,
                             list(tzi.drop(1, qs)))
            else:
                return _loop(tzd.update_in(parsed, [key], lambda prev: prev + q if prev else [q]),
                             key,
                             list(tzi.drop(1, qs)))
        else:
            return parsed

    return _loop({}, None, query)


def parse_query(query: list):
    pass


def main():
    # k = Keyword("dataserpent")
    # print(k)
    # print(repr(k))

    query = [K('find'), S('?e'),
             K('where'), [S('?e'), K('name')]]
    print(query2map(query))


if __name__:
    main()
