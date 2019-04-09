from pprint import pprint as pp

import src.parser as parser
from src.clj import K, S


def main():
    # k = Keyword("dataserpent")
    # print(k)
    # print(repr(k))

    query = [K('find'), S('?e'),
             K('where'), [S('?e'), K('name')]]
    # pp(query2map(query))
    # => {Keyword('find'): [Symbol('?e')],
    #     Keyword('where'): [[Symbol('?e'), Keyword('name')]]}
    parsed_query = parser.parse_query(query)
    pp(parsed_query)


if __name__ == '__main__':
    main()
