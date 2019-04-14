import src.datascript.db as ddb
import src.clj as clj


empty_db = ddb.empty_db


def is_filtered(x):
    return clj.is_instance(ddb.FilteredDB, x)


def with_(db, tx_data, tx_meta=None):
    assert ddb.is_db(db)
    if is_filtered(db):
        assert False, "Filtered DB cannot be modified"
    else:
        return ddb.transact_tx_data(ddb.TxReport(db_before=db,
                                                 db_after=db,
                                                 tx_data=[],
                                                 tempids={},
                                                 tx_meta=tx_meta),
                                    tx_data)


def db_with(db, tx_data):
    assert ddb.is_db(db)
    return with_(db, tx_data).db_after
