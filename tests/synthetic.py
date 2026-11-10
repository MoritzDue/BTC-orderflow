"""Tiny synthetic order books for the tests."""
import json
import os
import tempfile

from orderflow import book

MIDNIGHT_MS = 1_764_547_200_000          # 2025-12-01 00:00 UTC


def msg(cts, u, kind, bids, asks):
    """One archive message; bids/asks are [(price string, size string)]."""
    return {'topic': book.TOPIC, 'type': kind, 'ts': cts, 'cts': cts,
            'data': {'s': 'BTCUSDT', 'b': [list(x) for x in bids], 'a': [list(x) for x in asks], 'u': u, 'seq': u}}


def lines(messages):
    return [json.dumps(m).encode() for m in messages]


def quiet_day(changes=(), until_s=120, step_ms=500):
    """Snapshot at midnight + 100 ms (bid 100.00 x 2, ask 100.10 x 3, plus deeper levels),
    then a heartbeat delta every step_ms until until_s. changes: {cts offset ms: (bids, asks)}
    replace the heartbeat at that time."""
    changes = dict(changes)
    bids = [('100.00', '2'), ('99.90', '5'), ('99.80', '10')]
    asks = [('100.10', '3'), ('100.20', '5'), ('100.30', '10')]
    out = [msg(MIDNIGHT_MS + 100, 1, 'snapshot', bids, asks)]
    u = 1
    for t in range(step_ms, until_s * 1000, step_ms):
        u += 1
        b, a = changes.get(t, ([], []))
        out.append(msg(MIDNIGHT_MS + t, u, 'delta', b, a))
    return out


def write_store(messages):
    """Replay messages into a temporary store file; returns its path."""
    fd, path = tempfile.mkstemp(suffix='.zlib')
    os.close(fd)
    b, w = book.Book(), book.StoreWriter(path)
    for m in messages:
        b.apply(m)
        w.add(m['cts'], b, m['type'] == 'snapshot')
    w.finish()
    return path
