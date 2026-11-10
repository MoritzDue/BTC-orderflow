"""Download one day of public Bybit data (top-200 order book + trades) into data/.

  python scripts/download_sample.py 2025-12-01

About 230-420 MB per day. Every file is checked against the size and SHA-256 recorded in
data/manifest.csv when the study ran; a file that does not match is deleted.
"""
import csv
import hashlib
from pathlib import Path
import sys
import urllib.request

DATA = Path(__file__).resolve().parent.parent / 'data'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def manifest(day):
    rows = [r for r in csv.DictReader(open(DATA / 'manifest.csv')) if r['date'] == day]
    if len(rows) != 2:
        raise SystemExit('day %s is not in data/manifest.csv (study days: 2025-10-01..11-29, '
                         '2025-12-01..2026-01-14)' % day)
    return rows


def local_path(row):
    return DATA / row['url'].rsplit('/', 1)[1]


def verified(row):
    p = local_path(row)
    return p.exists() and p.stat().st_size == int(row['bytes']) and sha256(p) == row['sha256']


def download(row):
    p = local_path(row)
    if verified(row):
        print('ok (already here):', p.name)
        return
    tmp = p.with_suffix(p.suffix + '.part')
    req = urllib.request.Request(row['url'], headers={'User-Agent': 'btc-orderflow-study'})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, 'wb') as f:
        done = 0
        for block in iter(lambda: r.read(1 << 20), b''):
            f.write(block)
            done += len(block)
            print('\r%s  %.0f / %.0f MB' % (p.name, done / 1e6, int(row['bytes']) / 1e6), end='', flush=True)
    print()
    tmp.rename(p)
    if not verified(row):
        p.unlink()
        raise SystemExit('size or SHA-256 differs from the manifest: %s (deleted)' % p.name)
    print('ok:', p.name)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    for row in manifest(sys.argv[1]):
        download(row)
