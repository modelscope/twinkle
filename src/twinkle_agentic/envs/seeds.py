# Copyright (c) ModelScope Contributors. All rights reserved.
"""Workspace fixture programs and dependency canaries used by sandbox profiles.

Replace only the literal {workspace!r} token when executing a program.
"""

__all__ = ['CLEAR_WORKSPACE', 'LIBRARY_CANARIES', 'SEED_PROGRAMS']

# Prove a library a profile depends on is actually present in the image. Separate
# from the seeds: a seed BUILDS state, a canary only asserts something exists, and
# the two fail for different reasons -- a canary failing means the image is wrong,
# a seed failing usually means the seed is.
LIBRARY_CANARIES = {
    'python-numpy-sympy-canary': 'import numpy\nimport sympy\nprint("profile canary ok")',
    'python-test-runner-canary': 'import unittest\nprint("profile canary ok")',
}

CLEAR_WORKSPACE = '''
import os, shutil
root = {workspace!r}
os.makedirs(root, exist_ok=True)
for name in os.listdir(root):
    path = os.path.join(root, name)
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path, ignore_errors=True)
    else:
        os.remove(path)
leftover = os.listdir(root)
assert not leftover, 'workspace not empty after clear: %r' % (leftover,)
'''

# The workspace path is substituted by sandbox.py, not by this module: the seed
# only knows the directory it is given.
SEED_SQL_ANALYTICS = r'''
import os, sqlite3
root = {workspace!r}
os.makedirs(root, exist_ok=True)
path = os.path.join(root, 'shop.db')
if os.path.exists(path):
    os.remove(path)
conn = sqlite3.connect(path)
conn.executescript("""
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    city        TEXT NOT NULL,
    region      TEXT,
    signup_date TEXT NOT NULL
);
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    category   TEXT NOT NULL,
    unit_price REAL NOT NULL
);
CREATE TABLE orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    product_id  INTEGER NOT NULL REFERENCES products(product_id),
    quantity    INTEGER NOT NULL,
    order_date  TEXT NOT NULL,
    status      TEXT NOT NULL
);
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_orders_customer ON orders(customer_id);
""")
conn.executemany('INSERT INTO customers VALUES (?,?,?,?,?)', [
    (1,'Ada Lovelace','London','EU','2019-03-02'),
    (2,'Grace Hopper','Arlington','NA','2019-07-14'),
    (3,'Edsger Dijkstra','Rotterdam','EU','2020-01-05'),
    (4,'Barbara Liskov','Boston','NA','2020-06-30'),
    (5,'Alan Turing','London','EU','2021-02-11'),
    (6,'Radia Perlman','Seattle','NA','2021-09-23'),
    (7,'Leslie Lamport','New York','NA','2022-04-17'),
    (8,'Karen Sparck Jones','Cambridge',None,'2022-11-08'),
])
conn.executemany('INSERT INTO products VALUES (?,?,?,?)', [
    (10,'Blue Widget','widgets',12.50),
    (11,'Red Widget','widgets',12.50),
    (12,'Green Gadget','gadgets',89.99),
    (13,'Blue Gadget','gadgets',94.00),
    (14,'Service Plan','services',200.00),
])
conn.executemany('INSERT INTO orders VALUES (?,?,?,?,?,?)', [
    (1001,1,10,3,'2023-01-15','shipped'),
    (1002,1,12,1,'2023-01-22','shipped'),
    (1003,2,10,10,'2023-02-03','shipped'),
    (1004,3,11,2,'2023-02-14','cancelled'),
    (1005,3,12,1,'2023-03-01','shipped'),
    (1006,4,14,2,'2023-03-12','shipped'),
    (1007,5,13,1,'2023-04-02','pending'),
    (1008,6,10,7,'2023-04-19','shipped'),
    (1009,7,12,3,'2023-05-05','shipped'),
    (1010,8,11,1,'2023-05-21','shipped'),
    (1011,1,10,3,'2023-01-15','shipped'),
])
conn.commit()
tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
assert tables == {'customers','products','orders'}, tables
assert conn.execute('SELECT COUNT(*) FROM orders').fetchone()[0] == 11
conn.close()
print('seed-sql-analytics ok')
'''

SEED_CSV_DIRTY = r'''
import csv, io, os
root = {workspace!r}
os.makedirs(root, exist_ok=True)
rows = [
    ['reading_id','station','timestamp','temperature_c','humidity_pct'],
    ['1','Alpha','2023-01-01T00:00:00Z','3.5','81'],
    ['2','Alpha','2023-01-01T01:00:00Z','3.1','83'],
    ['3','Bravo','2023-01-01T00:00:00Z','-2.0','90'],
    ['4','Bravo, North','2023-01-01T01:00:00Z','-2.4',''],
    ['5','Charlie','2023-01-01T00:00:00Z','11.25','64'],
    ['5','Charlie','2023-01-01T00:00:00Z','11.25','64'],
    ['6','alpha','2023-01-01T02:00:00Z','2.9','85'],
]
buf = io.StringIO()
writer = csv.writer(buf, lineterminator='\n')
for row in rows:
    writer.writerow(row)
# utf-8-sig writes a BOM, which a plain utf-8 read leaves on the first header
# cell ('\ufeffreading_id'). That is the defect most real exports arrive with.
with open(os.path.join(root, 'readings.csv'), 'w', encoding='utf-8-sig', newline='') as fh:
    fh.write(buf.getvalue())
with open(os.path.join(root, 'stations.txt'), 'w', encoding='utf-8', newline='') as fh:
    fh.write('Alpha\nBravo, North\nCharlie\n')
with open(os.path.join(root, 'readings.csv'), 'rb') as fh:
    head = fh.read(3)
assert head == b'\xef\xbb\xbf', head
print('seed-csv-dirty ok')
'''

SEED_DOCS_CORPUS = r'''
import json, os
root = {workspace!r}
docs = os.path.join(root, 'corpus')
os.makedirs(docs, exist_ok=True)
files = {
  'handbook.md': (
    '# Operations Handbook\n\n'
    '## Retry policy\n\n'
    'The ingest worker retries a failed batch three times. The delay before the\n'
    'first retry is 5 seconds, and each subsequent delay doubles it.\n\n'
    '## Escalation\n\n'
    'After the third failure the batch is parked and an alert is raised.\n'),
  'release-notes.md': (
    '# Release notes\n\n'
    '## 4.2.0\n\n'
    'The ingest worker retry limit was raised from two to three. This is the only\n'
    'change to retry behaviour in this release.\n'),
  'decoy-faq.md': (
    '# FAQ\n\n'
    '## How many times does the ingest worker retry?\n\n'
    'Historically this was four, and older runbooks still say so. Check the\n'
    'release notes for the current value.\n'),
}
for name in sorted(files):
    with open(os.path.join(docs, name), 'w', encoding='utf-8', newline='') as fh:
        fh.write(files[name])
index = {
    'question': ('How many times does the ingest worker retry a failed batch, and '
                 'what is the delay before the first retry?'),
    'answer_document': 'handbook.md',
    'superseded_document': 'decoy-faq.md',
    'note': 'decoy-faq.md states a superseded value; a correct answer cites handbook.md',
}
with open(os.path.join(root, 'question.json'), 'w', encoding='utf-8', newline='') as fh:
    json.dump(index, fh, indent=2, sort_keys=True)
    fh.write('\n')
assert sorted(os.listdir(docs)) == ['decoy-faq.md', 'handbook.md', 'release-notes.md']
print('seed-docs-corpus ok')
'''

SEED_BROKEN_REPO = r'''
import os, textwrap
root = {workspace!r}
src = os.path.join(root, 'src')
tests = os.path.join(root, 'tests')
os.makedirs(src, exist_ok=True)
os.makedirs(tests, exist_ok=True)
for pkg in (src, tests):
    open(os.path.join(pkg, '__init__.py'), 'w').close()
module = textwrap.dedent("""
    def merge_ranges(ranges):
        # Merge overlapping [start, end) intervals, sorted ascending.
        # Touching intervals (end == next start) merge.
        if not ranges:
            return []
        ordered = sorted(ranges)
        out = [list(ordered[0])]
        for start, end in ordered[1:]:
            if start < out[-1][1]:
                out[-1][1] = max(out[-1][1], end)
            else:
                out.append([start, end])
        return [tuple(pair) for pair in out]
""").lstrip()
with open(os.path.join(src, 'ranges.py'), 'w', encoding='utf-8', newline='') as fh:
    fh.write(module)
test = textwrap.dedent("""
    import unittest
    from src.ranges import merge_ranges

    class TestMergeRanges(unittest.TestCase):
        def test_empty(self):
            self.assertEqual(merge_ranges([]), [])
        def test_single(self):
            self.assertEqual(merge_ranges([(1, 5)]), [(1, 5)])
        def test_touching_merges(self):
            self.assertEqual(merge_ranges([(1, 5), (5, 9)]), [(1, 9)])
        def test_disjoint_kept(self):
            self.assertEqual(merge_ranges([(1, 5), (7, 9)]), [(1, 5), (7, 9)])
        def test_unsorted_input(self):
            self.assertEqual(merge_ranges([(10, 12), (1, 3)]), [(1, 3), (10, 12)])
        def test_contained(self):
            self.assertEqual(merge_ranges([(1, 10), (2, 3)]), [(1, 10)])

    if __name__ == '__main__':
        unittest.main()
""").lstrip()
with open(os.path.join(tests, 'test_ranges.py'), 'w', encoding='utf-8', newline='') as fh:
    fh.write(test)
print('seed-broken-repo ok')
'''

SEED_LOG_CORPUS = r'''
import os
root = {workspace!r}
os.makedirs(root, exist_ok=True)
app = [
  '2023-06-01T10:00:00Z INFO  req=7f2a started route=/ingest',
  '2023-06-01T10:00:01Z INFO  req=7f2a wrote 120 rows',
  '2023-06-01T10:00:02Z ERROR req=7f2a failed stage=commit error="disk quota exceeded"',
  '2023-06-01T10:00:03Z INFO  req=91bc started route=/ingest',
  '2023-06-01T10:00:09Z INFO  req=91bc wrote 90 rows',
  '2023-06-01T10:00:10Z INFO  req=91bc finished ok',
]
proxy = [
  '2023-06-01T10:00:00Z 200 POST /ingest req=7f2a 1204ms',
  '2023-06-01T10:00:03Z 200 POST /ingest req=91bc 890ms',
  '2023-06-01T10:00:04Z 500 POST /ingest req=4cd1',
]
with open(os.path.join(root, 'app.log'), 'w', encoding='utf-8', newline='') as fh:
    fh.write('\n'.join(app) + '\n')
with open(os.path.join(root, 'proxy.log'), 'w', encoding='utf-8', newline='') as fh:
    # The final line is truncated mid-timestamp, the way a log looks when the
    # writer died. A parser that assumes well-formed lines will raise on it.
    fh.write('\n'.join(proxy) + '\n2023-06-01T10:00:')
print('seed-log-corpus ok')
'''

SEED_PROGRAMS = {
    'seed-sql-analytics': SEED_SQL_ANALYTICS,
    'seed-csv-dirty': SEED_CSV_DIRTY,
    'seed-docs-corpus': SEED_DOCS_CORPUS,
    'seed-broken-repo': SEED_BROKEN_REPO,
    'seed-log-corpus': SEED_LOG_CORPUS,
}
