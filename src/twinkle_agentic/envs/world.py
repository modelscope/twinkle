# Copyright (c) ModelScope Contributors. All rights reserved.
"""Host-served read-only SQLite, document and search tools.

TWINKLE_WORLD_ROOT names a bundle containing manifest.json, ops.db and docs/.
TWINKLE_WORLD_SEARCH_ROOT optionally names a hashed corpus.json bundle.
TWINKLE_WORLD_SEARCH_BACKEND selects corpus (default) or a live backend.
For live search, TWINKLE_WORLD_SEARCH_MODULE contains search_backend.py with
Backend.web_search(query, num_results, freshness) and fetch_page(url, query),
both returning observation strings.

Database tools require db_enabled and TWINKLE_WORLD_DB_MODULE containing
provision(host, port, ttl=...) -> credentials and drop(role). Host, port and
lease use TWINKLE_WORLD_DB_HOST / _PORT / _LEASE, defaulting to
127.0.0.1 / 5432 / 21600. The deployment supplies permissions and credentials.

WorldToolsMixin dispatches these tools through the existing Env interface.
Search receipts record served URLs; they do not establish source correctness.
"""
from __future__ import annotations

import copy
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Sequence

from twinkle.data_format.message import Tool as ToolInfo
from .base import StepResult

MAX_ROWS = 5000

# Deliberately the same crude overlap the corpus builder uses. It is a stand-in
# for a ranker, not a ranker: a good one would hide exactly the failures the
# corpus exists to reproduce.
_WORD = re.compile(r'[a-z0-9]+')
_STOP = {
    'the', 'a', 'an', 'of', 'in', 'is', 'are', 'and', 'or', 'to', 'for', 'on', 'by', 'with', 'as', 'at', 'it', 'its',
    'that', 'this', 'be', 'from'
}


def _terms(text: str) -> set:
    return {word for word in _WORD.findall((text or '').lower()) if word not in _STOP and len(word) > 2}


# The schemas are the contract the model is shown. Written here rather than read
# from the world service because the host side must be able to advertise a tool
# before the service is up -- and a tool that disappears from the prompt when the
# service is down would make an outage look like a different task.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        'type': 'function',
        'function': {
            'name':
            'world_sql',
            'description': ('Run a read-only SQL query against the shared operations database. '
                            'Tables: regions, sites, machines, incidents, procedures. The '
                            'database is shared and read-only; writes are rejected. Use this to '
                            'obtain the real values instead of assuming them.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'A single SELECT statement.'
                    },
                    'params': {
                        'type': 'array',
                        'description': 'Optional bound parameters for ? placeholders.',
                        'items': {}
                    },
                },
                'required': ['query'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name':
            'world_schema',
            'description': ('Describe the shared database: call with no name to list the tables, '
                            'or with a table name to get its columns and types. Cheaper and more '
                            'reliable than guessing column names.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'name': {
                        'type': 'string',
                        'description': 'Optional table name.'
                    }
                },
                'required': [],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name':
            'world_doc',
            'description': ('Read one document from the shared procedures corpus. Call with no '
                            'name to list what is available. The documents state the rules the '
                            'database alone does not, such as how an incident should have been '
                            'classified.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'name': {
                        'type': 'string',
                        'description': 'Document name, e.g. disk.md.'
                    }
                },
                'required': [],
            },
        },
    },
]

SEARCH_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        'type': 'function',
        'function': {
            'name':
            'web_search',
            'description': ('Search the shared document index. Returns ranked results with a url, '
                            'title, publication date and short quoted excerpts. The ranking is NOT '
                            'reliable: a result can be on-topic and out of date, and the top hit is '
                            'sometimes superseded. Fetch the url before trusting its content.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'Search query.'
                    },
                    'num_results': {
                        'type': 'integer',
                        'description': 'How many results to return (1-25).'
                    },
                    'freshness': {
                        'type': 'string',
                        'description': "Optional 'YYYY-MM-DD'; results on or after this date first."
                    },
                },
                'required': ['query'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name':
            'fetch_page',
            'description': ('Fetch the full text of a url returned by web_search. Use this on every '
                            'result whose content matters: search excerpts are shortened and are not '
                            'the document. A long page is returned in part, not whole; pass `query` '
                            'to get the part that bears on what you are looking for.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'url': {
                        'type': 'string',
                        'description': 'A url from web_search.'
                    },
                    # Optional and not required: a policy that does not know what
                    # it is looking for should still be able to read the page.
                    'query': {
                        'type':
                        'string',
                        'description':
                        'Optional. What you are looking for on '
                        'the page; the returned text is the part '
                        'that mentions it.'
                    },
                },
                'required': ['url'],
            },
        },
    },
]

DB_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        'type': 'function',
        'function': {
            'name':
            'db_query',
            'description': ('Run a read-only SQL query against the shared operations database '
                            '(PostgreSQL). Your account can SELECT and create temporary tables; '
                            'it cannot change anything. Tables: regions, sites, machines, '
                            'incidents, procedures, incident_notes.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'One SQL statement (SELECT).'
                    },
                    'params': {
                        'type': 'array',
                        'description': 'Optional bound parameters for %s placeholders.',
                        'items': {}
                    },
                },
                'required': ['query'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name':
            'db_schema',
            'description': ('List the tables in the shared database, or describe one table. '
                            'Cheaper and more reliable than guessing column names.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'table': {
                        'type': 'string',
                        'description': 'Optional table name.'
                    }
                },
                'required': [],
            },
        },
    },
]

TOOL_NAMES = tuple(schema['function']['name'] for schema in TOOL_SCHEMAS + SEARCH_TOOL_SCHEMAS + DB_TOOL_SCHEMAS)


def _world_name(tool_name: str | None) -> str | None:
    """Resolve a plain tool name or a harness-qualified server---tool name."""
    if not tool_name:
        return None
    short = tool_name.rsplit('---', 1)[-1]
    return short if short in TOOL_NAMES else None


class WorldUnavailable(RuntimeError):
    """The world is not configured for this run."""


class HostWorld:
    """Read a world bundle on the host; mutable receipts and credentials belong to this instance."""

    def __init__(self, root: str | None = None):
        # A library does not guess where its data lives. Without an explicit root
        # the world reports itself unavailable rather than reading whatever
        # happens to sit at a path baked in at import time.
        configured = root or os.environ.get('TWINKLE_WORLD_ROOT', '')
        self.root: Path | None = Path(configured) if configured else None
        self.manifest: dict[str, Any] = {}
        self.db_path = self.root / 'ops.db' if self.root is not None else None
        self.docs: dict[str, Path] = {}
        # The search corpus is optional: a run that only uses world_* tools should
        # not fail because the search index has not been built.
        self.search_root: Path | None = None
        self.search_documents: list[dict[str, Any]] = []
        # The episode's own database account, created on first use and dropped on
        # close. `db_enabled` is set from the profile's tool list, so a run that
        # never names db_* never touches the service at all.
        self.db_enabled = False
        self._db_creds: dict[str, Any] | None = None
        # Which engine answers web_search/fetch_page. The corpus is the default
        # and stays the default: it is deterministic, so a run built on it
        # replays, and most task development happens against it. A live engine
        # is opted into per run because its answers change by the hour.
        self.backend = 'corpus'
        self.live = None
        self.receipts: list[dict[str, Any]] = []

    def fork(self, *, db_enabled: bool = False) -> HostWorld:
        """Share loaded data while keeping receipts and database roles independent."""
        world = copy.copy(self)
        world.receipts = []
        world._db_creds = None
        world.db_enabled = db_enabled
        return world

    def _load_search(self) -> None:
        # Only the explicit env var is honoured. Walking up from __file__ to find
        # a corpus was tried and dropped: how many parents to count depends on
        # where a run checked the tree out, and getting it wrong silently reads
        # somebody else's index.
        self.backend = (os.environ.get('TWINKLE_WORLD_SEARCH_BACKEND') or 'corpus').strip().lower()
        if self.backend != 'corpus':
            self._load_live_search()
            return
        # No corpus root means no search tools, which is a supported
        # configuration: a run that only reads the database should not have to
        # build a search index it will never query.
        search_root = os.environ.get('TWINKLE_WORLD_SEARCH_ROOT', '')
        if not search_root:
            return
        root = Path(search_root)
        corpus = root / 'corpus.json'
        manifest = root / 'manifest.json'
        if not corpus.is_file() or not manifest.is_file():
            return
        expected = json.loads(manifest.read_text(encoding='utf-8'))['files']['corpus.json']
        import hashlib
        actual = hashlib.sha256(corpus.read_bytes()).hexdigest()
        if actual != expected:
            raise WorldUnavailable(f'search corpus does not match its manifest: expected {expected}, '
                                   f'got {actual}. Rebuild the corpus and its manifest together -- '
                                   'a drifted corpus would silently change what every search task '
                                   'retrieves, which is the same run producing a different score.')
        self.search_documents = json.loads(corpus.read_text(encoding='utf-8'))['documents']
        self.search_root = root

    def _load_live_search(self) -> None:
        """Attach a live engine.

        Imported lazily so a corpus-only run does not depend on the plug-in
        existing, and so a broken live backend cannot take down a run that never
        asked for it. The contract is two methods returning observation strings;
        see the module docstring.
        """
        module_dir = os.environ.get('TWINKLE_WORLD_SEARCH_MODULE', '')
        if not module_dir:
            raise WorldUnavailable('TWINKLE_WORLD_SEARCH_BACKEND is set to '
                                   f'{self.backend!r} but TWINKLE_WORLD_SEARCH_MODULE is not: a live '
                                   'engine is a deployment choice, so the directory holding its '
                                   'backend module has to be named rather than assumed.')
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        import search_backend
        self.live = search_backend.Backend()

    @property
    def available(self) -> bool:
        if self.root is None or self.db_path is None:
            return False
        return self.db_path.is_file() and bool(self.docs or (self.root / 'docs').is_dir())

    def schemas(self) -> list[ToolInfo]:
        """Advertise only tools whose backing data or database account support is enabled."""
        out: list[ToolInfo] = []
        if self.available:
            out.extend(TOOL_SCHEMAS)
        if self.search_documents or self.live is not None:
            out.extend(SEARCH_TOOL_SCHEMAS)
        if self.db_enabled:
            out.extend(DB_TOOL_SCHEMAS)
        return out

    def load(self) -> None:
        """Validate manifest hashes and load documents and optional search at startup."""
        if self.root is None:
            raise WorldUnavailable('no world root configured: pass root=... to HostWorld or set '
                                   'TWINKLE_WORLD_ROOT to the directory holding manifest.json, ops.db '
                                   'and docs/')
        manifest_path = self.root / 'manifest.json'
        if not manifest_path.is_file():
            raise WorldUnavailable(f'no world manifest at {manifest_path}')
        import hashlib
        self.manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for name, expected in sorted(self.manifest.get('files', {}).items()):
            path = self.root / name
            if not path.is_file():
                raise WorldUnavailable(f'world file missing: {path}')
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise WorldUnavailable(f'{name} does not match the manifest\n  expected {expected}\n'
                                       f'  actual   {actual}\nRebuild with build_world.py. Serving a world '
                                       'that drifted would silently change every task built on it.')
        docs_dir = self.root / 'docs'
        if docs_dir.is_dir():
            self.docs = {path.name: path for path in sorted(docs_dir.glob('*.md'))}
        self._load_search()

    def _connect(self) -> sqlite3.Connection:
        # mode=ro is the enforcement, exactly as in world_service.py: a property of
        # the handle rather than a filter over SQL text, so there is no keyword list
        # to get clever around.
        return sqlite3.connect(f'file:{self.db_path}?mode=ro', uri=True)

    def sql(self, query: str, params: Any) -> str:
        if not isinstance(query, str) or not query.strip():
            return 'world_sql: `query` must be a non-empty string.'
        if not isinstance(params, (list, tuple)):
            return 'world_sql: `params` must be a list.'
        head = query.strip().lstrip('(').split(None, 1)[0].upper()
        if head == 'PRAGMA':
            return 'world_sql: PRAGMA is not available on the shared world.'
        try:
            conn = self._connect()
        except sqlite3.Error as error:
            return f'world_sql error: database unavailable: {error}'
        try:
            cursor = conn.execute(query, list(params))
            if cursor.description is None:
                return 'world_sql: statement produced no result set.'
            columns = [c[0] for c in cursor.description]
            rows = cursor.fetchmany(MAX_ROWS + 1)
            truncated = len(rows) > MAX_ROWS
            rows = rows[:MAX_ROWS]
        except sqlite3.Error as error:
            return f'world_sql error: {type(error).__name__}: {error}'
        finally:
            conn.close()
        lines = [' | '.join(str(c) for c in columns), '-+-'.join('-' * len(str(c)) for c in columns)]
        for row in rows:
            lines.append(' | '.join('NULL' if v is None else str(v) for v in row))
        if truncated:
            lines.append(f'... truncated at {MAX_ROWS} rows')
        if not rows:
            lines.append('(no rows)')
        return '\n'.join(lines)

    def schema(self, name: str) -> str:
        name = (name or '').strip()
        try:
            conn = self._connect()
        except sqlite3.Error as error:
            return f'world_schema error: database unavailable: {error}'
        try:
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            if name and name not in tables:
                return f"world_schema error: unknown table {name!r}. Tables: {', '.join(tables)}"
            if not name:
                return 'Tables: ' + ', '.join(tables)
            columns = [(r[1], r[2], bool(r[3]), bool(r[5])) for r in conn.execute(f'PRAGMA table_info({name})')]
        except sqlite3.Error as error:
            return f'world_schema error: {type(error).__name__}: {error}'
        finally:
            conn.close()
        lines = [f'table {name}:']
        for column, kind, notnull, pk in columns:
            flags = []
            if pk:
                flags.append('primary key')
            if notnull:
                flags.append('not null')
            lines.append(f'  {column} {kind}' + (f" ({', '.join(flags)})" if flags else ''))
        return '\n'.join(lines)

    def doc(self, name: str) -> str:
        name = (name or '').strip()
        if not name:
            return ('Available documents: ' + ', '.join(sorted(self.docs)) + '. Call again with one name.')
        path = self.docs.get(name)
        if path is None:
            return (f'world_doc error: unknown document {name!r}. '
                    f'Available: {", ".join(sorted(self.docs))}')
        text = path.read_text(encoding='utf-8')
        return text if len(text) <= 20000 else text[:20000] + '\n... truncated'

    def _rank(self, query: str) -> list[tuple[float, dict[str, Any]]]:
        want = _terms(query)
        if not want:
            return []
        scored = []
        for doc in self.search_documents:
            have = _terms(doc['title'] + ' ' + doc['body'])
            score = len(want & have) + 2.0 * len(want & _terms(doc['title']))
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda pair: -pair[0])
        return scored

    def web_search(self, query: str, num_results: Any = 5, freshness: str = '') -> str:
        """Ranked results, rendered the way a tool observation is.

        The output is JSON rather than prose so the fields a policy has to reason
        about -- `published`, `url`, `rank` -- stay separable. A paragraph would
        bury the date, and the date is what decides between two documents that
        contradict each other.
        """
        if not isinstance(query, str) or not query.strip():
            return 'web_search: `query` must be a non-empty string.'
        if self.live is not None:
            return self.live.web_search(query, num_results, freshness)
        if not self.search_documents:
            return 'web_search: no index is configured for this run.'
        try:
            limit = int(num_results or 5)
        except (TypeError, ValueError):
            limit = 5
        limit = max(1, min(limit, 25))
        ranked = self._rank(query)
        if freshness:
            # A freshness filter without a stable sort is a different answer every
            # call, so rank is the tiebreak and order is total.
            ranked.sort(key=lambda pair: (pair[1].get('published') or '', pair[0]), reverse=True)
        results = []
        for rank, (_score, doc) in enumerate(ranked[:limit], 1):
            quoted = []
            want = _terms(query)
            for sentence in re.split(r'(?<=\.)\s+', doc['body'].replace('\n', ' ')):
                sentence = sentence.strip(' #')
                if sentence and want & _terms(sentence):
                    quoted.append(sentence)
                if len(quoted) >= 2:
                    break
            results.append({
                'rank': rank,
                'url': doc['url'],
                'title': doc['title'],
                'published': doc.get('published'),
                'excerpts': quoted,
            })
        if not results:
            # Saying so is the answer. A policy trained to expect results will
            # invent one here, and that is exactly the failure to catch.
            return ('web_search: no results for that query. Try different terms, or '
                    'conclude the information is not in the index.')
        return json.dumps({
            'query': query,
            'num_results': len(results),
            'results': results
        },
                          ensure_ascii=False,
                          indent=2)

    def fetch_page(self, url: str, query: str = '') -> str:
        if not isinstance(url, str) or not url.strip():
            return 'fetch_page: `url` must be a string.'
        if self.live is not None:
            return self.live.fetch_page(url, query if isinstance(query, str) else '')
        for doc in self.search_documents:
            if doc['url'] == url:
                return (f"# {doc['title']}\n"
                        f"published: {doc.get('published')}\n"
                        f"url: {doc['url']}\n\n{doc['body']}")
        return (f'fetch_page: no document at {url!r}. Call web_search again to get a '
                'url that exists.')

    # ---- the shared PostgreSQL service --------------------------------
    def _db_credentials(self) -> dict[str, Any] | None:
        """Lazily provision this instance's database role; release() drops it."""
        if self._db_creds is not None:
            return self._db_creds
        if not self.db_enabled:
            return None
        provision_dir = os.environ.get('TWINKLE_WORLD_DB_MODULE', '')
        if not provision_dir:
            raise WorldUnavailable('db_enabled is set but TWINKLE_WORLD_DB_MODULE is not: the '
                                   'per-episode role provisioning is deployment-specific, so the '
                                   'directory holding its `provision` module has to be named.')
        if provision_dir not in sys.path:
            sys.path.insert(0, provision_dir)
        import provision
        self._db_creds = provision.provision(
            os.environ.get('TWINKLE_WORLD_DB_HOST', '127.0.0.1'),
            int(os.environ.get('TWINKLE_WORLD_DB_PORT', '5432')),
            ttl=int(os.environ.get('TWINKLE_WORLD_DB_LEASE', '21600')))
        return self._db_creds

    def _db_connect(self):
        creds = self._db_credentials()
        if creds is None:
            raise RuntimeError('no database is configured for this run')
        import psycopg2
        conn = psycopg2.connect(
            host=creds['host'],
            port=creds['port'],
            dbname=creds['database'],
            user=creds['role'],
            password=creds['password'],
            connect_timeout=10)
        conn.autocommit = True
        return conn

    def db_query(self, query: str, params: Any = None) -> str:
        if not isinstance(query, str) or not query.strip():
            return 'db_query: `query` must be a non-empty string.'
        if params is not None and not isinstance(params, (list, tuple)):
            return 'db_query: `params` must be a list.'
        try:
            conn = self._db_connect()
        except Exception as error:  # noqa: BLE001
            return f'db_query: cannot reach the shared database: {error}'
        try:
            with conn.cursor() as cur:
                cur.execute(query, list(params) if params else None)
                if cur.description is None:
                    # A statement that returns nothing is allowed to succeed (a
                    # temp table, a SET) but the model is told, so it does not read
                    # silence as a result.
                    return 'db_query: statement completed with no result set.'
                columns = [d[0] for d in cur.description]
                rows = cur.fetchmany(MAX_ROWS + 1)
                truncated = len(rows) > MAX_ROWS
                rows = rows[:MAX_ROWS]
        except Exception as error:  # noqa: BLE001
            return f'db_query error: {type(error).__name__}: {str(error).strip()}'
        finally:
            conn.close()
        lines = [' | '.join(str(c) for c in columns), '-+-'.join('-' * len(str(c)) for c in columns)]
        for row in rows:
            lines.append(' | '.join('NULL' if v is None else str(v) for v in row))
        if truncated:
            lines.append(f'... truncated at {MAX_ROWS} rows')
        if not rows:
            lines.append('(no rows)')
        return '\n'.join(lines)

    def db_schema(self, table: str = '') -> str:
        table = (table or '').strip()
        try:
            conn = self._db_connect()
        except Exception as error:  # noqa: BLE001
            return f'db_schema: cannot reach the shared database: {error}'
        try:
            with conn.cursor() as cur:
                cur.execute('SELECT table_name FROM information_schema.tables '
                            "WHERE table_schema = 'public' ORDER BY table_name")
                tables = [r[0] for r in cur.fetchall()]
                if table and table not in tables:
                    return (f'db_schema error: unknown table {table!r}. '
                            f'Tables: {", ".join(tables)}')
                if not table:
                    return 'Tables: ' + ', '.join(tables)
                cur.execute(
                    'SELECT column_name, data_type, is_nullable '
                    'FROM information_schema.columns '
                    "WHERE table_schema = 'public' AND table_name = %s "
                    'ORDER BY ordinal_position', (table, ))
                columns = cur.fetchall()
        except Exception as error:  # noqa: BLE001
            return f'db_schema error: {type(error).__name__}: {str(error).strip()}'
        finally:
            conn.close()
        lines = [f'table {table}:']
        for name, kind, nullable in columns:
            suffix = '' if nullable == 'YES' else ' not null'
            lines.append(f'  {name} {kind}{suffix}')
        return '\n'.join(lines)

    def release(self) -> None:
        """Drop this instance's database role, if any. Safe to call repeatedly."""
        if self._db_creds is None:
            return
        role = self._db_creds['role']
        self._db_creds = None
        try:
            provision_dir = os.environ.get('TWINKLE_WORLD_DB_MODULE', '')
            if provision_dir and provision_dir not in sys.path:
                sys.path.insert(0, provision_dir)
            import provision
            provision.drop(role)
        except Exception as error:  # noqa: BLE001
            # Not fatal: the account is read-only and leased, so a failed drop
            # leaves an inert role rather than a live hazard.
            print(f'[world] failed to drop database role {role}: {error}')

    def _receipt(self, tool_name: str, arguments: dict[str, Any], result: str) -> None:
        """Record what was served, parsed enough to be checkable.

        The urls are extracted from the tool's own output rather than from its
        arguments: `fetch_page` takes one url, but `web_search` is asked a query
        and decides which urls to return, so the urls that were actually served
        are only knowable from the result.
        """
        entry = {'tool': tool_name, 'args': dict(arguments), 'urls': [], 'query': ''}
        if tool_name == 'web_search':
            entry['query'] = (arguments.get('query') or '').strip()
            try:
                payload = json.loads(result)
            except (TypeError, ValueError):
                payload = {}
            for hit in payload.get('results') or []:
                if isinstance(hit, dict) and hit.get('url'):
                    entry['urls'].append({
                        'url': hit['url'],
                        'published': hit.get('published') or '',
                        'title': hit.get('title') or ''
                    })
        elif tool_name == 'fetch_page':
            url = (arguments.get('url') or '').strip()
            # A fetch that failed is not a receipt for the page it named.
            if url and not result.startswith('fetch_page:'):
                entry['urls'].append({'url': url, 'published': '', 'title': ''})
        else:
            return
        self.receipts.append(entry)

    def _answer(self, tool_name: str, arguments: dict[str, Any]) -> str | None:
        if tool_name == 'world_sql':
            return self.sql(arguments.get('query'), arguments.get('params') or [])
        if tool_name == 'world_schema':
            return self.schema(arguments.get('name'))
        if tool_name == 'world_doc':
            return self.doc(arguments.get('name'))
        if tool_name == 'web_search':
            return self.web_search(
                arguments.get('query'), arguments.get('num_results'),
                arguments.get('freshness') or '')
        if tool_name == 'fetch_page':
            return self.fetch_page(arguments.get('url'), arguments.get('query') or '')
        if tool_name == 'db_query':
            return self.db_query(arguments.get('query'), arguments.get('params'))
        if tool_name == 'db_schema':
            return self.db_schema(arguments.get('table') or '')
        return None

    def call(self, tool_name: str, arguments: dict[str, Any]) -> str | None:
        """Dispatch a tool and record search receipts. Unknown names return None."""
        arguments = arguments or {}
        result = self._answer(tool_name, arguments)
        if result is not None:
            self._receipt(tool_name, arguments, result)
        return result


class WorldToolsMixin:
    """Add host tools before the underlying Env in the MRO.

    Use class Slot(WorldToolsMixin, LocalEnv). When combined with profiles,
    ProfiledEnvMixin must precede WorldToolsMixin so it filters world tools too.
    By default the caller owns world cleanup; release_world_on_close=True makes
    this environment responsible for releasing its world.
    """

    def __init__(self,
                 *args: Any,
                 world: HostWorld | None = None,
                 release_world_on_close: bool = False,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._world = world
        self._release_world_on_close = release_world_on_close

    @property
    def world(self) -> HostWorld | None:
        """The host world this environment reads from, or None."""
        return self._world

    def tools(self) -> list[ToolInfo]:
        declared = list(super().tools())
        if self._world is None:
            return declared
        # A name the environment already owns wins. Shadowing it silently would
        # promise the model one behaviour and hand it another.
        names = {(t.get('function') or {}).get('name') for t in declared}
        return declared + [
            schema for schema in self._world.schemas() if (schema.get('function') or {}).get('name') not in names
        ]

    def step(self, tool_name: str, arguments: dict[str, Any] = None) -> StepResult:
        answer = self._answer_world(tool_name, arguments or {})
        if answer is not None:
            return StepResult(observation=answer, reward=0.0, done=False)
        return super().step(tool_name, arguments)

    def step_batch(self, calls: Sequence[tuple[str, dict[str, Any]]]) -> list[StepResult]:
        """Dispatch host and sandbox calls separately, preserving their original order."""
        if self._world is None:
            return super().step_batch(calls)
        host_at: list[int] = []
        sandbox_at: list[int] = []
        for index, (name, _) in enumerate(calls):
            if _world_name(name) is not None:
                host_at.append(index)
            else:
                sandbox_at.append(index)
        out: list[StepResult | None] = [None] * len(calls)
        for index in host_at:
            name = _world_name(calls[index][0])
            answer = self._world.call(name, calls[index][1] or {})
            out[index] = StepResult(
                observation=(answer if answer is not None else f'Error: {name} is not available in this environment.'),
                reward=0.0,
                done=False)
        if sandbox_at:
            forwarded = super().step_batch([calls[i] for i in sandbox_at])
            if len(forwarded) != len(sandbox_at):
                raise RuntimeError(f'step_batch returned {len(forwarded)} results for '
                                   f'{len(sandbox_at)} forwarded call(s); cannot pair observations '
                                   'with calls, so refusing to guess')
            for index, result in zip(sandbox_at, forwarded):
                out[index] = result
        return [r if r is not None else StepResult(observation='Error: no result') for r in out]

    def _answer_world(self, tool_name: str, arguments: dict[str, Any]) -> str | None:
        """Return a host observation, or None for a non-world tool."""
        if self._world is None:
            return None
        name = _world_name(tool_name)
        if name is None:
            return None
        answer = self._world.call(name, arguments)
        if answer is None:
            return f'Error: {name} is not available in this environment.'
        return answer

    def close(self) -> None:
        try:
            super().close()
        finally:
            if self._release_world_on_close and self._world is not None:
                self._world.release()

    def clear(self) -> None:
        """Discard the owned world's previous episode state before reusing a slot."""
        if self._release_world_on_close and self._world is not None:
            self._world.release()
            self._world.receipts.clear()
        super().clear()
