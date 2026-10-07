"""Independent offline boundaries for the finite retry census."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3

import pytest

from tools.diagnostics import siwc_lme_retry_reasons_v1 as m


def _scope(root):
    # Execute the actual census definitions, not the SSH/host gate, on invented
    # local SQLite fixtures. Source/terminal binding is checked separately.
    tree = ast.parse(m.REMOTE)
    tree.body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    scope = {'root': root, 'reader': {'ROOT_UID': os.getuid()},
             'REASONS': m.REASONS, 'MAX_ROWS': m.MAX_ROWS}
    exec(compile(tree, '<root-census-fixture>', 'exec'), scope)
    return scope


def _database(tmp_path, schema, rows):
    directory = tmp_path / 'run' / 'q-0000'
    directory.mkdir(parents=True)
    path = directory / 'hymem.sqlite'
    with sqlite3.connect(path) as conn:
        conn.executescript(schema)
        if rows:
            conn.executemany('INSERT INTO chunk_extraction_attempts VALUES(?,?)', rows)
    return path


def test_exact_reason_classification_and_no_writes(tmp_path):
    path = _database(tmp_path,
        'CREATE TABLE chunk_extraction_attempts(attempts, last_failure_reason TEXT COLLATE NOCASE);',
        [(1, 'PARSE_FAILURE'), (2, 'parse_failure'), (3, 'parse_failure\x00private'),
         (1, b'parse_failure'), (2, None)])
    before = path.read_bytes()
    value = _scope(tmp_path)['_census'](path)
    assert value['rows'] == 5
    assert value['by_attempt']['1']['other'] == 2
    assert value['by_attempt']['2']['parse_failure'] == 1
    assert value['by_attempt']['2']['other'] == 1
    assert value['by_attempt']['3']['other'] == 1
    assert path.read_bytes() == before
    assert set(p.name for p in path.parent.iterdir()) == {'hymem.sqlite'}
    assert 'private' not in json.dumps(value)


def test_view_cannot_read_private_columns(tmp_path):
    path = _database(tmp_path,
        'CREATE TABLE secret(value TEXT);'
        "INSERT INTO secret VALUES('private-row-text');"
        'CREATE VIEW chunk_extraction_attempts AS SELECT 1 AS attempts, value AS last_failure_reason FROM secret;',
        [])
    with pytest.raises(sqlite3.DatabaseError):
        _scope(tmp_path)['_census'](path)


def test_world_writable_database_and_parent_rejected(tmp_path):
    path = _database(tmp_path,
        'CREATE TABLE chunk_extraction_attempts(attempts, last_failure_reason);', [])
    scope = _scope(tmp_path)
    for target in (path, path.parent):
        original = target.stat().st_mode & 0o777
        target.chmod(original | 0o002)
        with pytest.raises(ValueError, match='unsafe_path'):
            scope['_census'](path)
        target.chmod(original)


def test_projection_rejects_extra_private_shapes_and_invalid_counts():
    good = {'schema': m.SCHEMA, 'source_receipt_terminal_cleanup_verified': True,
        'questions': {f'q-{i:04d}': {'rows': 0, 'by_attempt': {
            str(n): {code: 0 for code in m.REASONS} for n in (1, 2, 3)}} for i in range(4)}}
    assert m._validated(good) is good
    checks = 0
    for qid in good['questions']:
        for n in ('1', '2', '3'):
            for invalid in (True, -1, 1.0, None, 'private', float('nan'), m.MAX_ROWS + 1):
                bad = copy.deepcopy(good)
                bad['questions'][qid]['by_attempt'][n]['other'] = invalid
                assert m._validated(bad) is None
                checks += 1
    for where in ('top', 'question', 'bucket'):
        bad = copy.deepcopy(good)
        place = bad if where == 'top' else bad['questions']['q-0000']
        if where == 'bucket':
            place = place['by_attempt']['1']
        place['private'] = 'must not leave host'
        assert m._validated(bad) is None
        checks += 1
    assert checks == 87


def test_historical_pins_and_reason_taxonomy():
    assert hashlib.sha256(m.METADATA.read_bytes()).hexdigest() == m.METADATA_SHA
    assert hashlib.sha256(m.READER.read_bytes()).hexdigest() == m.READER_SHA
    source = Path('/private/tmp/hymem-siwc-pilot-source-fUk67B2v/bundle/candidate/hymem/extraction/chunk.py')
    assert hashlib.sha256(source.read_bytes()).hexdigest() == 'c644513152e2ddefbe0d0d18dce7d597d18b4ba78ce275dd04ad4b2133ee7b92'
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == '_FAILURE_REASONS' for t in n.targets))
    assert set(ast.literal_eval(node.value.args[0])) | {'other'} == set(m.REASONS)
