"""Independent recovery controls; invented sources, no provider calls."""
from contextlib import closing
from dataclasses import replace
import json

import pytest

from hymem import HyMem
from hymem.core import db
from hymem.dreaming import facts
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.extraction.llm import StubLLMClient


def setup_store(cfg, client, texts):
    hy = HyMem(replace(cfg, aggregation_nodes_enabled=False,
                      profile_extraction_enabled=False), llm=client)
    ids = [hy.log_message("root-capacity", "user", text) for text in texts]
    with db.transaction(hy.conn):
        materialize_message_coverage(hy.conn, "root-capacity")
    return hy, ids


class SourceClient(StubLLMClient):
    def __init__(self, texts):
        super().__init__()
        self.texts = texts

    def complete(self, request):
        self.calls.append(request)
        return json.dumps({"facts": [
            {"text": text, "date": None, "entities": ["Dax"]}
            for text in self.texts if text in request.user
        ]})


def test_root_capacity_recovery_commits_only_smaller_prefix_then_all_tail(cfg):
    texts = [f"Dax shipped item {i} with note " + str(i) * 320 + "." for i in range(9)]
    client = SourceClient(texts)
    hy, ids = setup_store(cfg, client, texts)
    with closing(hy):
        original = tuple(tuple(row) for row in hy.conn.execute("SELECT * FROM messages"))
        first = facts.extract_facts(hy.conn, "root-capacity", client, hy.config,
                                    max_chars=12000)
        assert len(client.calls) == 2
        assert client.calls[0].user != client.calls[1].user
        assert not first.parse_failed and 0 < len(first.items) <= 8
        assert first.covered_message_id in ids[:-1] and not first.caught_up
        assert first.partial_message_id is None
        assert hy.conn.execute("SELECT COUNT(*) FROM fact_extraction_outcomes").fetchone()[0] == 0
        with db.transaction(hy.conn):
            facts.persist_facts(hy.conn, "root-capacity", first)
        tail = facts.extract_facts(hy.conn, "root-capacity", client, hy.config,
                                   since_message_id=first.covered_message_id,
                                   max_chars=12000)
        assert not tail.parse_failed and tail.caught_up
        assert tail.covered_message_id == ids[-1] and len(client.calls) == 3
        with db.transaction(hy.conn):
            facts.persist_facts(hy.conn, "root-capacity", tail)
        stored = {row[0] for row in hy.conn.execute("SELECT text FROM narrative_facts")}
        assert stored == set(texts)
        assert tuple(tuple(row) for row in hy.conn.execute("SELECT * FROM messages")) == original
        assert hy.conn.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("failure", [RuntimeError("transport stopped"), KeyboardInterrupt()])
def test_root_second_call_failure_is_not_swallowed_or_persisted(cfg, failure):
    texts = [f"Dax shipped item {i}. " + str(i) * 300 for i in range(9)]
    class Stops(SourceClient):
        def complete(self, request):
            if self.calls:
                self.calls.append(request)
                raise failure
            return super().complete(request)
    client = Stops(texts)
    hy, _ = setup_store(cfg, client, texts)
    with closing(hy):
        before = tuple(tuple(row) for row in hy.conn.execute("SELECT * FROM sessions"))
        with pytest.raises(type(failure)):
            facts.extract_facts(hy.conn, "root-capacity", client, hy.config)
        assert len(client.calls) == 2
        assert tuple(tuple(row) for row in hy.conn.execute("SELECT * FROM sessions")) == before
        assert hy.conn.execute("SELECT COUNT(*) FROM narrative_facts").fetchone()[0] == 0
        assert hy.conn.execute("SELECT COUNT(*) FROM fact_extraction_outcomes").fetchone()[0] == 0
