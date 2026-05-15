from __future__ import annotations

from hymem.dreaming.canonicalize import (
    find_canonical_drift,
    merge,
    normalize,
    register_alias,
    repair_canonical_drift,
    resolve,
)


def test_normalize_strips_articles_punct_and_paren():
    assert normalize("The Docker (container)") == "docker"
    assert normalize("  uv ") == "uv"
    assert normalize("Local Dev Environment") == "local_dev_environment"
    assert normalize("Postgres 18") == "postgres_18"
    assert normalize("Café") == "cafe"


def test_normalize_strips_latin_script_articles():
    # Leading articles across common Latin-script European languages.
    assert normalize("de Docker") == "docker"
    assert normalize("het Platform") == "platform"
    assert normalize("der Server") == "server"
    assert normalize("le Cache") == "cache"
    assert normalize("el Servidor") == "servidor"
    # Accent-folding stays consistent whether or not an article leads.
    assert normalize("de café-server") == normalize("café-server") == "cafe_server"


def test_match_known_entities_handles_accented_tokens(hy):
    from hymem.query.entities import match_known_entities

    conn = hy.conn
    register_alias(conn, "préfère-tool", "prefere_tool")
    # The accented mention must tokenize whole and resolve — not shred at the
    # accent (which the old [A-Za-z] tokenizer did).
    hits = match_known_entities(conn, "do we still use préfère-tool here?")
    assert "prefere_tool" in hits


def test_match_known_entities_filters_one_off_object_gerund(hy):
    """An object-only canonical with a single edge and no other entity-shape
    signal is treated as an LLM artefact (e.g. a gerund extracted as object)
    and not surfaced as a known entity."""
    from hymem.query.entities import match_known_entities

    conn = hy.conn
    # `working` exists only as an object in one edge — no subject usage, no
    # entity_types row, no other edges. Should be filtered.
    conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical) "
        "VALUES ('deepseek', 'rejects', 'working')"
    )
    hits = match_known_entities(conn, "is anything working here")
    assert "working" not in hits


def test_match_known_entities_keeps_object_with_multiple_edges(hy):
    from hymem.query.entities import match_known_entities

    conn = hy.conn
    conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical) "
        "VALUES ('backend', 'uses', 'postgres')"
    )
    conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical) "
        "VALUES ('worker', 'uses', 'postgres')"
    )
    hits = match_known_entities(conn, "what about postgres")
    assert "postgres" in hits


def test_match_known_entities_keeps_object_with_entity_type(hy):
    from hymem.query.entities import match_known_entities

    conn = hy.conn
    conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical) "
        "VALUES ('backend', 'uses', 'redis')"
    )
    conn.execute(
        "INSERT INTO entity_types(entity_canonical, type, confidence) "
        "VALUES ('redis', 'database', 0.9)"
    )
    hits = match_known_entities(conn, "tell me about redis")
    assert "redis" in hits


def test_match_known_entities_keeps_object_also_used_as_subject(hy):
    from hymem.query.entities import match_known_entities

    conn = hy.conn
    # `medflow` appears once as object, once as subject — entity-shaped.
    conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical) "
        "VALUES ('atta', 'part_of', 'medflow')"
    )
    conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical) "
        "VALUES ('medflow', 'uses', 'fast_api')"
    )
    hits = match_known_entities(conn, "tell me about medflow")
    assert "medflow" in hits


def test_normalize_splits_camelcase():
    assert normalize("MedFlow") == "med_flow"
    assert normalize("med-flow") == "med_flow"
    assert normalize("MedFlow") == normalize("med-flow")
    assert normalize("FastAPI") == "fast_api"
    assert normalize("HyMem") == "hy_mem"


def test_normalize_splits_acronym_prefix():
    assert normalize("JSONParser") == "json_parser"
    assert normalize("XMLHttpRequest") == "xml_http_request"
    assert normalize("URLParser") == "url_parser"
    assert normalize("API") == "api"
    assert normalize("JSONParser") == normalize("json parser")


def test_resolve_uses_alias_when_present(hy):
    conn = hy.conn
    register_alias(conn, "DockerCE", "docker")
    assert resolve(conn, "DockerCE") == "docker"
    # Unknown surface falls back to its normalized form.
    assert resolve(conn, "Kubernetes") == "kubernetes"


def test_merge_folds_edges_into_kept_canonical(hy):
    conn = hy.conn
    # Seed two edges on different canonical names.
    conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, pos_evidence) "
        "VALUES ('local_dev', 'uses', 'docker_ce', 2)"
    )
    conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, pos_evidence) "
        "VALUES ('local_dev', 'uses', 'docker', 3)"
    )

    merge(conn, keep="docker", drop="docker_ce")

    rows = conn.execute(
        "SELECT subject_canonical, predicate, object_canonical, pos_evidence "
        "FROM knowledge_graph ORDER BY object_canonical"
    ).fetchall()
    # Only one edge remains, evidence summed.
    assert len(rows) == 1
    assert rows[0]["object_canonical"] == "docker"
    assert rows[0]["pos_evidence"] == 5

    # Alias is recorded so future surface forms resolve to the survivor.
    assert resolve(conn, "docker_ce") == "docker"


def test_find_canonical_drift_reports_all_four_columns(hy):
    conn = hy.conn
    # Drift in entity_aliases.canonical (the YantrikDB-style symptom).
    conn.execute(
        "INSERT INTO entity_aliases(alias, canonical) VALUES (?, ?)",
        ("atta", "Atta_van_Westreenen"),
    )
    # Drift in entity_aliases.alias (uppercase alias key).
    conn.execute(
        "INSERT INTO entity_aliases(alias, canonical) VALUES (?, ?)",
        ("Mixed_Case_Key", "clean_value"),
    )
    # Drift in knowledge_graph subject and object columns.
    conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical) "
        "VALUES ('Bad_Subject', 'uses', 'clean_obj')"
    )
    conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical) "
        "VALUES ('clean_subj', 'uses', 'Bad_Object')"
    )

    findings = find_canonical_drift(conn)
    locations = {loc for loc, _ in findings}
    assert "entity_aliases.canonical" in locations
    assert "entity_aliases.alias" in locations
    assert "knowledge_graph.subject_canonical" in locations
    assert "knowledge_graph.object_canonical" in locations


def test_find_canonical_drift_returns_empty_when_clean(hy):
    conn = hy.conn
    register_alias(conn, "docker", "docker")
    conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical) "
        "VALUES ('docker', 'uses', 'postgres')"
    )
    assert find_canonical_drift(conn) == []


def test_repair_canonical_drift_normalizes_canonicals(hy):
    conn = hy.conn
    conn.execute(
        "INSERT INTO entity_aliases(alias, canonical) VALUES (?, ?)",
        ("atta", "Atta_van_Westreenen"),
    )
    conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, pos_evidence) "
        "VALUES ('Atta_van_Westreenen', 'part_of', 'medflow', 5)"
    )

    fixes = repair_canonical_drift(conn)

    assert any(
        f["column"] == "canonical" and f["from"] == "Atta_van_Westreenen"
        and f["to"] == "atta_van_westreenen"
        for f in fixes
    )
    # Alias canonical rewritten in place.
    row = conn.execute(
        "SELECT canonical FROM entity_aliases WHERE alias = 'atta'"
    ).fetchone()
    assert row["canonical"] == "atta_van_westreenen"
    # Knowledge graph subject rewritten.
    row = conn.execute(
        "SELECT subject_canonical FROM knowledge_graph WHERE predicate = 'part_of'"
    ).fetchone()
    assert row["subject_canonical"] == "atta_van_westreenen"
    # No drift remains anywhere.
    assert find_canonical_drift(conn) == []


def test_repair_canonical_drift_merges_collisions(hy):
    conn = hy.conn
    # Two edges that share canonical (subject, predicate, object) once the
    # drifted form is normalized. Evidence should sum, duplicate removed.
    conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, pos_evidence) "
        "VALUES ('docker', 'uses', 'postgres', 3)"
    )
    conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, pos_evidence) "
        "VALUES ('Docker', 'uses', 'postgres', 2)"
    )

    repair_canonical_drift(conn)

    rows = conn.execute(
        "SELECT subject_canonical, pos_evidence FROM knowledge_graph "
        "WHERE predicate = 'uses' AND object_canonical = 'postgres'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["subject_canonical"] == "docker"
    assert rows[0]["pos_evidence"] == 5


def test_repair_canonical_drift_handles_alias_key_drift(hy):
    conn = hy.conn
    conn.execute(
        "INSERT INTO entity_aliases(alias, canonical) VALUES (?, ?)",
        ("Mixed_Key", "clean_canonical"),
    )

    fixes = repair_canonical_drift(conn)

    assert any(
        f["column"] == "alias" and f["from"] == "Mixed_Key"
        and f["to"] == "mixed_key" and not f.get("collision")
        for f in fixes
    )
    rows = conn.execute(
        "SELECT alias, canonical FROM entity_aliases"
    ).fetchall()
    assert (rows[0]["alias"], rows[0]["canonical"]) == ("mixed_key", "clean_canonical")


def test_repair_canonical_drift_drops_drifted_alias_on_collision(hy):
    conn = hy.conn
    # Drifted alias key whose normalized form is already taken.
    conn.execute(
        "INSERT INTO entity_aliases(alias, canonical) VALUES (?, ?)",
        ("docker", "docker"),
    )
    conn.execute(
        "INSERT INTO entity_aliases(alias, canonical) VALUES (?, ?)",
        ("Docker", "other_canonical"),
    )

    fixes = repair_canonical_drift(conn)

    assert any(
        f["column"] == "alias" and f["from"] == "Docker"
        and f.get("collision") is True
        for f in fixes
    )
    # The pre-existing normalized row survives, the drifted one is gone.
    rows = conn.execute(
        "SELECT alias, canonical FROM entity_aliases ORDER BY alias"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["alias"] == "docker"
    assert rows[0]["canonical"] == "docker"
