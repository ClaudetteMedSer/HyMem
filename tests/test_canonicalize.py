from __future__ import annotations

from hymem.dreaming.canonicalize import merge, normalize, register_alias, resolve


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
