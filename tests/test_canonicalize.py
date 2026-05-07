from __future__ import annotations

from hymem.dreaming.canonicalize import merge, normalize, register_alias, resolve


def test_normalize_strips_articles_punct_and_paren():
    assert normalize("The Docker (container)") == "docker"
    assert normalize("  uv ") == "uv"
    assert normalize("Local Dev Environment") == "local_dev_environment"
    assert normalize("Postgres 18") == "postgres_18"
    assert normalize("Café") == "cafe"


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
