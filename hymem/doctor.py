"""`hymem-doctor` — preflight diagnostics for a HyMem deployment.

Verifies that the environment is configured well enough to run the servers,
and surfaces the silent failure modes (missing keys, unreachable endpoints,
embedding-dimension drift) before they bite mid-request. Prints the *resolved*
configuration so there is no guessing about which provider/model is in use.

Exit code 0 if every check passes (warnings allowed), 1 if any check fails.
"""
from __future__ import annotations

import sys

from hymem.bootstrap import EnvConfig, resolve_env
from hymem.config import HyMemConfig
from hymem.core import db as core_db

OK, WARN, FAIL = "OK", "WARN", "FAIL"
_GLYPH = {OK: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]"}


class _Result:
    def __init__(self, status: str, name: str, detail: str) -> None:
        self.status = status
        self.name = name
        self.detail = detail

    def render(self) -> str:
        return f"{_GLYPH[self.status]} {self.name}: {self.detail}"


def _check_root(cfg: EnvConfig) -> _Result:
    try:
        cfg.root.mkdir(parents=True, exist_ok=True)
        probe = cfg.root / ".hymem-doctor-write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return _Result(OK, "storage root", f"{cfg.root} (writable)")
    except Exception as exc:  # noqa: BLE001
        return _Result(FAIL, "storage root", f"{cfg.root} not writable: {exc}")


def _check_llm(cfg: EnvConfig) -> _Result:
    if not cfg.has_llm_key:
        return _Result(
            FAIL, "extraction LLM",
            "no API key — set HYMEM_LLM_API_KEY (or DEEPSEEK_API_KEY / OPENAI_API_KEY)",
        )
    try:
        from openai import OpenAI
    except ImportError:
        return _Result(WARN, "extraction LLM",
                       "key present; openai package not installed, cannot verify")
    try:
        client = OpenAI(api_key=cfg.llm_api_key, base_url=cfg.llm_base_url)
        client.models.list()
        return _Result(OK, "extraction LLM",
                       f"{cfg.llm_model} @ {cfg.llm_base_url} (reachable)")
    except Exception as exc:  # noqa: BLE001
        return _Result(FAIL, "extraction LLM",
                       f"{cfg.llm_model} @ {cfg.llm_base_url} unreachable: {exc}")


def _check_embedding(cfg: EnvConfig) -> tuple[_Result, int | None]:
    """Returns the check result and the live embedding dimension (or None)."""
    if not cfg.has_embedding_key:
        return (
            _Result(WARN, "embeddings",
                    "no API key — FTS-only retrieval (vector search disabled)"),
            None,
        )
    try:
        from hymem.contrib.openai_embedding_client import OpenAICompatibleEmbeddingClient
    except ImportError:
        return _Result(WARN, "embeddings", "key present; openai package not installed"), None
    try:
        embedder = OpenAICompatibleEmbeddingClient(
            api_key=cfg.embedding_api_key,
            base_url=cfg.embedding_base_url,
            model=cfg.embedding_model,
        )
        embedder.embed(["preflight probe"])  # also resolves the true dimension
        return (
            _Result(OK, "embeddings",
                    f"{cfg.embedding_model} @ {cfg.embedding_base_url} "
                    f"(reachable, dim={embedder.dim})"),
            embedder.dim,
        )
    except Exception as exc:  # noqa: BLE001
        return _Result(FAIL, "embeddings",
                       f"{cfg.embedding_model} @ {cfg.embedding_base_url} "
                       f"unreachable: {exc}"), None


def _check_sqlite_vec() -> _Result:
    try:
        import sqlite_vec  # noqa: F401
    except ImportError:
        return _Result(WARN, "sqlite-vec",
                       "extension not installed — falls back to Python cosine search")
    import sqlite3
    try:
        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.close()
        return _Result(OK, "sqlite-vec", "extension loads (native vector search available)")
    except Exception as exc:  # noqa: BLE001
        return _Result(WARN, "sqlite-vec",
                       f"failed to load ({exc}) — falls back to Python cosine search")


def _check_schema_and_dim(cfg: EnvConfig, live_dim: int | None) -> list[_Result]:
    results: list[_Result] = []
    hy_cfg = HyMemConfig(root=cfg.root)
    try:
        conn = core_db.connect(hy_cfg.db_path)
        core_db.initialize(conn)
        version = core_db.schema_version(conn)
        results.append(_Result(OK, "schema",
                               f"initialized/migrated cleanly (version {version})"))
    except Exception as exc:  # noqa: BLE001
        results.append(_Result(FAIL, "schema", f"initialize/migrate failed: {exc}"))
        return results

    try:
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'vec_dim'"
        ).fetchone()
        stored_dim = int(row["value"]) if row else None
    except Exception:  # noqa: BLE001
        stored_dim = None
    finally:
        conn.close()

    if stored_dim is None:
        results.append(_Result(OK, "embedding dimension",
                               "no vector table yet — will be created on first dream"))
    elif live_dim is None:
        results.append(_Result(WARN, "embedding dimension",
                               f"stored dim={stored_dim}; embedding client not "
                               "verified, cannot confirm a match"))
    elif live_dim != stored_dim:
        results.append(_Result(
            FAIL, "embedding dimension",
            f"MISMATCH: configured model produces dim={live_dim} but the "
            f"existing vector table is dim={stored_dim}. Vector search will "
            f"silently return garbage. Re-embed (rebuild vec_chunks) or revert "
            f"the embedding model.",
        ))
    else:
        results.append(_Result(OK, "embedding dimension",
                               f"configured model matches stored table (dim={live_dim})"))
    return results


def run_doctor() -> int:
    cfg = resolve_env()

    print("HyMem doctor — resolved configuration")
    print("─" * 60)
    print(f"  storage root      : {cfg.root}")
    print(f"  LLM model         : {cfg.llm_model}")
    print(f"  LLM base URL      : {cfg.llm_base_url}")
    print(f"  LLM API key       : {'set' if cfg.has_llm_key else 'MISSING'}")
    print(f"  embedding model   : {cfg.embedding_model}")
    print(f"  embedding base URL: {cfg.embedding_base_url}")
    print(f"  embedding API key : {'set' if cfg.has_embedding_key else 'missing (FTS-only)'}")
    print("─" * 60)

    results: list[_Result] = [_check_root(cfg), _check_llm(cfg), _check_sqlite_vec()]
    embedding_result, live_dim = _check_embedding(cfg)
    results.append(embedding_result)
    results.extend(_check_schema_and_dim(cfg, live_dim))

    for r in results:
        print(r.render())
    print("─" * 60)

    fails = sum(1 for r in results if r.status == FAIL)
    warns = sum(1 for r in results if r.status == WARN)
    if fails:
        print(f"{fails} failure(s), {warns} warning(s) — HyMem is not ready to run.")
        return 1
    if warns:
        print(f"0 failures, {warns} warning(s) — HyMem can run with reduced functionality.")
        return 0
    print("All checks passed — HyMem is ready.")
    return 0


def main() -> None:
    sys.exit(run_doctor())


if __name__ == "__main__":
    main()
