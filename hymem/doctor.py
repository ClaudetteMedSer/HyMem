"""`hymem-doctor` — preflight diagnostics for a HyMem deployment.

Verifies that the environment is configured well enough to run the servers,
and surfaces the silent failure modes (missing keys, unreachable endpoints,
embedding model/dimension drift) before they bite mid-request. Prints the *resolved*
configuration so there is no guessing about which provider/model is in use.

Exit code 0 if every check passes (warnings allowed), 1 if any check fails.
"""
from __future__ import annotations

import sqlite3
import sys

from hymem.bootstrap import EnvConfig, resolve_env
from hymem.config import HyMemConfig
from hymem.contrib.endpoint_policy import safe_endpoint_label
from hymem.core import db as core_db
from hymem.core.vectors import decode_vector, encode_vector

OK, WARN, FAIL = "OK", "WARN", "FAIL"
_GLYPH = {OK: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]"}


class _Result:
    def __init__(self, status: str, name: str, detail: str) -> None:
        self.status = status
        self.name = name
        self.detail = detail

    def render(self) -> str:
        return f"{_GLYPH[self.status]} {self.name}: {self.detail}"


def _close_probe(
    resource,
    *,
    primary_exception: BaseException | None,
    reported_failure: Exception | None = None,
) -> None:
    """Close one doctor-owned transport with explicit failure precedence.

    ``primary_exception`` is exception control flow currently propagating and
    must never be replaced.  ``reported_failure`` is an ordinary connectivity
    exception already converted into a FAIL result: an ordinary close fault is
    secondary to it, but cleanup control flow must still propagate unchanged.
    """

    close = getattr(resource, "close", None)
    if not callable(close):
        return
    try:
        close()
    except BaseException as exc:
        if primary_exception is not None:
            try:
                primary_exception.add_note(
                    f"doctor probe cleanup failed: {type(exc).__name__}"
                )
            except (AttributeError, TypeError):  # pragma: no cover
                pass
            return
        if reported_failure is not None and isinstance(exc, Exception):
            try:
                reported_failure.add_note(
                    f"doctor probe cleanup failed: {type(exc).__name__}"
                )
            except (AttributeError, TypeError):  # pragma: no cover
                pass
            return
        raise


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
    display_url = safe_endpoint_label(cfg.llm_base_url, label="LLM")
    from hymem.contrib.model_policy import (
        DeprecatedModelAliasError,
        require_active_model,
    )
    try:
        require_active_model(cfg.llm_model, role="HyMem doctor LLM")
    except DeprecatedModelAliasError as exc:
        return _Result(FAIL, "extraction LLM", str(exc))
    if not cfg.has_llm_key:
        return _Result(
            FAIL, "extraction LLM",
            "no API key authorized for endpoint — set HYMEM_LLM_API_KEY or "
            "the matching exact-origin provider key",
        )
    try:
        from hymem.contrib.openai_client import OpenAICompatibleClient
        from hymem.extraction.llm import LLMRequest
    except ImportError:
        return _Result(WARN, "extraction LLM",
                       "key present; openai package not installed, cannot verify")
    # Probe with a minimal chat completion — the capability HyMem actually
    # uses. /v1/models is not exposed by every OpenAI-compatible proxy, so a
    # models.list() failure would be a false negative.
    client = None
    failure: Exception | None = None
    try:
        client = OpenAICompatibleClient(
            api_key=cfg.llm_api_key, base_url=cfg.llm_base_url, model=cfg.llm_model,
        )
        client.complete(LLMRequest(
            system="", user="ping", response_format="text", max_tokens=1,
        ))
        result = _Result(OK, "extraction LLM",
                         f"{cfg.llm_model} @ {display_url} (reachable)")
    except Exception as exc:  # noqa: BLE001
        failure = exc
        result = _Result(FAIL, "extraction LLM",
                         f"{cfg.llm_model} @ {display_url} unreachable: "
                         f"{type(exc).__name__}")
    finally:
        _close_probe(
            client,
            # The connectivity exception is represented by ``result`` rather
            # than re-raised, but it is still the authoritative probe failure:
            # a secondary transport-close fault must not replace that result.
            # If control flow (KeyboardInterrupt/SystemExit/DeadlineExceeded)
            # is active, ``sys.exc_info`` supplies that exact primary instead.
            primary_exception=(
                sys.exc_info()[1] if failure is None else None
            ),
            reported_failure=failure,
        )
    return result


def _check_embedding(
    cfg: EnvConfig,
) -> tuple[_Result, int | None, str | None]:
    """Return result plus the exact live storage-space identity."""
    from hymem.dreaming.aggregation_material import embedding_execution_identity

    if cfg.embedding_backend == "local_feature_hash":
        from hymem.extraction.embeddings import LocalHashEmbeddingClient
        embedder = LocalHashEmbeddingClient(
            dim_value=cfg.embedding_dim, model_name=cfg.embedding_model
        )
        embedder.embed(["preflight probe"])
        _binding, model_key, live_dim = embedding_execution_identity(embedder)
        status = OK
        if cfg.embedding_fallback_reason == "remote_embedding_credentials_missing":
            status = WARN
        elif cfg.embedding_fallback_reason == "remote_embedding_endpoint_rejected":
            status = FAIL
        fallback_detail = (
            f", fallback_reason={cfg.embedding_fallback_reason}"
            if cfg.embedding_fallback_reason else ""
        )
        return (
            _Result(
                status, "embeddings",
                f"{model_key} (local deterministic lexical fallback, "
                f"no network, dim={embedder.dim}{fallback_detail})",
            ),
            live_dim,
            model_key,
        )
    if not cfg.has_embedding_key:
        return _Result(FAIL, "embeddings", "remote backend has no API key"), None, None
    if (
        not cfg.embedding_pin_dimension
        or not cfg.embedding_deployment_revision
        or not cfg.embedding_deployment_tenant
    ):
        return (
            _Result(
                FAIL, "embeddings",
                "remote backend lacks pinned dimension/deployment/tenant authority",
            ),
            None,
            None,
        )
    try:
        from hymem.contrib.openai_embedding_client import (
            OpenAICompatibleEmbeddingClient,
            safe_embedding_base_url,
        )
    except ImportError:
        return _Result(WARN, "embeddings", "key present; openai package not installed"), None, None
    display_url = safe_embedding_base_url(cfg.embedding_base_url)
    embedder = None
    failure: Exception | None = None
    try:
        embedder = OpenAICompatibleEmbeddingClient(
            api_key=cfg.embedding_api_key,
            base_url=cfg.embedding_base_url,
            model=cfg.embedding_model,
            dim=cfg.embedding_dim,
            pin_dimension=cfg.embedding_pin_dimension,
            deployment_revision=cfg.embedding_deployment_revision,
            deployment_tenant=cfg.embedding_deployment_tenant,
        )
        embedder.embed(["preflight probe"])  # also resolves the true dimension
        _binding, model_key, live_dim = embedding_execution_identity(embedder)
        result = (
            _Result(OK, "embeddings",
                    f"exact producer {model_key} @ {display_url} "
                    f"(reachable, dim={live_dim})"),
            live_dim,
            model_key,
        )
    except Exception as exc:  # noqa: BLE001
        failure = exc
        result = (
            _Result(FAIL, "embeddings",
                    f"configured producer @ {display_url} "
                    f"unreachable ({type(exc).__name__})"),
            None,
            None,
        )
    finally:
        _close_probe(
            embedder,
            primary_exception=(
                sys.exc_info()[1] if failure is None else None
            ),
            reported_failure=failure,
        )
    return result


def _check_sqlite_vec() -> _Result:
    try:
        import sqlite_vec  # noqa: F401
    except ImportError:
        return _Result(WARN, "sqlite-vec",
                       "extension not installed — exact durable vector scoring remains available")
    import sqlite3
    try:
        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.close()
        return _Result(OK, "sqlite-vec", "extension loads (vector shadows available)")
    except Exception as exc:  # noqa: BLE001
        return _Result(WARN, "sqlite-vec",
                       f"failed to load ({exc}) — exact durable vector scoring remains available")


def _check_schema_and_dim(
    cfg: EnvConfig, live_dim: int | None, live_model: str | None,
) -> list[_Result]:
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

    metadata_error: str | None = None
    try:
        dim_row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'vec_dim'"
        ).fetchone()
        model_row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'vec_model'"
        ).fetchone()
        vec_tables = [
            row["name"] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name LIKE 'vec_%' ORDER BY name"
            ).fetchall()
        ]
        stored_dim = None
        if dim_row is not None:
            try:
                stored_dim = int(dim_row["value"])
            except (TypeError, ValueError, OverflowError):
                metadata_error = f"malformed vec_dim={dim_row['value']!r}"
            else:
                if stored_dim <= 0:
                    metadata_error = f"invalid vec_dim={stored_dim!r}"
        stored_model = model_row["value"] if model_row else None
        if model_row is not None and (
            not isinstance(stored_model, str) or not stored_model
        ):
            metadata_error = (
                f"{metadata_error}; " if metadata_error else ""
            ) + "malformed vec_model"
    except Exception as exc:  # noqa: BLE001
        stored_dim = None
        stored_model = None
        vec_tables = []
        metadata_error = f"could not read vector metadata: {exc}"
    finally:
        conn.close()

    if metadata_error is not None:
        results.append(_Result(
            FAIL, "embedding identity",
            f"{metadata_error}; durable model/dimension filters remain safe, "
            "but rebuild vector shadows with the configured embedder",
        ))
    elif stored_dim is None and (stored_model is not None or vec_tables):
        results.append(_Result(
            WARN, "embedding identity",
            "vector shadow metadata is incomplete "
            f"(vec_model={stored_model!r}, tables={vec_tables}); the next "
            "embedding persist will rebuild it",
        ))
    elif stored_dim is None:
        results.append(_Result(OK, "embedding dimension",
                               "no vector table yet — will be created on first dream"))
    elif stored_model is None:
        results.append(_Result(
            WARN, "embedding identity",
            f"stored vec_dim={stored_dim} has no vec_model metadata; durable "
            "model filters remain safe and the next embedding persist will rebuild shadows",
        ))
    elif live_dim is None:
        results.append(_Result(WARN, "embedding identity",
                               f"stored model={stored_model} dim={stored_dim}; "
                               "embedding client not verified"))
    elif live_dim != stored_dim or live_model != stored_model:
        results.append(_Result(
            FAIL, "embedding identity",
            f"MISMATCH: configured identity={live_model} dim={live_dim}; "
            f"stored vec model={stored_model} dim={stored_dim}. Retrieval skips "
            "incompatible durable rows; run a dream to re-embed/rebuild shadows "
            "or restore the prior model.",
        ))
    else:
        results.append(_Result(
            OK, "embedding identity",
            f"configured identity={live_model} dim={live_dim} matches stored shadows",
        ))
    return results


def _check_canonical_drift(cfg: EnvConfig) -> _Result:
    """Surface canonicals that fail normalize(v) == v. Advisory only — auto-
    repair can collide with existing rows and needs the operator's judgement.
    See `hymem.dreaming.canonicalize.repair_canonical_drift` for the fix."""
    from hymem.dreaming.canonicalize import find_canonical_drift

    hy_cfg = HyMemConfig(root=cfg.root)
    try:
        conn = core_db.connect(hy_cfg.db_path)
        core_db.initialize(conn)
        findings = find_canonical_drift(conn)
    except Exception as exc:  # noqa: BLE001
        return _Result(WARN, "canonical drift", f"could not check: {exc}")
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass

    if not findings:
        return _Result(OK, "canonical drift", "all canonicals are normalized")
    by_loc: dict[str, int] = {}
    for loc, _ in findings:
        by_loc[loc] = by_loc.get(loc, 0) + 1
    sample = ", ".join(f"{v!r} in {loc}" for loc, v in findings[:3])
    breakdown = ", ".join(f"{n} in {loc}" for loc, n in sorted(by_loc.items()))
    return _Result(
        WARN, "canonical drift",
        f"{len(findings)} drifted value(s) ({breakdown}); sample: {sample}. "
        f"Run hymem.dreaming.canonicalize.repair_canonical_drift(conn) to fix.",
    )


def repack_embeddings(conn: sqlite3.Connection) -> int:
    """Re-encode legacy JSON-text vectors to the compact packed form across all
    embedding tables. Optional, idempotent operator maintenance — new writes are
    already packed and reads transparently handle both forms, so legacy rows
    also convert lazily on any rewrite. Returns the number of rows repacked.

    Caller owns the transaction. Rows already packed (vector_json LIKE
    'b64f32:%') are skipped by the SQL filter, so re-running is cheap.
    """
    # (table, key_columns) — key columns uniquely identify a row for UPDATE.
    targets = [
        ("chunk_embeddings", ("chunk_id",)),
        ("edge_embeddings", ("edge_text",)),
        ("episode_embeddings", ("episode_id",)),
        ("message_embeddings", ("message_id",)),
        ("narrative_fact_embeddings", ("fact_id",)),
        ("embedding_cache", ("text_hash", "model")),
    ]
    repacked = 0
    for table, keys in targets:
        key_cols = ", ".join(keys)
        rows = conn.execute(
            f"SELECT {key_cols}, vector_json FROM {table} "
            f"WHERE vector_json NOT LIKE 'b64f32:%'"
        ).fetchall()
        where = " AND ".join(f"{k} = ?" for k in keys)
        for r in rows:
            try:
                packed = encode_vector(decode_vector(r["vector_json"]))
            except (ValueError, TypeError):
                continue
            conn.execute(
                f"UPDATE {table} SET vector_json = ? WHERE {where}",
                (packed, *(r[k] for k in keys)),
            )
            repacked += 1
    return repacked


def run_doctor() -> int:
    cfg = resolve_env()
    from hymem.contrib.openai_embedding_client import safe_embedding_base_url

    print("HyMem doctor — resolved configuration")
    print("─" * 60)
    print(f"  storage root      : {cfg.root}")
    print(f"  LLM model         : {cfg.llm_model}")
    print(f"  LLM base URL      : {safe_endpoint_label(cfg.llm_base_url, label='LLM')}")
    print(f"  LLM API key       : {'set' if cfg.has_llm_key else 'MISSING'}")
    import hashlib
    embedding_model_digest = "sha256:" + hashlib.sha256(
        cfg.embedding_model.encode("utf-8")
    ).hexdigest()
    print(f"  embedding model   : {embedding_model_digest}")
    print(f"  embedding backend : {cfg.embedding_backend}")
    print(f"  embedding base URL: {safe_embedding_base_url(cfg.embedding_base_url)}")
    print(f"  embedding API key : {'set' if cfg.has_embedding_key else 'not needed'}")
    print(f"  embedding fallback: {cfg.embedding_fallback_reason or 'none'}")
    print("─" * 60)

    results: list[_Result] = [_check_root(cfg), _check_llm(cfg), _check_sqlite_vec()]
    embedding_result, live_dim, live_model = _check_embedding(cfg)
    results.append(embedding_result)
    results.extend(_check_schema_and_dim(cfg, live_dim, live_model))
    results.append(_check_canonical_drift(cfg))

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
