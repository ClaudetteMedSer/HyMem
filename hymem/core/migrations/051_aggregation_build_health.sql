-- v51: fail-closed, restart-safe aggregation build health.
--
-- The singleton is marked pending before calling build_aggregation_nodes.
-- Only a clean applicable result clears it. It is deliberately bounded and
-- stores no source/prompt/exception/model/key/endpoint text.

ALTER TABLE dream_runs ADD COLUMN
    aggregation_build_exceptions INTEGER NOT NULL DEFAULT 0
    CHECK(aggregation_build_exceptions >= 0);

ALTER TABLE dream_runs ADD COLUMN
    aggregation_config_version TEXT CHECK (
        aggregation_config_version IS NULL OR (
            length(aggregation_config_version) = 92
            AND substr(aggregation_config_version, 1, 28) =
                'aggregation-build-config-v1:'
            AND substr(aggregation_config_version, 29) NOT GLOB '*[^0-9a-f]*'
        )
    );

CREATE TABLE IF NOT EXISTS aggregation_build_health (
    id INTEGER PRIMARY KEY CHECK(id = 1),
    last_success_config_version TEXT CHECK (
        last_success_config_version IS NULL OR (
            length(last_success_config_version) = 92
            AND substr(last_success_config_version, 1, 28) =
                'aggregation-build-config-v1:'
            AND substr(last_success_config_version, 29)
                NOT GLOB '*[^0-9a-f]*'
        )
    ),
    last_success_at TIMESTAMP,
    pending_config_version TEXT CHECK (
        pending_config_version IS NULL OR (
            length(pending_config_version) = 92
            AND substr(pending_config_version, 1, 28) =
                'aggregation-build-config-v1:'
            AND substr(pending_config_version, 29)
                NOT GLOB '*[^0-9a-f]*'
        )
    ),
    pending_attempts INTEGER NOT NULL DEFAULT 0
        CHECK(pending_attempts BETWEEN 0 AND 2147483647),
    pending_caught_exceptions INTEGER NOT NULL DEFAULT 0
        CHECK(pending_caught_exceptions BETWEEN 0 AND 2147483647),
    pending_fusion_failures INTEGER NOT NULL DEFAULT 0
        CHECK(pending_fusion_failures BETWEEN 0 AND 2147483647),
    first_pending_at TIMESTAMP,
    last_attempt_at TIMESTAMP,
    total_caught_exceptions INTEGER NOT NULL DEFAULT 0
        CHECK(total_caught_exceptions BETWEEN 0 AND 2147483647),
    total_fusion_failures INTEGER NOT NULL DEFAULT 0
        CHECK(total_fusion_failures BETWEEN 0 AND 2147483647),
    superseded_pending_configs INTEGER NOT NULL DEFAULT 0
        CHECK(superseded_pending_configs BETWEEN 0 AND 2147483647),
    last_failure_config_version TEXT CHECK (
        last_failure_config_version IS NULL OR (
            length(last_failure_config_version) = 92
            AND substr(last_failure_config_version, 1, 28) =
                'aggregation-build-config-v1:'
            AND substr(last_failure_config_version, 29)
                NOT GLOB '*[^0-9a-f]*'
        )
    ),
    last_failure_kind TEXT CHECK (
        last_failure_kind IS NULL OR last_failure_kind IN (
            'exception', 'fusion_failure', 'exception_and_fusion'
        )
    ),
    last_failure_at TIMESTAMP,
    CHECK (
        (last_success_config_version IS NULL AND last_success_at IS NULL)
        OR
        (last_success_config_version IS NOT NULL AND last_success_at IS NOT NULL)
    ),
    CHECK (
        (
            pending_config_version IS NULL
            AND pending_attempts = 0
            AND pending_caught_exceptions = 0
            AND pending_fusion_failures = 0
            AND first_pending_at IS NULL
            AND last_attempt_at IS NULL
        )
        OR
        (
            pending_config_version IS NOT NULL
            AND pending_attempts >= 1
            AND first_pending_at IS NOT NULL
            AND last_attempt_at IS NOT NULL
        )
    ),
    CHECK (
        (
            last_failure_config_version IS NULL
            AND last_failure_kind IS NULL
            AND last_failure_at IS NULL
        )
        OR
        (
            last_failure_config_version IS NOT NULL
            AND last_failure_kind IS NOT NULL
            AND last_failure_at IS NOT NULL
        )
    )
);
