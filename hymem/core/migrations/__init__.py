# Forward-only schema migrations, one ``NNN_description.sql`` file per version.
# The runner in hymem/core/db.py discovers these by leading integer, applies any
# whose version exceeds the DB's schema_version, and bumps schema_version to
# match. Files are idempotent (CREATE ... IF NOT EXISTS; ALTER ADD COLUMN errors
# tolerated) so they no-op safely against a fresh schema.sql database.
