-- v13: FTS5 over raw message text. A direct keyword path to the `messages`
-- table, complementing chunks_fts (which only covers high-salience spans
-- materialized during dreaming). Indexed live at ingest via triggers so a turn
-- is searchable the moment it is logged — across sessions and before any dream
-- runs. Only user/assistant turns are indexed; tool/system turns are excluded
-- as noise. messages.id is INTEGER PRIMARY KEY (rowid alias) -> content_rowid='id'.
--
-- schema.sql creates the same objects for fresh DBs; this migration adds them to
-- existing (<= v12) DBs and backfills the turns already logged. The backfill is
-- an explicit role-filtered INSERT rather than the FTS 'rebuild' command, which
-- would ignore the role filter and index every turn. Version-gating makes the
-- backfill run exactly once (fresh DBs have no messages yet, so it is a no-op).
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    content='messages',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages
WHEN new.role IN ('user','assistant') BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages
WHEN old.role IN ('user','assistant') BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;

INSERT INTO messages_fts(rowid, content)
    SELECT id, content FROM messages WHERE role IN ('user','assistant');
