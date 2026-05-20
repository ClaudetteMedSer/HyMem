-- v7: persistent token-overlap index for entity expansion in augment().
CREATE TABLE IF NOT EXISTS token_overlap_index (
    token TEXT NOT NULL,
    canonical TEXT NOT NULL,
    PRIMARY KEY (token, canonical)
);
CREATE INDEX IF NOT EXISTS idx_token_overlap_token ON token_overlap_index(token);
