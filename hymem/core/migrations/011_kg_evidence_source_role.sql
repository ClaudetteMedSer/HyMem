-- v11: speaker-weighted evidence. Records the role of the chunk's first
-- message (typically the preceding assistant turn, or the user when the chunk
-- opens a turn) so phase-1 / phase-3 can weight pos_evidence by author.
ALTER TABLE kg_evidence ADD COLUMN source_role TEXT;
