-- v64: local ordered-input proof for exact claim replay. Historical rows stay
-- NULL; the proof is earned only by a fresh successful local publication.
ALTER TABLE kg_claim_extraction_outcomes ADD COLUMN local_replay_proof TEXT
CHECK (
    local_replay_proof IS NULL OR (
        length(local_replay_proof) = 71
        AND substr(local_replay_proof, 1, 7) = 'sha256:'
        AND substr(local_replay_proof, 8) NOT GLOB '*[^0-9a-f]*'
    )
);
