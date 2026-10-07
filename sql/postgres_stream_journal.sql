-- Gabby PostgreSQL SSE journal, schema version 1.
-- Apply with the host application's migration system before injecting the adapter.

CREATE TABLE IF NOT EXISTS gabby_stream_sessions (
    session_key TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    principal_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    event_count BIGINT NOT NULL DEFAULT 0 CHECK (event_count >= 0),
    total_bytes BIGINT NOT NULL DEFAULT 0 CHECK (total_bytes >= 0),
    max_response_bytes BIGINT NOT NULL CHECK (max_response_bytes > 0),
    finished_at TIMESTAMPTZ,
    CHECK (finished_at IS NULL OR finished_at >= created_at)
);

CREATE INDEX IF NOT EXISTS gabby_stream_sessions_cleanup
    ON gabby_stream_sessions(finished_at, created_at);

CREATE TABLE IF NOT EXISTS gabby_stream_events (
    session_key TEXT NOT NULL REFERENCES gabby_stream_sessions(session_key) ON DELETE CASCADE,
    event_id BIGINT NOT NULL CHECK (event_id > 0),
    frame BYTEA NOT NULL,
    PRIMARY KEY (session_key, event_id)
);
