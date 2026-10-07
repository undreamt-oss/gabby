-- Gabby PostgreSQL generation manifest, schema version 1.
-- Apply with the host application's migration system before constructing the adapter.

CREATE TABLE IF NOT EXISTS gabby_index_sources (
    source TEXT PRIMARY KEY,
    fencing_token BIGINT NOT NULL DEFAULT 0 CHECK (fencing_token >= 0),
    active_generation TEXT,
    active_document_count BIGINT NOT NULL DEFAULT 0 CHECK (active_document_count >= 0),
    pending_generation TEXT,
    pending_document_count BIGINT CHECK (pending_document_count >= 0),
    pending_lexical_ready BOOLEAN NOT NULL DEFAULT FALSE,
    pending_vector_ready BOOLEAN NOT NULL DEFAULT FALSE,
    pending_created_at TIMESTAMPTZ,
    pending_owner_id TEXT,
    pending_lease_expires_at TIMESTAMPTZ,
    pending_error_type TEXT,
    CHECK (
        (pending_generation IS NULL
         AND pending_document_count IS NULL
         AND pending_lexical_ready = FALSE
         AND pending_vector_ready = FALSE
         AND pending_created_at IS NULL
         AND pending_owner_id IS NULL
         AND pending_lease_expires_at IS NULL
         AND pending_error_type IS NULL)
        OR
        (pending_generation IS NOT NULL
         AND pending_document_count IS NOT NULL
         AND pending_created_at IS NOT NULL
         AND pending_owner_id IS NOT NULL
         AND pending_lease_expires_at IS NOT NULL)
    ),
    CHECK (active_generation IS NOT NULL OR active_document_count = 0)
);

CREATE INDEX IF NOT EXISTS gabby_index_pending_sources
    ON gabby_index_sources(pending_created_at, source)
    WHERE pending_generation IS NOT NULL;

CREATE TABLE IF NOT EXISTS gabby_index_retired_generations (
    source TEXT NOT NULL REFERENCES gabby_index_sources(source),
    generation TEXT NOT NULL,
    retired_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (source, generation)
);

CREATE INDEX IF NOT EXISTS gabby_index_retired_cleanup
    ON gabby_index_retired_generations(source, retired_at, generation);
