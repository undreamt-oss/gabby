-- Gabby PostgreSQL lexical knowledge store, schema version 1.
-- Apply through the host application's migration system before constructing the adapter.

CREATE TABLE IF NOT EXISTS gabby_knowledge_documents (
    storage_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    text TEXT NOT NULL CHECK (length(text) > 0),
    source TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    generation TEXT,
    search_vector TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CHECK (jsonb_typeof(metadata) = 'object'),
    CHECK (generation IS NULL OR length(btrim(generation)) > 0)
);

CREATE INDEX IF NOT EXISTS gabby_knowledge_documents_search
    ON gabby_knowledge_documents USING GIN (search_vector);

CREATE INDEX IF NOT EXISTS gabby_knowledge_documents_source
    ON gabby_knowledge_documents (source, generation, document_id);

CREATE TABLE IF NOT EXISTS gabby_knowledge_source_fences (
    source TEXT PRIMARY KEY,
    fencing_token BIGINT NOT NULL CHECK (fencing_token >= 0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
