-- Gabby PostgreSQL vector store, schema version 1.
-- Apply through the host application's migration system before constructing the adapter.
-- The migration role must be permitted to install pgvector when it is not already available.

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;

DO $$
DECLARE
    installed_version TEXT;
BEGIN
    SELECT extversion INTO installed_version
    FROM pg_extension
    WHERE extname = 'vector';

    IF installed_version IS NULL
       OR string_to_array(installed_version, '.')::INTEGER[] < ARRAY[0, 8, 0] THEN
        RAISE EXCEPTION 'Gabby requires pgvector 0.8.0 or newer for filtered HNSW scans';
    END IF;
END
$$;

CREATE TABLE IF NOT EXISTS gabby_vector_store_config (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    dimensions INTEGER NOT NULL CHECK (dimensions BETWEEN 1 AND 2000),
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE IF NOT EXISTS gabby_vector_documents (
    storage_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    text TEXT NOT NULL CHECK (length(text) > 0),
    source TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    generation TEXT,
    embedding vector NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CHECK (jsonb_typeof(metadata) = 'object'),
    CHECK (generation IS NULL OR length(btrim(generation)) > 0)
);

CREATE INDEX IF NOT EXISTS gabby_vector_documents_cosine
    ON gabby_vector_documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS gabby_vector_documents_source
    ON gabby_vector_documents (source, generation, document_id);

CREATE TABLE IF NOT EXISTS gabby_vector_source_fences (
    source TEXT PRIMARY KEY,
    fencing_token BIGINT NOT NULL CHECK (fencing_token >= 0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
