-- AB_11: transcript summaries + chunk vector store
-- Run this entire file in Supabase SQL Editor before Option B.

create extension if not exists vector;

drop function if exists public.match_trace_eval_chunks(vector, integer);
drop table if exists public.trace_eval_chunks cascade;
drop table if exists public.trace_eval_summaries cascade;

create table public.trace_eval_summaries (
    run_id text primary key,
    model text,
    system_prompt_version text,
    user_input text,
    final_response text,
    user_request_intention text,
    animal_harm_level text,
    eco_issue boolean,
    model_refused boolean,
    safety_evaluation text,
    reason text,
    final_summary text,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table public.trace_eval_chunks (
    chunk_id text primary key,
    run_id text not null references public.trace_eval_summaries(run_id) on delete cascade,
    chunk_index integer not null,
    chunk_count integer not null,
    previous_chunk_id text,
    next_chunk_id text,
    chunk_text text not null,
    chunk_summary text,
    chunk_evaluation jsonb not null default '{}'::jsonb,
    embedding vector(1536),
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (run_id, chunk_index)
);

create index trace_eval_chunks_run_id_idx
    on public.trace_eval_chunks(run_id);

create index trace_eval_chunks_embedding_hnsw_idx
    on public.trace_eval_chunks
    using hnsw (embedding vector_cosine_ops);

create or replace function public.match_trace_eval_chunks(
    query_embedding vector(1536),
    match_count integer default 5
)
returns table (
    chunk_id text,
    run_id text,
    chunk_index integer,
    chunk_count integer,
    previous_chunk_id text,
    next_chunk_id text,
    chunk_text text,
    chunk_summary text,
    chunk_evaluation jsonb,
    metadata jsonb,
    similarity double precision
)
language sql
stable
as $$
    select
        c.chunk_id,
        c.run_id,
        c.chunk_index,
        c.chunk_count,
        c.previous_chunk_id,
        c.next_chunk_id,
        c.chunk_text,
        c.chunk_summary,
        c.chunk_evaluation,
        c.metadata,
        1 - (c.embedding <=> query_embedding) as similarity
    from public.trace_eval_chunks c
    where c.embedding is not null
    order by c.embedding <=> query_embedding
    limit greatest(match_count, 1);
$$;

-- Toy/demo permissions. Review and tighten before production.
grant usage on schema public to anon, authenticated;
grant select, insert, update, delete on public.trace_eval_summaries to anon, authenticated;
grant select, insert, update, delete on public.trace_eval_chunks to anon, authenticated;
grant execute on function public.match_trace_eval_chunks(vector, integer) to anon, authenticated;
