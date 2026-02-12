create extension if not exists pgcrypto;

do $$
begin
  if not exists (select 1 from pg_type where typname = 'candidate_status') then
    create type candidate_status as enum ('pending', 'locked', 'approved', 'rejected', 'skipped');
  end if;
end $$;

do $$
begin
  if not exists (select 1 from pg_type where typname = 'decision_type') then
    create type decision_type as enum ('approved', 'rejected', 'skipped');
  end if;
end $$;

create table if not exists projects (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  created_at timestamptz not null default now(),
  created_by text not null,
  csv_filename text not null,
  row_count integer not null,
  notes text,
  removable_tokens jsonb not null default '[]'::jsonb,
  matching_config jsonb not null default '{"min_score_to_show":90,"allow_category_assisted_low_confidence":true,"category_assisted_min_score":85,"include_low_confidence_candidates":false,"low_confidence_compare_threshold":78}'::jsonb
);

alter table projects
add column if not exists matching_config jsonb not null default '{"min_score_to_show":90,"allow_category_assisted_low_confidence":true,"category_assisted_min_score":85,"include_low_confidence_candidates":false,"low_confidence_compare_threshold":78}'::jsonb;

create table if not exists brands (
  project_id uuid not null references projects(id) on delete cascade,
  brand_id text not null,
  brand_name text not null,
  website_url text,
  logo_url text,
  product_count integer,
  category_raw text,
  category_norm text,
  name_norm text not null,
  compare_norm text not null,
  host_norm text,
  domain_norm text,
  url_norm text,
  primary key (project_id, brand_id)
);

create index if not exists idx_brands_project_domain on brands(project_id, domain_norm);
create index if not exists idx_brands_project_compare on brands(project_id, compare_norm);

alter table brands add column if not exists category_raw text;
alter table brands add column if not exists category_norm text;
alter table brands add column if not exists logo_url text;
drop index if exists idx_brands_project_category;

create table if not exists candidates (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references projects(id) on delete cascade,
  brand_id_a text not null,
  brand_id_b text not null,
  score integer not null,
  reasons jsonb not null default '[]'::jsonb,
  status candidate_status not null default 'pending',
  locked_by text,
  locked_at timestamptz,
  created_at timestamptz not null default now(),
  unique (project_id, brand_id_a, brand_id_b),
  foreign key (project_id, brand_id_a) references brands(project_id, brand_id) on delete cascade,
  foreign key (project_id, brand_id_b) references brands(project_id, brand_id) on delete cascade
);

create index if not exists idx_candidates_project_status on candidates(project_id, status);
create index if not exists idx_candidates_project_score on candidates(project_id, score desc);

create table if not exists decisions (
  id uuid primary key default gen_random_uuid(),
  candidate_id uuid not null references candidates(id) on delete cascade,
  project_id uuid not null references projects(id) on delete cascade,
  decision decision_type not null,
  winner_brand_id text,
  loser_brand_id text,
  reviewer_name text not null,
  decided_at timestamptz not null default now(),
  notes text,
  winner_reason text,
  updated_winner_brand_name text,
  updated_winner_website_url text,
  foreign key (project_id, winner_brand_id) references brands(project_id, brand_id) on delete set null,
  foreign key (project_id, loser_brand_id) references brands(project_id, brand_id) on delete set null
);

alter table decisions add column if not exists updated_winner_brand_name text;
alter table decisions add column if not exists updated_winner_website_url text;

create index if not exists idx_decisions_project on decisions(project_id, decided_at desc);
create index if not exists idx_decisions_candidate on decisions(candidate_id);
