-- MySQL schema (same structure as PostgreSQL; same credentials/host)

create table if not exists projects (
  id char(36) not null primary key,
  name text not null,
  created_at datetime(6) not null default (current_timestamp(6)),
  created_by text not null,
  csv_filename text not null,
  row_count integer not null,
  notes text,
  removable_tokens json not null default ('[]'),
  matching_config json not null default ('{"min_score_to_show":90,"allow_category_assisted_low_confidence":true,"category_assisted_min_score":85,"include_low_confidence_candidates":false,"low_confidence_compare_threshold":78}')
);

create table if not exists brands (
  project_id char(36) not null,
  brand_id varchar(255) not null,
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
  primary key (project_id, brand_id),
  foreign key (project_id) references projects(id) on delete cascade
);

create index if not exists idx_brands_project_domain on brands(project_id, domain_norm(255));
create index if not exists idx_brands_project_compare on brands(project_id, compare_norm(255));

create table if not exists candidates (
  id char(36) not null primary key,
  project_id char(36) not null,
  brand_id_a varchar(255) not null,
  brand_id_b varchar(255) not null,
  score integer not null,
  reasons json not null default ('[]'),
  status enum('pending', 'locked', 'approved', 'rejected', 'skipped') not null default 'pending',
  locked_by text,
  locked_at datetime(6),
  created_at datetime(6) not null default (current_timestamp(6)),
  unique key uk_candidates_pair (project_id, brand_id_a, brand_id_b),
  foreign key (project_id) references projects(id) on delete cascade,
  foreign key (project_id, brand_id_a) references brands(project_id, brand_id) on delete cascade,
  foreign key (project_id, brand_id_b) references brands(project_id, brand_id) on delete cascade
);

create index if not exists idx_candidates_project_status on candidates(project_id, status);
create index if not exists idx_candidates_project_score on candidates(project_id, score desc);

create table if not exists decisions (
  id char(36) not null primary key,
  candidate_id char(36) not null,
  project_id char(36) not null,
  decision enum('approved', 'rejected', 'skipped') not null,
  winner_brand_id varchar(255),
  loser_brand_id varchar(255),
  reviewer_name text not null,
  decided_at datetime(6) not null default (current_timestamp(6)),
  notes text,
  winner_reason text,
  updated_winner_brand_name text,
  updated_winner_website_url text,
  foreign key (candidate_id) references candidates(id) on delete cascade,
  foreign key (project_id) references projects(id) on delete cascade,
  foreign key (project_id, winner_brand_id) references brands(project_id, brand_id) on delete set null,
  foreign key (project_id, loser_brand_id) references brands(project_id, brand_id) on delete set null
);

create index if not exists idx_decisions_project on decisions(project_id, decided_at desc);
create index if not exists idx_decisions_candidate on decisions(candidate_id);
