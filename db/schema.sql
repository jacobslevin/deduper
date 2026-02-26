-- schema_mysql.sql
-- Compatible with MySQL 8.x

SET NAMES utf8mb4;

-- ---------- projects ----------
CREATE TABLE IF NOT EXISTS projects (
  id CHAR(36) NOT NULL,
  name TEXT NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  created_by TEXT NOT NULL,
  csv_filename TEXT NOT NULL,
  row_count INT NOT NULL,
  notes TEXT,
  removable_tokens JSON NOT NULL,
  matching_config JSON NOT NULL,

  PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------- brands ----------
CREATE TABLE IF NOT EXISTS brands (
  project_id CHAR(36) NOT NULL,
  brand_id VARCHAR(255) NOT NULL,
  brand_name TEXT NOT NULL,
  website_url TEXT,
  logo_url TEXT,
  product_count INT,
  category_raw TEXT,
  category_norm TEXT,
  name_norm TEXT NOT NULL,
  compare_norm TEXT NOT NULL,
  host_norm TEXT,
  domain_norm TEXT,
  url_norm TEXT,

  PRIMARY KEY (project_id, brand_id),
  CONSTRAINT fk_brands_project
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE INDEX idx_brands_project_domain  ON brands(project_id, domain_norm(255));
CREATE INDEX idx_brands_project_compare ON brands(project_id, compare_norm(255));

-- ---------- candidates ----------
CREATE TABLE IF NOT EXISTS candidates (
  id CHAR(36) NOT NULL,
  project_id CHAR(36) NOT NULL,
  brand_id_a VARCHAR(255) NOT NULL,
  brand_id_b VARCHAR(255) NOT NULL,
  score INT NOT NULL,

  -- JSON sin DEFAULT => trigger
  reasons JSON NOT NULL,

  status ENUM('pending', 'locked', 'approved', 'rejected', 'skipped') NOT NULL DEFAULT 'pending',
  locked_by TEXT,
  locked_at DATETIME(6),
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),

  UNIQUE KEY uk_candidates_pair (project_id, brand_id_a, brand_id_b),

  CONSTRAINT fk_candidates_project
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,

  CONSTRAINT fk_candidates_brand_a
    FOREIGN KEY (project_id, brand_id_a) REFERENCES brands(project_id, brand_id) ON DELETE CASCADE,

  CONSTRAINT fk_candidates_brand_b
    FOREIGN KEY (project_id, brand_id_b) REFERENCES brands(project_id, brand_id) ON DELETE CASCADE,

  PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE INDEX idx_candidates_project_status ON candidates(project_id, status);

CREATE INDEX idx_candidates_project_score  ON candidates(project_id, score);

-- ---------- decisions ----------
CREATE TABLE IF NOT EXISTS decisions (
  id CHAR(36) NOT NULL,
  candidate_id CHAR(36) NOT NULL,
  project_id CHAR(36) NOT NULL,
  decision ENUM('approved', 'rejected', 'skipped') NOT NULL,
  winner_brand_id VARCHAR(255),
  loser_brand_id VARCHAR(255),
  reviewer_name TEXT NOT NULL,
  decided_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  notes TEXT,
  winner_reason TEXT,
  updated_winner_brand_name TEXT,
  updated_winner_website_url TEXT,

  CONSTRAINT fk_decisions_candidate
    FOREIGN KEY (candidate_id) REFERENCES candidates(id) ON DELETE CASCADE,

  CONSTRAINT fk_decisions_project
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,

  CONSTRAINT fk_decisions_winner_brand
    FOREIGN KEY (project_id, winner_brand_id) REFERENCES brands(project_id, brand_id) ON DELETE RESTRICT,

  CONSTRAINT fk_decisions_loser_brand
    FOREIGN KEY (project_id, loser_brand_id) REFERENCES brands(project_id, brand_id) ON DELETE RESTRICT,

  PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE INDEX idx_decisions_project   ON decisions(project_id, decided_at);
CREATE INDEX idx_decisions_candidate ON decisions(candidate_id);


DELIMITER $$

CREATE TRIGGER projects_bi_defaults
BEFORE INSERT ON projects
FOR EACH ROW
BEGIN
  IF NEW.removable_tokens IS NULL THEN
    SET NEW.removable_tokens = JSON_ARRAY();
  END IF;

  IF NEW.matching_config IS NULL THEN
    SET NEW.matching_config = JSON_OBJECT(
      'min_score_to_show', 90,
      'allow_category_assisted_low_confidence', TRUE,
      'category_assisted_min_score', 85,
      'include_low_confidence_candidates', FALSE,
      'low_confidence_compare_threshold', 78
    );
  END IF;
END$$

CREATE TRIGGER candidates_bi_defaults
BEFORE INSERT ON candidates
FOR EACH ROW
BEGIN
  IF NEW.reasons IS NULL THEN
    SET NEW.reasons = JSON_ARRAY();
  END IF;
END$$

DELIMITER ;
