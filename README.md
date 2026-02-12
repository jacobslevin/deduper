# Brand Dedupe (Streamlit + Supabase)

Collaborative duplicate-brand review tool for non-technical users.

## What it does
- Upload a brands CSV and map columns (`brand_id`, `brand_name`, optional `website_url`, `product_count`)
- Normalize names/URLs and generate candidate duplicate pairs (bucketed to avoid O(n^2) globally)
- Let multiple reviewers approve/reject/skip in a shared queue with 10-minute optimistic locks
- Export:
  - `merge_instructions.csv` (`winner_brand_id`, `loser_brand_id`, plus metadata)
  - `decisions.csv` (all decisions)

## Stack
- Python 3.11+
- Streamlit
- Supabase Postgres
- RapidFuzz

## Local run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set database URL:
   - Environment variable: `DATABASE_URL`
   - or Streamlit secrets (`.streamlit/secrets.toml`):
     ```toml
     DATABASE_URL = "postgresql://..."
     ```
3. (Optional, recommended) set admin code:
   - Environment variable: `ADMIN_ACCESS_CODE`
   - or Streamlit secrets:
     ```toml
     ADMIN_ACCESS_CODE = "your-internal-admin-code"
     ```
4. Start app:
   ```bash
   streamlit run app.py
   ```
5. In sidebar, enable **Admin tools**, unlock with admin code, then click **Run DB setup (first time)**.

## Supabase setup
- Use the project's Postgres connection string as `DATABASE_URL`.
- Schema SQL is in `/Users/jacobslevin/Documents/New project/db/schema.sql`.

## Streamlit Community Cloud deploy
1. Push this repo to GitHub.
2. In Streamlit Community Cloud, create app from repo.
3. Set Python version 3.11+.
4. Add secret:
   - `DATABASE_URL = "postgresql://..."`
5. Deploy and run DB setup from sidebar once.

## Notes
- No auto-approve merges.
- Winner default is suggested and reviewer-overridable.
- Regenerating candidates in Project Settings clears existing decisions for that project.
