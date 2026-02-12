from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from io import StringIO
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import psycopg
from psycopg.rows import dict_row
from rapidfuzz import fuzz
import streamlit as st

try:
    import tldextract
except Exception:  # pragma: no cover - optional dependency behavior
    tldextract = None


DEFAULT_REMOVABLE = [
    "furniture",
    "kitchen",
    "bath",
    "bathrooms",
    "lighting",
    "interiors",
    "interior",
    "home",
    "design",
    "studio",
    "group",
    "collections",
    "collection",
    "international",
    "usa",
    "us",
    "america",
    "north",
    "south",
    "europe",
    "asia",
    "contract",
    "commercial",
    "hospitality",
    "residential",
    "kitchen and bath",
    "kitchen & bath",
    "kitchen bath",
]
DEFAULT_MATCHING_CONFIG = {
    "min_score_to_show": 90,
    "allow_category_assisted_low_confidence": True,
    "category_assisted_min_score": 85,
    "include_low_confidence_candidates": False,
    "low_confidence_compare_threshold": 78,
}
LEGAL_SUFFIXES = {
    "inc",
    "llc",
    "ltd",
    "corp",
    "company",
    "co",
    "incorporated",
    "limited",
}

COLUMN_ALIASES = {
    "brand_id": [
        "brand_id",
        "brand id",
        "id",
        "manufacturer id",
    ],
    "brand_name": [
        "brand_name",
        "brand name",
        "name",
        "manufacturer name",
    ],
    "website_url": [
        "website_url",
        "website url",
        "url",
        "brand url",
        "website",
        "domain",
    ],
    "logo_url": [
        "logo_url",
        "logo url",
        "logo",
        "brand_logo_url",
        "brand logo",
        "image url",
    ],
    "product_count": [
        "product_count",
        "product count",
        "# products",
        "products",
        "num products",
    ],
    "category": [
        "category",
        "categories",
        "dpcategorie",
        "dpcategories",
        "dp category",
        "dp categories",
        "product category",
    ],
}


def db_url() -> str:
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return env_url
    try:
        if "DATABASE_URL" in st.secrets:
            return st.secrets["DATABASE_URL"]
        if "database" in st.secrets and "url" in st.secrets["database"]:
            return st.secrets["database"]["url"]
    except Exception:
        # Streamlit raises if no secrets file exists.
        pass
    raise RuntimeError("DATABASE_URL not set. Configure Streamlit secrets or environment variable.")


def admin_access_code() -> str | None:
    env_code = os.getenv("ADMIN_ACCESS_CODE")
    if env_code:
        return env_code
    try:
        if "ADMIN_ACCESS_CODE" in st.secrets:
            return str(st.secrets["ADMIN_ACCESS_CODE"])
    except Exception:
        pass
    return None


def get_conn() -> psycopg.Connection:
    # Supabase poolers can error on server-side prepared statements across sessions.
    return psycopg.connect(db_url(), row_factory=dict_row, prepare_threshold=None)


def run_schema_setup() -> None:
    with open("db/schema.sql", "r", encoding="utf-8") as f:
        sql = f.read()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


def clean_text_for_norm(value: str) -> str:
    text = value.strip().lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_name(name: str) -> str:
    text = clean_text_for_norm(name)
    if not text:
        return ""
    tokens = text.split(" ")
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens).strip()


def split_removable(removable_items: list[str]) -> tuple[set[str], list[str]]:
    token_set: set[str] = set()
    phrase_list: list[str] = []
    for raw_item in removable_items:
        normalized = clean_text_for_norm(str(raw_item))
        if not normalized:
            continue
        if " " in normalized:
            phrase_list.append(normalized)
        else:
            token_set.add(normalized)
    phrase_list = sorted(set(phrase_list), key=lambda x: len(x), reverse=True)
    return token_set, phrase_list


def normalize_compare(name_norm: str, removable_items: list[str]) -> str:
    if not name_norm:
        return ""
    text = name_norm
    token_set, phrase_list = split_removable(removable_items)

    for phrase in phrase_list:
        text = re.sub(rf"\b{re.escape(phrase)}\b", " ", text)

    filtered_tokens = [tok for tok in text.split(" ") if tok and tok not in token_set]
    compare = re.sub(r"\s+", " ", " ".join(filtered_tokens)).strip()
    return compare if compare else name_norm


def best_effort_domain(host: str) -> str | None:
    if not host:
        return None
    if tldextract:
        ext = tldextract.extract(host)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}".lower()
        if ext.domain:
            return ext.domain.lower()
    parts = [p for p in host.split(".") if p]
    if len(parts) >= 2:
        return ".".join(parts[-2:]).lower()
    if len(parts) == 1:
        return parts[0].lower()
    return None


def normalize_url(url: str | None) -> tuple[str | None, str | None, str | None]:
    if url is None:
        return None, None, None
    raw = str(url).strip()
    if not raw:
        return None, None, None

    candidate = raw if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw) else f"https://{raw}"
    try:
        parsed = urlparse(candidate)
    except ValueError:
        return None, None, None

    host = (parsed.hostname or "").strip().lower()
    if not host or " " in host:
        return None, None, None
    if host.startswith("www."):
        host = host[4:]

    path = parsed.path or ""
    clean_path = path.rstrip("/")
    normalized_url = f"{parsed.scheme.lower() or 'https'}://{host}{clean_path}"
    domain = best_effort_domain(host)
    return host, domain, normalized_url


def parse_tokens_input(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        clean = line.strip()
        if clean:
            out.append(clean)
    return out


def parse_matching_config(raw_value: Any) -> dict[str, Any]:
    config = dict(DEFAULT_MATCHING_CONFIG)
    if isinstance(raw_value, str):
        try:
            raw_value = json.loads(raw_value)
        except Exception:
            raw_value = None
    if isinstance(raw_value, dict):
        min_score = to_int(raw_value.get("min_score_to_show"))
        cat_min = to_int(raw_value.get("category_assisted_min_score"))
        low_conf_threshold = to_int(raw_value.get("low_confidence_compare_threshold"))
        allow_cat = raw_value.get("allow_category_assisted_low_confidence")
        include_low = raw_value.get("include_low_confidence_candidates")
        if min_score is not None:
            config["min_score_to_show"] = max(0, min(100, int(min_score)))
        if cat_min is not None:
            config["category_assisted_min_score"] = max(0, min(100, int(cat_min)))
        if low_conf_threshold is not None:
            config["low_confidence_compare_threshold"] = max(50, min(100, int(low_conf_threshold)))
        if isinstance(allow_cat, bool):
            config["allow_category_assisted_low_confidence"] = allow_cat
        if isinstance(include_low, bool):
            config["include_low_confidence_candidates"] = include_low
    return config


def suggest_column_mapping(headers: list[str]) -> dict[str, str | None]:
    normalized_map = {h: clean_text_for_norm(h) for h in headers}

    def pick(field: str, used: set[str]) -> str | None:
        aliases = COLUMN_ALIASES[field]
        for alias in aliases:
            alias_norm = clean_text_for_norm(alias)
            for header, header_norm in normalized_map.items():
                if header in used:
                    continue
                if header_norm == alias_norm:
                    used.add(header)
                    return header
        for alias in aliases:
            alias_norm = clean_text_for_norm(alias)
            for header, header_norm in normalized_map.items():
                if header in used:
                    continue
                if alias_norm and alias_norm in header_norm:
                    used.add(header)
                    return header
        return None

    used_headers: set[str] = set()
    return {
        "brand_id": pick("brand_id", used_headers),
        "brand_name": pick("brand_name", used_headers),
        "website_url": pick("website_url", used_headers),
        "logo_url": pick("logo_url", used_headers),
        "product_count": pick("product_count", used_headers),
        "category": pick("category", used_headers),
    }


def to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return int(float(raw))
    except Exception:
        return None


def to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if pd.isna(value):
        return ""
    return str(value).strip()


def to_bool(value: Any) -> bool:
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return bool(value)


def _winner_key(candidate_id: str, brand_id: str) -> str:
    return f"group_winner_chk_{candidate_id}_{brand_id}"


def _loser_key(candidate_id: str, brand_id: str) -> str:
    return f"group_loser_chk_{candidate_id}_{brand_id}"


def handle_winner_checkbox_change(candidate_id: str, chosen_brand_id: str, all_brand_ids: list[str]) -> None:
    chosen_key = _winner_key(candidate_id, chosen_brand_id)
    chosen_now = to_bool(st.session_state.get(chosen_key, False))
    if chosen_now:
        for bid in all_brand_ids:
            st.session_state[_winner_key(candidate_id, bid)] = (bid == chosen_brand_id)
        st.session_state[_loser_key(candidate_id, chosen_brand_id)] = False
    else:
        # Require at least one winner selected in UI state.
        current_winners = [
            bid for bid in all_brand_ids if to_bool(st.session_state.get(_winner_key(candidate_id, bid), False))
        ]
        if not current_winners:
            st.session_state[chosen_key] = True


def handle_loser_checkbox_change(candidate_id: str, brand_id: str) -> None:
    if to_bool(st.session_state.get(_winner_key(candidate_id, brand_id), False)):
        st.session_state[_loser_key(candidate_id, brand_id)] = False


def normalize_category(value: Any) -> tuple[str | None, str | None]:
    raw = to_str(value)
    if not raw:
        return None, None
    tokens = []
    for part in re.split(r"[|,;/>\n]+", raw):
        clean = clean_text_for_norm(part)
        if clean:
            tokens.append(clean)
    if not tokens:
        return raw, None
    normalized = " | ".join(sorted(set(tokens)))
    # Avoid extremely large normalized category strings from long source lists.
    if len(normalized) > 512:
        normalized = normalized[:512].rstrip()
    return raw, normalized


def category_overlap_ratio(category_a: str | None, category_b: str | None) -> float:
    if not category_a or not category_b:
        return 0.0
    set_a = {x.strip() for x in str(category_a).split("|") if x.strip()}
    set_b = {x.strip() for x in str(category_b).split("|") if x.strip()}
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    if union == 0:
        return 0.0
    return inter / union


def build_brand_records(
    frame: pd.DataFrame,
    column_map: dict[str, str | None],
    removable_tokens: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        brand_id = to_str(row[column_map["brand_id"]])
        brand_name = to_str(row[column_map["brand_name"]])
        if not brand_id or not brand_name:
            continue

        website_url = None
        if column_map.get("website_url"):
            website_url = to_str(row[column_map["website_url"]]) or None
        logo_url = None
        if column_map.get("logo_url"):
            logo_url = to_str(row[column_map["logo_url"]]) or None

        product_count = None
        if column_map.get("product_count"):
            product_count = to_int(row[column_map["product_count"]])
        category_raw = None
        category_norm = None
        if column_map.get("category"):
            category_raw, category_norm = normalize_category(row[column_map["category"]])

        name_norm = normalize_name(brand_name)
        compare_norm = normalize_compare(name_norm, removable_tokens)
        host_norm, domain_norm, url_norm = normalize_url(website_url)

        records.append(
            {
                "brand_id": brand_id,
                "brand_name": brand_name,
                "website_url": website_url,
                "logo_url": logo_url,
                "product_count": product_count,
                "category_raw": category_raw,
                "category_norm": category_norm,
                "name_norm": name_norm,
                "compare_norm": compare_norm,
                "host_norm": host_norm,
                "domain_norm": domain_norm,
                "url_norm": url_norm,
            }
        )
    return records


def score_pair(
    a: dict[str, Any],
    b: dict[str, Any],
    compare_ratio: float,
    name_ratio: float,
    matching_config: dict[str, Any] | None = None,
) -> tuple[int, list[str]]:
    config = parse_matching_config(matching_config)
    category_assisted_min_score = int(config["category_assisted_min_score"])

    reasons: list[str] = []

    same_domain = bool(a["domain_norm"] and b["domain_norm"] and a["domain_norm"] == b["domain_norm"])
    compare_a = a["compare_norm"] or a["name_norm"]
    compare_b = b["compare_norm"] or b["name_norm"]
    exact_compare = bool(compare_a and compare_a == compare_b)
    missing_url_one = not a.get("website_url") or not b.get("website_url")
    category_overlap = category_overlap_ratio(a.get("category_norm"), b.get("category_norm"))

    if same_domain:
        reasons.append("same domain_norm")
    if exact_compare:
        reasons.append("exact compare_norm")
    if compare_ratio >= 92:
        reasons.append(f"rapidfuzz compare_norm ratio {int(round(compare_ratio))} >= 92")
    if name_ratio >= 94:
        reasons.append(f"rapidfuzz name_norm ratio {int(round(name_ratio))} >= 94")
    if category_overlap > 0:
        reasons.append(f"category overlap {int(round(category_overlap * 100))}%")

    if same_domain and compare_ratio >= 90:
        score = 99
        reasons.append("score rule: same domain_norm and compare_norm ratio >= 90")
    elif exact_compare and (same_domain or missing_url_one):
        score = 97
        reasons.append("score rule: exact compare_norm with same domain or missing URL")
    elif compare_ratio >= 92:
        score = 92
        reasons.append("score rule: compare_norm ratio >= 92")
    elif name_ratio >= 94:
        score = 94
        reasons.append("score rule: backup name_norm ratio >= 94")
    elif same_domain:
        score = 90
        reasons.append("score rule: same domain_norm")
    else:
        score = int(max(compare_ratio, name_ratio))

    if missing_url_one and category_overlap >= 0.6 and compare_ratio >= 88:
        score = max(score, category_assisted_min_score)
        reasons.append("score rule: category overlap supports missing-URL match")

    if missing_url_one:
        reasons.append("missing URL on one record")

    deduped = []
    seen = set()
    for reason in reasons:
        if reason not in seen:
            deduped.append(reason)
            seen.add(reason)
    return min(100, max(0, int(score))), deduped


def confidence_from_score(score: int) -> tuple[str, str]:
    if score >= 98:
        return "Very high", "Strong evidence these are the same brand."
    if score >= 95:
        return "High", "Likely the same brand; do a quick visual check."
    if score >= 92:
        return "Medium-high", "Good match signal, but verify carefully."
    if score >= 90:
        return "Medium", "Possible match; review details before deciding."
    if score >= 85:
        return "Careful review", "Borderline signal; validate details before merging."
    return "Low", "Weak signal; likely not a true duplicate."


def plain_reasons(raw_reasons: list[str] | None) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for reason in raw_reasons or []:
        msg = None
        if reason == "same domain_norm":
            msg = "Both records use the same website domain."
        elif reason == "exact compare_norm":
            msg = "After cleanup, both brand names are exactly the same."
        elif reason.startswith("rapidfuzz compare_norm ratio "):
            m = re.search(r"(\d+)", reason)
            if m:
                msg = f"The cleaned names are very similar ({m.group(1)}% match)."
            else:
                msg = "The cleaned names are very similar."
        elif reason.startswith("rapidfuzz name_norm ratio "):
            m = re.search(r"(\d+)", reason)
            if m:
                msg = f"The normalized names are very similar ({m.group(1)}% match)."
            else:
                msg = "The normalized names are very similar."
        elif reason.startswith("category overlap "):
            m = re.search(r"(\d+)", reason)
            if m:
                msg = f"The categories overlap ({m.group(1)}%)."
            else:
                msg = "The categories overlap."
        elif reason == "missing URL on one record":
            msg = "One record is missing a website URL."
        elif reason.startswith("score rule:"):
            continue

        if msg and msg not in seen:
            output.append(msg)
            seen.add(msg)
    return output


def winner_reason_text(rule: str, winner: dict[str, Any], loser: dict[str, Any]) -> str:
    if rule == "has logo_url":
        return (
            f"Suggested winner is {winner['brand_name']} (ID {winner['brand_id']}) "
            f"because it has a logo URL and the other record does not."
        )
    if rule == "higher product_count":
        return (
            f"Suggested winner is {winner['brand_name']} (ID {winner['brand_id']}) "
            f"because it has a higher product count."
        )
    if rule == "has website_url":
        return (
            f"Suggested winner is {winner['brand_name']} (ID {winner['brand_id']}) "
            f"because it has a website URL and the other record does not."
        )
    if rule == "longer brand_name":
        return (
            f"Suggested winner is {winner['brand_name']} (ID {winner['brand_id']}) "
            f"because the brand name is more descriptive (longer)."
        )
    return (
        f"Suggested winner is {winner['brand_name']} (ID {winner['brand_id']}) "
        f"as a tie-breaker using the lower brand_id."
    )


def save_decision_from_state(project_id: str, candidate_id: str, reviewer_name: str, winner_reason: str) -> None:
    action_label = st.session_state.get(f"decision_action_{candidate_id}", "Approve Merge")
    action_map = {
        "Approve Merge": "approved",
        "Reject": "rejected",
        "Skip": "skipped",
    }
    action = action_map.get(action_label, "approved")

    left_id = st.session_state.get(f"left_id_{candidate_id}")
    right_id = st.session_state.get(f"right_id_{candidate_id}")
    chosen_winner = st.session_state.get(f"winner_{candidate_id}")
    notes = st.session_state.get(f"notes_{candidate_id}")

    if action == "approved":
        if not chosen_winner or chosen_winner not in (left_id, right_id):
            st.session_state["global_flash"] = ("error", "Winner selection is invalid. Please reselect and try again.")
            st.session_state[f"queue_flash_{project_id}"] = ("error", "Winner selection is invalid. Please reselect and try again.")
            st.session_state["queue_should_rerun"] = True
            return
        loser = right_id if chosen_winner == left_id else left_id
        winner_value = chosen_winner
        loser_value = loser
    else:
        winner_value = None
        loser_value = None

    ok, msg = submit_decision(
        candidate_id=candidate_id,
        project_id=project_id,
        reviewer_name=reviewer_name,
        decision=action,
        winner_brand_id=winner_value,
        loser_brand_id=loser_value,
        notes=(notes or "").strip() or None,
        winner_reason=winner_reason,
    )
    if ok:
        success_msg = f"{action_label} saved."
        st.session_state[f"queue_flash_{project_id}"] = ("success", success_msg)
        st.session_state["global_flash"] = ("success", success_msg)
        st.session_state["last_action_result"] = ("success", success_msg)
    else:
        st.session_state[f"queue_flash_{project_id}"] = ("error", msg)
        st.session_state["global_flash"] = ("error", msg)
        st.session_state["last_action_result"] = ("error", msg)
    st.session_state["queue_should_rerun"] = True


def winner_default_for_group_rows(rows: list[dict[str, Any]]) -> tuple[str, str]:
    if not rows:
        return "", "No records available"

    with_logo = [r for r in rows if str(r.get("logo_url") or "").strip()]
    if with_logo and len(with_logo) != len(rows):
        if len(with_logo) == 1:
            r = with_logo[0]
            return str(r["brand_id"]), "has logo URL while others do not"
        rows = with_logo

    with_product = [r for r in rows if r.get("product_count") is not None]
    if with_product:
        top = max(r["product_count"] for r in with_product)
        top_rows = [r for r in with_product if r["product_count"] == top]
        if len(top_rows) == 1:
            r = top_rows[0]
            return str(r["brand_id"]), f"highest product_count ({top})"

    with_url = [r for r in rows if str(r.get("website_url") or "").strip()]
    if with_url and len(with_url) != len(rows):
        if len(with_url) == 1:
            r = with_url[0]
            return str(r["brand_id"]), "has website URL while others do not"
        rows = with_url

    max_len = max(len(str(r.get("brand_name") or "")) for r in rows)
    long_rows = [r for r in rows if len(str(r.get("brand_name") or "")) == max_len]
    if len(long_rows) == 1:
        r = long_rows[0]
        return str(r["brand_id"]), "most descriptive brand name"

    winner = min(str(r["brand_id"]) for r in long_rows)
    return winner, "tie-breaker by lower brand_id"


def submit_group_merge(
    project_id: str,
    reviewer_name: str,
    winner_brand_id: str,
    loser_brand_ids: list[str],
    notes: str | None,
    winner_reason: str,
    updated_winner_brand_name: str | None = None,
    updated_winner_website_url: str | None = None,
) -> tuple[bool, str]:
    losers = sorted({str(x) for x in loser_brand_ids if str(x).strip() and str(x) != str(winner_brand_id)})
    if not losers:
        return False, "Select at least one loser row to merge."

    unique_members = sorted({str(winner_brand_id), *losers})
    project_matching_config = fetch_project_matching_config(project_id)

    with get_conn() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select
                      brand_id, brand_name, website_url, product_count, name_norm, compare_norm, domain_norm
                    from brands
                    where project_id = %s and brand_id = any(%s)
                    """,
                    (project_id, unique_members),
                )
                brand_rows = {str(r["brand_id"]): r for r in cur.fetchall()}
                missing = [x for x in unique_members if x not in brand_rows]
                if missing:
                    conn.rollback()
                    return False, f"Missing brand records for: {', '.join(missing)}"

                merged_count = 0
                for loser_id in losers:
                    a, b = sorted([winner_brand_id, loser_id], key=lambda x: str(x))
                    cur.execute(
                        """
                        select id
                        from candidates
                        where project_id = %s and brand_id_a = %s and brand_id_b = %s
                        for update
                        """,
                        (project_id, a, b),
                    )
                    existing = cur.fetchone()
                    if existing:
                        candidate_id = str(existing["id"])
                        cur.execute(
                            """
                            update candidates
                            set status = 'approved',
                                locked_by = null,
                                locked_at = null
                            where id = %s
                            """,
                            (candidate_id,),
                        )
                    else:
                        brand_a = brand_rows[a]
                        brand_b = brand_rows[b]
                        compare_a = brand_a["compare_norm"] or brand_a["name_norm"]
                        compare_b = brand_b["compare_norm"] or brand_b["name_norm"]
                        name_a = brand_a["name_norm"]
                        name_b = brand_b["name_norm"]
                        compare_ratio = fuzz.ratio(compare_a, compare_b) if compare_a and compare_b else 0.0
                        name_ratio = fuzz.ratio(name_a, name_b) if name_a and name_b else 0.0
                        score, reasons = score_pair(brand_a, brand_b, compare_ratio, name_ratio, project_matching_config)
                        cur.execute(
                            """
                            insert into candidates(project_id, brand_id_a, brand_id_b, score, reasons, status)
                            values (%s, %s, %s, %s, %s, 'approved')
                            returning id
                            """,
                            (project_id, a, b, score, json.dumps(reasons)),
                        )
                        candidate_id = str(cur.fetchone()["id"])

                    cur.execute(
                        """
                        insert into decisions(
                          candidate_id, project_id, decision, winner_brand_id, loser_brand_id,
                          reviewer_name, notes, winner_reason, updated_winner_brand_name, updated_winner_website_url
                        )
                        values (%s, %s, 'approved', %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            candidate_id,
                            project_id,
                            winner_brand_id,
                            loser_id,
                            reviewer_name,
                            notes,
                            winner_reason,
                            updated_winner_brand_name,
                            updated_winner_website_url,
                        ),
                    )
                    merged_count += 1
            conn.commit()
            return True, f"Merged {merged_count} brands into winner {winner_brand_id}."
        except Exception as exc:
            conn.rollback()
            return False, f"Group merge failed: {exc}"


def save_group_merge_from_state(project_id: str, candidate_id: str, reviewer_name: str) -> None:
    winner_id = st.session_state.get(f"group_winner_{candidate_id}")
    loser_ids = st.session_state.get(f"group_loser_ids_{candidate_id}", [])
    notes = st.session_state.get(f"group_notes_{candidate_id}")
    winner_reason = st.session_state.get(f"group_winner_reason_{candidate_id}", "Selected as group winner by reviewer.")
    updated_name = st.session_state.get(f"group_updated_winner_name_{candidate_id}")
    updated_url = st.session_state.get(f"group_updated_winner_url_{candidate_id}")
    if not winner_id:
        msg = "Select exactly one winner before saving."
        st.session_state[f"queue_flash_{project_id}"] = ("error", msg)
        st.session_state["global_flash"] = ("error", msg)
        st.session_state["last_action_result"] = ("error", msg)
        st.session_state["queue_should_rerun"] = True
        return

    ok, msg = submit_group_merge(
        project_id=project_id,
        reviewer_name=reviewer_name,
        winner_brand_id=str(winner_id),
        loser_brand_ids=[str(x) for x in loser_ids],
        notes=(notes or "").strip() or None,
        winner_reason=winner_reason,
        updated_winner_brand_name=(updated_name or "").strip() or None,
        updated_winner_website_url=(updated_url or "").strip() or None,
    )
    if ok:
        st.session_state[f"queue_flash_{project_id}"] = ("success", msg)
        st.session_state["global_flash"] = ("success", msg)
        st.session_state["last_action_result"] = ("success", msg)
        st.session_state[f"active_candidate_id_{project_id}"] = None
    else:
        st.session_state[f"queue_flash_{project_id}"] = ("error", msg)
        st.session_state["global_flash"] = ("error", msg)
        st.session_state["last_action_result"] = ("error", msg)
    st.session_state["queue_should_rerun"] = True


def mark_cluster_no_dupes_from_state(project_id: str, candidate_id: str, reviewer_name: str) -> None:
    candidate_ids = [str(x) for x in st.session_state.get(f"group_candidate_ids_{candidate_id}", [])]
    if not candidate_ids:
        msg = "No cluster candidates available to mark."
        st.session_state[f"queue_flash_{project_id}"] = ("error", msg)
        st.session_state["global_flash"] = ("error", msg)
        st.session_state["last_action_result"] = ("error", msg)
        st.session_state["queue_should_rerun"] = True
        return

    notes = "No dupes in this cluster"
    with get_conn() as conn:
        try:
            changed = 0
            with conn.cursor() as cur:
                cur.execute(
                    """
                    update candidates
                    set status = 'rejected',
                        locked_by = null,
                        locked_at = null
                    where project_id = %s
                      and id = any(%s)
                      and status in ('pending', 'locked')
                    returning id
                    """,
                    (project_id, candidate_ids),
                )
                updated_ids = [str(r["id"]) for r in cur.fetchall()]
                changed = len(updated_ids)
                if updated_ids:
                    cur.executemany(
                        """
                        insert into decisions(
                          candidate_id, project_id, decision, winner_brand_id, loser_brand_id,
                          reviewer_name, notes, winner_reason
                        )
                        values (%s, %s, 'rejected', null, null, %s, %s, %s)
                        """,
                        [
                            (
                                cid,
                                project_id,
                                reviewer_name,
                                notes,
                                "Reviewer marked cluster as no dupes.",
                            )
                            for cid in updated_ids
                        ],
                    )
            conn.commit()
            msg = f"Marked {changed} candidate pairs as no dupes for this cluster."
            st.session_state[f"queue_flash_{project_id}"] = ("success", msg)
            st.session_state["global_flash"] = ("success", msg)
            st.session_state["last_action_result"] = ("success", msg)
            st.session_state[f"active_candidate_id_{project_id}"] = None
        except Exception as exc:
            conn.rollback()
            msg = f"Could not mark cluster as no dupes: {exc}"
            st.session_state[f"queue_flash_{project_id}"] = ("error", msg)
            st.session_state["global_flash"] = ("error", msg)
            st.session_state["last_action_result"] = ("error", msg)
    st.session_state["queue_should_rerun"] = True


def generate_candidates(records: list[dict[str, Any]], matching_config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    config = parse_matching_config(matching_config)
    min_score_to_show = int(config["min_score_to_show"])
    allow_category_assisted_low = bool(config["allow_category_assisted_low_confidence"])
    include_low_confidence = bool(config["include_low_confidence_candidates"])
    low_conf_threshold = int(config["low_confidence_compare_threshold"])

    by_id = {r["brand_id"]: r for r in records}

    buckets: dict[tuple[str, str], list[str]] = {}
    for rec in records:
        compare_key = rec["compare_norm"] or rec["name_norm"] or ""
        prefix = compare_key[:2]
        if rec["domain_norm"]:
            key = ("domain", rec["domain_norm"])
        else:
            key = ("prefix", prefix)
        buckets.setdefault(key, []).append(rec["brand_id"])

    candidates: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for bucket_kind, bucket_value in buckets:
        ids = sorted(set(buckets[(bucket_kind, bucket_value)]))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a = by_id[ids[i]]
                b = by_id[ids[j]]
                pair = (a["brand_id"], b["brand_id"])
                if pair in seen_pairs:
                    continue

                compare_a = a["compare_norm"] or a["name_norm"]
                compare_b = b["compare_norm"] or b["name_norm"]
                name_a = a["name_norm"]
                name_b = b["name_norm"]

                compare_ratio = fuzz.ratio(compare_a, compare_b) if compare_a and compare_b else 0.0
                name_ratio = fuzz.ratio(name_a, name_b) if name_a and name_b else 0.0

                same_domain = bool(a["domain_norm"] and b["domain_norm"] and a["domain_norm"] == b["domain_norm"])
                exact_compare = bool(compare_a and compare_a == compare_b)
                category_overlap = category_overlap_ratio(a.get("category_norm"), b.get("category_norm"))
                missing_url_one = not a.get("website_url") or not b.get("website_url")

                show = False
                if same_domain:
                    show = True
                elif exact_compare:
                    show = True
                elif bucket_kind == "prefix" and compare_ratio >= 92:
                    show = True
                elif name_ratio >= 94:
                    show = True
                elif allow_category_assisted_low and missing_url_one and category_overlap >= 0.6 and compare_ratio >= 88:
                    show = True
                elif include_low_confidence and bucket_kind == "prefix" and compare_ratio >= low_conf_threshold:
                    show = True

                if not show:
                    continue

                score, reasons = score_pair(a, b, compare_ratio, name_ratio, config)
                if score < min_score_to_show:
                    continue
                candidates.append(
                    {
                        "brand_id_a": a["brand_id"],
                        "brand_id_b": b["brand_id"],
                        "score": score,
                        "reasons": reasons,
                    }
                )
                seen_pairs.add(pair)
    candidates.sort(key=lambda x: (-x["score"], x["brand_id_a"], x["brand_id_b"]))
    return candidates


def winner_default(left: dict[str, Any], right: dict[str, Any]) -> tuple[str, str, str]:
    left_has_logo = bool(left.get("logo_url"))
    right_has_logo = bool(right.get("logo_url"))
    if left_has_logo != right_has_logo:
        if left_has_logo:
            return left["brand_id"], right["brand_id"], "has logo_url"
        return right["brand_id"], left["brand_id"], "has logo_url"

    lp = left.get("product_count")
    rp = right.get("product_count")

    if lp is not None and rp is not None and lp != rp:
        if lp > rp:
            return left["brand_id"], right["brand_id"], "higher product_count"
        return right["brand_id"], left["brand_id"], "higher product_count"

    left_has_url = bool(left.get("website_url"))
    right_has_url = bool(right.get("website_url"))
    if left_has_url != right_has_url:
        if left_has_url:
            return left["brand_id"], right["brand_id"], "has website_url"
        return right["brand_id"], left["brand_id"], "has website_url"

    if len(left.get("brand_name", "")) != len(right.get("brand_name", "")):
        if len(left.get("brand_name", "")) > len(right.get("brand_name", "")):
            return left["brand_id"], right["brand_id"], "longer brand_name"
        return right["brand_id"], left["brand_id"], "longer brand_name"

    left_id = str(left["brand_id"])
    right_id = str(right["brand_id"])
    if left_id <= right_id:
        return left["brand_id"], right["brand_id"], "lower brand_id"
    return right["brand_id"], left["brand_id"], "lower brand_id"


def insert_project_and_data(
    reviewer: str,
    project_name: str,
    filename: str,
    records: list[dict[str, Any]],
    removable_tokens: list[str],
    matching_config: dict[str, Any],
    notes: str | None,
) -> tuple[str, int]:
    matching_config = parse_matching_config(matching_config)
    candidates = generate_candidates(records, matching_config)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into projects(name, created_by, csv_filename, row_count, notes, removable_tokens, matching_config)
                values (%s, %s, %s, %s, %s, %s, %s)
                returning id
                """,
                (
                    project_name,
                    reviewer,
                    filename,
                    len(records),
                    notes,
                    json.dumps(removable_tokens),
                    json.dumps(matching_config),
                ),
            )
            project_id = str(cur.fetchone()["id"])

            brand_rows = [
                (
                    project_id,
                    r["brand_id"],
                    r["brand_name"],
                    r["website_url"],
                    r["logo_url"],
                    r["product_count"],
                    r["category_raw"],
                    r["category_norm"],
                    r["name_norm"],
                    r["compare_norm"],
                    r["host_norm"],
                    r["domain_norm"],
                    r["url_norm"],
                )
                for r in records
            ]
            cur.executemany(
                """
                insert into brands(
                    project_id, brand_id, brand_name, website_url, logo_url, product_count,
                    category_raw, category_norm, name_norm, compare_norm, host_norm, domain_norm, url_norm
                )
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                on conflict (project_id, brand_id) do update
                set
                    brand_name = excluded.brand_name,
                    website_url = excluded.website_url,
                    logo_url = excluded.logo_url,
                    product_count = excluded.product_count,
                    category_raw = excluded.category_raw,
                    category_norm = excluded.category_norm,
                    name_norm = excluded.name_norm,
                    compare_norm = excluded.compare_norm,
                    host_norm = excluded.host_norm,
                    domain_norm = excluded.domain_norm,
                    url_norm = excluded.url_norm
                """,
                brand_rows,
            )

            candidate_rows = [
                (project_id, c["brand_id_a"], c["brand_id_b"], c["score"], json.dumps(c["reasons"])) for c in candidates
            ]
            if candidate_rows:
                cur.executemany(
                    """
                    insert into candidates(project_id, brand_id_a, brand_id_b, score, reasons)
                    values (%s, %s, %s, %s, %s)
                    on conflict (project_id, brand_id_a, brand_id_b) do update
                    set score = excluded.score,
                        reasons = excluded.reasons,
                        status = 'pending',
                        locked_by = null,
                        locked_at = null
                    """,
                    candidate_rows,
                )
        conn.commit()
    return project_id, len(candidates)


def fetch_projects() -> list[dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select
                    p.*,
                    count(*) filter (where c.status = 'pending') as pending_count,
                    count(*) filter (where c.status = 'locked') as locked_count,
                    count(*) filter (where c.status = 'approved') as approved_count,
                    count(*) filter (where c.status = 'rejected') as rejected_count,
                    count(*) filter (where c.status = 'skipped') as skipped_count
                from projects p
                left join candidates c on c.project_id = p.id
                group by p.id
                order by p.created_at desc
                """
            )
            return list(cur.fetchall())


def fetch_project(project_id: str) -> dict[str, Any] | None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("select * from projects where id = %s", (project_id,))
            row = cur.fetchone()
            return row if row else None


def fetch_project_matching_config(project_id: str) -> dict[str, Any]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("select matching_config from projects where id = %s", (project_id,))
            row = cur.fetchone()
    if not row:
        return dict(DEFAULT_MATCHING_CONFIG)
    return parse_matching_config(row.get("matching_config"))


def regenerate_project_candidates(project_id: str, removable_tokens: list[str], matching_config: dict[str, Any]) -> int:
    matching_config = parse_matching_config(matching_config)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select brand_id, brand_name, website_url, logo_url, product_count, category_raw, category_norm
                from brands
                where project_id = %s
                """,
                (project_id,),
            )
            source_rows = list(cur.fetchall())

            records = []
            for row in source_rows:
                name_norm = normalize_name(row["brand_name"])
                compare_norm = normalize_compare(name_norm, removable_tokens)
                host_norm, domain_norm, url_norm = normalize_url(row["website_url"])
                records.append(
                    {
                        "brand_id": row["brand_id"],
                        "brand_name": row["brand_name"],
                        "website_url": row["website_url"],
                        "logo_url": row["logo_url"],
                        "product_count": row["product_count"],
                        "category_raw": row["category_raw"],
                        "category_norm": row["category_norm"],
                        "name_norm": name_norm,
                        "compare_norm": compare_norm,
                        "host_norm": host_norm,
                        "domain_norm": domain_norm,
                        "url_norm": url_norm,
                    }
                )

            cur.execute("delete from decisions where project_id = %s", (project_id,))
            cur.execute("delete from candidates where project_id = %s", (project_id,))

            cur.executemany(
                """
                update brands
                set name_norm = %s,
                    compare_norm = %s,
                    logo_url = %s,
                    category_raw = %s,
                    category_norm = %s,
                    host_norm = %s,
                    domain_norm = %s,
                    url_norm = %s
                where project_id = %s and brand_id = %s
                """,
                [
                    (
                        r["name_norm"],
                        r["compare_norm"],
                        r["logo_url"],
                        r["category_raw"],
                        r["category_norm"],
                        r["host_norm"],
                        r["domain_norm"],
                        r["url_norm"],
                        project_id,
                        r["brand_id"],
                    )
                    for r in records
                ],
            )

            candidates = generate_candidates(records, matching_config)
            rows = [
                (project_id, c["brand_id_a"], c["brand_id_b"], c["score"], json.dumps(c["reasons"])) for c in candidates
            ]
            if rows:
                cur.executemany(
                    """
                    insert into candidates(project_id, brand_id_a, brand_id_b, score, reasons)
                    values (%s, %s, %s, %s, %s)
                    """,
                    rows,
                )

            cur.execute(
                "update projects set removable_tokens = %s, matching_config = %s where id = %s",
                (json.dumps(removable_tokens), json.dumps(matching_config), project_id),
            )
        conn.commit()
    return len(rows)


def queue_query(
    project_id: str,
    reviewer_name: str,
    filters: dict[str, Any],
    limit: int,
    score_order: str = "desc",
) -> list[dict[str, Any]]:
    order_dir = "asc" if str(score_order).lower() == "asc" else "desc"
    clauses = ["c.project_id = %(project_id)s"]
    params: dict[str, Any] = {
        "project_id": project_id,
        "reviewer_name": reviewer_name,
        "limit": limit,
        "score_min": filters["score_min"],
        "search": f"%{filters['search']}%" if filters["search"] else None,
    }

    # pending or lock expired or lock owned by current reviewer
    clauses.append(
        """
        (
          c.status = 'pending'
          or (c.status = 'locked' and c.locked_at < now() - interval '10 minutes')
          or (c.status = 'locked' and c.locked_by = %(reviewer_name)s)
        )
        """
    )

    clauses.append("c.score >= %(score_min)s")

    if filters["same_domain_only"]:
        clauses.append("ba.domain_norm is not null and ba.domain_norm = bb.domain_norm")
    if filters["missing_url_only"]:
        clauses.append("(ba.website_url is null or bb.website_url is null)")
    if filters["product_count_diff"]:
        clauses.append("ba.product_count is not null and bb.product_count is not null and ba.product_count <> bb.product_count")
    if filters["search"]:
        clauses.append(
            """
            (
              ba.brand_name ilike %(search)s
              or bb.brand_name ilike %(search)s
              or ba.brand_id ilike %(search)s
              or bb.brand_id ilike %(search)s
            )
            """
        )

    where_sql = " and ".join(clauses)

    query = f"""
        select
          c.id,
          c.brand_id_a,
          c.brand_id_b,
          c.score,
          c.reasons,
          c.status,
          c.locked_by,
          c.locked_at,
          ba.brand_name as brand_name_a,
          ba.website_url as website_url_a,
          ba.product_count as product_count_a,
          ba.category_norm as category_norm_a,
          ba.name_norm as name_norm_a,
          ba.compare_norm as compare_norm_a,
          ba.domain_norm as domain_norm_a,
          bb.brand_name as brand_name_b,
          bb.website_url as website_url_b,
          bb.product_count as product_count_b,
          bb.category_norm as category_norm_b,
          bb.name_norm as name_norm_b,
          bb.compare_norm as compare_norm_b,
          bb.domain_norm as domain_norm_b
        from candidates c
        join brands ba on ba.project_id = c.project_id and ba.brand_id = c.brand_id_a
        join brands bb on bb.project_id = c.project_id and bb.brand_id = c.brand_id_b
        where {where_sql}
        order by c.score {order_dir}, c.created_at asc
        limit %(limit)s
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return list(cur.fetchall())


def fetch_candidate_by_id(project_id: str, candidate_id: str) -> dict[str, Any] | None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select
                  c.id,
                  c.brand_id_a,
                  c.brand_id_b,
                  c.score,
                  c.reasons,
                  c.status,
                  c.locked_by,
                  c.locked_at,
                  ba.brand_name as brand_name_a,
                  ba.website_url as website_url_a,
                  ba.product_count as product_count_a,
                  ba.category_norm as category_norm_a,
                  ba.name_norm as name_norm_a,
                  ba.compare_norm as compare_norm_a,
                  ba.domain_norm as domain_norm_a,
                  bb.brand_name as brand_name_b,
                  bb.website_url as website_url_b,
                  bb.product_count as product_count_b,
                  bb.category_norm as category_norm_b,
                  bb.name_norm as name_norm_b,
                  bb.compare_norm as compare_norm_b,
                  bb.domain_norm as domain_norm_b
                from candidates c
                join brands ba on ba.project_id = c.project_id and ba.brand_id = c.brand_id_a
                join brands bb on bb.project_id = c.project_id and bb.brand_id = c.brand_id_b
                where c.project_id = %s
                  and c.id = %s
                limit 1
                """,
                (project_id, candidate_id),
            )
            return cur.fetchone()


def fetch_duplicate_group_context(project_id: str, seed_brand_a: str, seed_brand_b: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                with recursive component as (
                  select %s::text as brand_id
                  union
                  select %s::text as brand_id
                  union
                  select
                    case
                      when c.brand_id = x.brand_id_a then x.brand_id_b
                      else x.brand_id_a
                    end as brand_id
                  from component c
                  join candidates x
                    on x.project_id = %s
                   and x.status in ('pending', 'locked', 'approved', 'skipped')
                   and (x.brand_id_a = c.brand_id or x.brand_id_b = c.brand_id)
                )
                select distinct brand_id
                from component
                """,
                (seed_brand_a, seed_brand_b, project_id),
            )
            brand_ids = [r["brand_id"] for r in cur.fetchall()]
            if not brand_ids:
                return (
                    pd.DataFrame(
                        columns=[
                            "brand_id",
                            "brand_name",
                            "website_url",
                            "logo_url",
                            "product_count",
                            "category_norm",
                            "domain_norm",
                        ]
                    ),
                    pd.DataFrame(columns=["candidate_id", "brand_id_a", "brand_name_a", "brand_id_b", "brand_name_b", "status", "score", "reasons"]),
                )

            cur.execute(
                """
                select brand_id, brand_name, website_url, logo_url, product_count, category_norm, domain_norm
                from brands
                where project_id = %s
                  and brand_id = any(%s)
                order by brand_name asc, brand_id asc
                """,
                (project_id, brand_ids),
            )
            members = list(cur.fetchall())

            cur.execute(
                """
                select
                  c.id as candidate_id,
                  c.brand_id_a,
                  ba.brand_name as brand_name_a,
                  c.brand_id_b,
                  bb.brand_name as brand_name_b,
                  c.status,
                  c.score,
                  c.reasons
                from candidates c
                join brands ba on ba.project_id = c.project_id and ba.brand_id = c.brand_id_a
                join brands bb on bb.project_id = c.project_id and bb.brand_id = c.brand_id_b
                where c.project_id = %s
                  and c.brand_id_a = any(%s)
                  and c.brand_id_b = any(%s)
                order by c.score desc, c.created_at asc
                """,
                (project_id, brand_ids, brand_ids),
            )
            pair_rows = list(cur.fetchall())

    members_df = pd.DataFrame(members)
    pairs_pretty = []
    for row in pair_rows:
        pairs_pretty.append(
            {
                "candidate_id": row["candidate_id"],
                "brand_id_a": row["brand_id_a"],
                "brand_name_a": row["brand_name_a"],
                "brand_id_b": row["brand_id_b"],
                "brand_name_b": row["brand_name_b"],
                "status": row["status"],
                "score": row["score"],
                "reasons": "; ".join(plain_reasons(row["reasons"])),
            }
        )
    pairs_df = pd.DataFrame(pairs_pretty)
    return members_df, pairs_df


def fetch_cluster_progress(project_id: str) -> tuple[int, int, float]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select id, brand_id_a, brand_id_b, status
                from candidates
                where project_id = %s
                """,
                (project_id,),
            )
            rows = list(cur.fetchall())

    if not rows:
        return 0, 0, 0.0

    parent: dict[str, str] = {}
    size: dict[str, int] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        sa = size.get(ra, 1)
        sb = size.get(rb, 1)
        if sa < sb:
            ra, rb = rb, ra
            sa, sb = sb, sa
        parent[rb] = ra
        size[ra] = sa + sb

    for row in rows:
        a = str(row["brand_id_a"])
        b = str(row["brand_id_b"])
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        size.setdefault(a, 1)
        size.setdefault(b, 1)
        union(a, b)

    cluster_statuses: dict[str, list[str]] = {}
    for row in rows:
        root = find(str(row["brand_id_a"]))
        cluster_statuses.setdefault(root, []).append(str(row["status"]))

    total_clusters = len(cluster_statuses)
    completed_clusters = 0
    for statuses in cluster_statuses.values():
        if all(s not in ("pending", "locked") for s in statuses):
            completed_clusters += 1
    pct = (completed_clusters / total_clusters * 100.0) if total_clusters else 0.0
    return completed_clusters, total_clusters, pct


def fetch_cluster_size(project_id: str, seed_brand_a: str, seed_brand_b: str) -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                with recursive component as (
                  select %s::text as brand_id
                  union
                  select %s::text as brand_id
                  union
                  select
                    case
                      when c.brand_id = x.brand_id_a then x.brand_id_b
                      else x.brand_id_a
                    end as brand_id
                  from component c
                  join candidates x
                    on x.project_id = %s
                   and x.status in ('pending', 'locked', 'approved', 'skipped')
                   and (x.brand_id_a = c.brand_id or x.brand_id_b = c.brand_id)
                )
                select count(distinct brand_id) as cluster_size
                from component
                """,
                (seed_brand_a, seed_brand_b, project_id),
            )
            row = cur.fetchone()
            return int(row["cluster_size"]) if row and row["cluster_size"] is not None else 2


def cluster_sizes_from_queue_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    parent: dict[str, str] = {}
    size: dict[str, int] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        sa = size.get(ra, 1)
        sb = size.get(rb, 1)
        if sa < sb:
            ra, rb = rb, ra
            sa, sb = sb, sa
        parent[rb] = ra
        size[ra] = sa + sb

    for row in rows:
        a = str(row["brand_id_a"])
        b = str(row["brand_id_b"])
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        size.setdefault(a, 1)
        size.setdefault(b, 1)
        union(a, b)

    out: dict[str, int] = {}
    for row in rows:
        root = find(str(row["brand_id_a"]))
        out[str(row["id"])] = size.get(root, 1)
    return out


def lock_candidate(project_id: str, candidate_id: str, reviewer_name: str) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                update candidates
                set status = 'locked', locked_by = %s, locked_at = now()
                where id = %s
                  and project_id = %s
                  and (
                    status = 'pending'
                    or (status = 'locked' and locked_at < now() - interval '10 minutes')
                    or (status = 'locked' and locked_by = %s)
                  )
                """,
                (reviewer_name, candidate_id, project_id, reviewer_name),
            )
            locked = cur.rowcount == 1
        conn.commit()
    return locked


def release_reviewer_locks(project_id: str, reviewer_name: str) -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                update candidates
                set status = 'pending',
                    locked_by = null,
                    locked_at = null
                where project_id = %s
                  and status = 'locked'
                  and locked_by = %s
                """,
                (project_id, reviewer_name),
            )
            released = cur.rowcount
        conn.commit()
    return released


def submit_decision(
    candidate_id: str,
    project_id: str,
    reviewer_name: str,
    decision: str,
    winner_brand_id: str | None,
    loser_brand_id: str | None,
    notes: str | None,
    winner_reason: str | None,
) -> tuple[bool, str]:
    with get_conn() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select status, locked_by
                    from candidates
                    where id = %s and project_id = %s
                    for update
                    """,
                    (candidate_id, project_id),
                )
                current = cur.fetchone()
                if not current:
                    conn.rollback()
                    return False, "Candidate not found."
                if current["status"] not in ("pending", "locked"):
                    conn.rollback()
                    return False, "Candidate was already decided. Refresh the queue."

                cur.execute(
                    """
                    update candidates
                    set status = %s,
                        locked_by = null,
                        locked_at = null
                    where id = %s
                      and project_id = %s
                    """,
                    (
                        decision,
                        candidate_id,
                        project_id,
                    ),
                )
                if cur.rowcount != 1:
                    conn.rollback()

                    return False, "Could not update candidate status."

                cur.execute(
                    """
                    insert into decisions(
                        candidate_id, project_id, decision, winner_brand_id, loser_brand_id,
                        reviewer_name, notes, winner_reason
                    )
                    values (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        candidate_id,
                        project_id,
                        decision,
                        winner_brand_id,
                        loser_brand_id,
                        reviewer_name,
                        notes,
                        winner_reason,
                    ),
                )
            conn.commit()
            return True, "Decision saved."
        except Exception as exc:
            conn.rollback()
            return False, f"Save failed: {exc}"


def fetch_export_merge(project_id: str) -> pd.DataFrame:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select
                    d.winner_brand_id,
                    d.loser_brand_id,
                    d.updated_winner_brand_name,
                    d.updated_winner_website_url,
                    c.score,
                    c.reasons,
                    d.reviewer_name,
                    d.decided_at
                from decisions d
                join candidates c on c.id = d.candidate_id
                where d.project_id = %s and d.decision = 'approved'
                order by d.decided_at asc
                """,
                (project_id,),
            )
            rows = list(cur.fetchall())
    data = []
    for r in rows:
        data.append(
            {
                "winner_brand_id": r["winner_brand_id"],
                "loser_brand_id": r["loser_brand_id"],
                "updated_winner_brand_name": r["updated_winner_brand_name"],
                "updated_winner_website_url": r["updated_winner_website_url"],
                "score": r["score"],
                "reasons": "; ".join(r["reasons"] or []),
                "reviewer_name": r["reviewer_name"],
                "decided_at": r["decided_at"],
            }
        )
    if not data:
        return pd.DataFrame(
            columns=[
                "winner_brand_id",
                "loser_brand_id",
                "updated_winner_brand_name",
                "updated_winner_website_url",
                "score",
                "reasons",
                "reviewer_name",
                "decided_at",
            ]
        )
    return pd.DataFrame(data)


def fetch_export_decisions(project_id: str) -> pd.DataFrame:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select
                  d.decision,
                  d.winner_brand_id,
                  d.loser_brand_id,
                  d.updated_winner_brand_name,
                  d.updated_winner_website_url,
                  d.winner_reason,
                  d.reviewer_name,
                  d.decided_at,
                  d.notes,
                  c.id as candidate_id,
                  c.brand_id_a,
                  c.brand_id_b,
                  c.score,
                  c.reasons
                from decisions d
                join candidates c on c.id = d.candidate_id
                where d.project_id = %s
                order by d.decided_at asc
                """,
                (project_id,),
            )
            rows = list(cur.fetchall())
    data = []
    for r in rows:
        data.append(
            {
                "decision": r["decision"],
                "candidate_id": r["candidate_id"],
                "brand_id_a": r["brand_id_a"],
                "brand_id_b": r["brand_id_b"],
                "winner_brand_id": r["winner_brand_id"],
                "loser_brand_id": r["loser_brand_id"],
                "updated_winner_brand_name": r["updated_winner_brand_name"],
                "updated_winner_website_url": r["updated_winner_website_url"],
                "score": r["score"],
                "reasons": "; ".join(r["reasons"] or []),
                "winner_reason": r["winner_reason"],
                "reviewer_name": r["reviewer_name"],
                "decided_at": r["decided_at"],
                "notes": r["notes"],
            }
        )
    if not data:
        return pd.DataFrame(
            columns=[
                "decision",
                "candidate_id",
                "brand_id_a",
                "brand_id_b",
                "winner_brand_id",
                "loser_brand_id",
                "updated_winner_brand_name",
                "updated_winner_website_url",
                "score",
                "reasons",
                "winner_reason",
                "reviewer_name",
                "decided_at",
                "notes",
            ]
        )
    return pd.DataFrame(data)


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def fetch_diagnostics(project_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select status, count(*) as count
                from candidates
                where project_id = %s
                group by status
                order by status
                """,
                (project_id,),
            )
            status_rows = list(cur.fetchall())

            cur.execute(
                """
                select decision, count(*) as count
                from decisions
                where project_id = %s
                group by decision
                order by decision
                """,
                (project_id,),
            )
            decision_counts = list(cur.fetchall())

            cur.execute(
                """
                select
                    d.decided_at,
                    d.reviewer_name,
                    d.decision,
                    d.winner_brand_id,
                    d.loser_brand_id,
                    d.candidate_id
                from decisions d
                where d.project_id = %s
                order by d.decided_at desc
                limit 20
                """,
                (project_id,),
            )
            recent = list(cur.fetchall())

    status_df = pd.DataFrame(status_rows)
    decisions_df = pd.DataFrame(decision_counts)
    recent_df = pd.DataFrame(recent)
    if decisions_df.empty:
        decisions_df = pd.DataFrame(columns=["decision", "count"])
    if status_df.empty:
        status_df = pd.DataFrame(columns=["status", "count"])
    if recent_df.empty:
        recent_df = pd.DataFrame(
            columns=["decided_at", "reviewer_name", "decision", "winner_brand_id", "loser_brand_id", "candidate_id"]
        )
    return status_df, decisions_df, recent_df


def render_diagnostics(project: dict[str, Any]) -> None:
    st.header("Diagnostics")
    st.caption("Use this page to confirm whether decisions are being saved in the database.")
    st.write(f"Project: `{project['name']}`")
    st.write(f"Project ID: `{project['id']}`")

    try:
        status_df, decision_counts_df, recent_df = fetch_diagnostics(str(project["id"]))
    except Exception as exc:
        st.error(f"Diagnostics query failed: {exc}")
        return

    st.subheader("Candidate Status Counts")
    if status_df.empty:
        st.write("No candidates found.")
    else:
        st.dataframe(status_df, use_container_width=True)

    st.subheader("Decision Counts")
    if decision_counts_df.empty:
        st.write("No decisions saved yet.")
    else:
        st.dataframe(decision_counts_df, use_container_width=True)

    st.subheader("Recent Saved Decisions (last 20)")
    if recent_df.empty:
        st.write("No decision rows in database.")
    else:
        st.dataframe(recent_df, use_container_width=True)

    st.subheader("Last Action Result (this browser session)")
    last_action = st.session_state.get("last_action_result")
    if not last_action:
        st.write("No action feedback captured yet.")
    else:
        level, msg = last_action
        if level == "success":
            st.success(msg)
        else:
            st.error(msg)


def render_upload_page(reviewer_name: str) -> None:
    st.header("Upload CSV")
    st.caption("Create a project, map columns, and generate candidate duplicate pairs.")

    uploaded = st.file_uploader("Brands CSV", type=["csv"])
    project_name = st.text_input("Project name")
    notes = st.text_area("Notes (optional)")

    if not uploaded:
        return

    try:
        raw = uploaded.getvalue().decode("utf-8")
        df = pd.read_csv(StringIO(raw))
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    headers = list(df.columns)
    options = ["<none>"] + headers
    suggested = suggest_column_mapping(headers)

    def idx_header(header: str | None, choices: list[str]) -> int:
        if header and header in choices:
            return choices.index(header)
        return 0

    st.subheader("Column Mapping")
    c1, c2 = st.columns(2)
    with c1:
        brand_id_col = st.selectbox(
            "brand_id (required)",
            headers,
            index=idx_header(suggested["brand_id"], headers),
        )
        website_col = st.selectbox(
            "website_url (optional)",
            options,
            index=idx_header(suggested["website_url"], options),
        )
        logo_col = st.selectbox(
            "logo_url (optional)",
            options,
            index=idx_header(suggested["logo_url"], options),
        )
    with c2:
        brand_name_col = st.selectbox(
            "brand_name (required)",
            headers,
            index=idx_header(suggested["brand_name"], headers),
        )
        product_count_col = st.selectbox(
            "product_count (optional)",
            options,
            index=idx_header(suggested["product_count"], options),
        )
        category_col = st.selectbox(
            "category (optional)",
            options,
            index=idx_header(suggested["category"], options),
        )

    st.subheader("Removable Tokens / Phrases")
    token_text = st.text_area(
        "One token or phrase per line",
        value="\n".join(DEFAULT_REMOVABLE),
        height=220,
    )
    removable_tokens = parse_tokens_input(token_text)

    st.subheader("Matching Settings")
    cset1, cset2, cset3, cset4, cset5 = st.columns(5)
    with cset1:
        min_score_to_show = st.number_input(
            "Min confidence to show",
            min_value=0,
            max_value=100,
            value=int(DEFAULT_MATCHING_CONFIG["min_score_to_show"]),
            step=1,
        )
    with cset2:
        allow_category_assisted_low = st.checkbox(
            "Allow category-assisted low confidence",
            value=bool(DEFAULT_MATCHING_CONFIG["allow_category_assisted_low_confidence"]),
        )
    with cset3:
        category_assisted_min_score = st.number_input(
            "Category-assisted min score",
            min_value=0,
            max_value=100,
            value=int(DEFAULT_MATCHING_CONFIG["category_assisted_min_score"]),
            step=1,
        )
    with cset4:
        include_low_conf = st.checkbox(
            "Include low-confidence candidates",
            value=bool(DEFAULT_MATCHING_CONFIG["include_low_confidence_candidates"]),
            help="When enabled, adds lower-similarity name matches inside the same prefix bucket.",
        )
    with cset5:
        low_conf_threshold = st.number_input(
            "Low-confidence name threshold",
            min_value=50,
            max_value=100,
            value=int(DEFAULT_MATCHING_CONFIG["low_confidence_compare_threshold"]),
            step=1,
            help="Lower values include more low-confidence candidates.",
        )
    matching_config = {
        "min_score_to_show": int(min_score_to_show),
        "allow_category_assisted_low_confidence": bool(allow_category_assisted_low),
        "category_assisted_min_score": int(category_assisted_min_score),
        "include_low_confidence_candidates": bool(include_low_conf),
        "low_confidence_compare_threshold": int(low_conf_threshold),
    }

    mapping = {
        "brand_id": brand_id_col,
        "brand_name": brand_name_col,
        "website_url": None if website_col == "<none>" else website_col,
        "logo_url": None if logo_col == "<none>" else logo_col,
        "product_count": None if product_count_col == "<none>" else product_count_col,
        "category": None if category_col == "<none>" else category_col,
    }

    preview = build_brand_records(df.head(25), mapping, removable_tokens)
    if preview:
        st.subheader("Preview (first 25 normalized)")
        st.dataframe(pd.DataFrame(preview), use_container_width=True)

    if st.button("Create Project", type="primary"):
        if not reviewer_name:
            st.error("Reviewer name is required.")
            return
        if not project_name.strip():
            st.error("Project name is required.")
            return
        records = build_brand_records(df, mapping, removable_tokens)
        if not records:
            st.error("No valid rows after mapping/normalization.")
            return

        dedupe_check = pd.Series([r["brand_id"] for r in records]).duplicated().any()
        if dedupe_check:
            st.error("Duplicate brand_id values detected in upload. brand_id must be unique per project.")
            return

        with st.spinner("Creating project and generating candidates..."):
            project_id, candidate_count = insert_project_and_data(
                reviewer=reviewer_name,
                project_name=project_name.strip(),
                filename=uploaded.name,
                records=records,
                removable_tokens=removable_tokens,
                matching_config=matching_config,
                notes=notes.strip() or None,
            )
        st.success(f"Project created: {project_id}. Candidates generated: {candidate_count}")


def render_project_admin(project: dict[str, Any]) -> None:
    st.subheader("Project Settings")
    st.write(f"Project ID: `{project['id']}`")
    st.write(f"Created: {project['created_at']}")
    st.write(f"Rows: {project['row_count']}")

    edit_name = st.text_input("Project name", value=str(project.get("name") or ""), key=f"project_name_{project['id']}")
    edit_notes = st.text_area("Project notes", value=str(project.get("notes") or ""), key=f"project_notes_{project['id']}")

    removable_tokens = project.get("removable_tokens") or []
    if isinstance(removable_tokens, str):
        try:
            removable_tokens = json.loads(removable_tokens)
        except Exception:
            removable_tokens = []
    matching_config = parse_matching_config(project.get("matching_config"))
    cset1, cset2, cset3, cset4, cset5 = st.columns(5)
    with cset1:
        min_score_to_show = st.number_input(
            "Min confidence to show",
            min_value=0,
            max_value=100,
            value=int(matching_config["min_score_to_show"]),
            step=1,
            key=f"min_score_{project['id']}",
        )
    with cset2:
        allow_category_assisted_low = st.checkbox(
            "Allow category-assisted low confidence",
            value=bool(matching_config["allow_category_assisted_low_confidence"]),
            key=f"allow_cat_low_{project['id']}",
        )
    with cset3:
        category_assisted_min_score = st.number_input(
            "Category-assisted min score",
            min_value=0,
            max_value=100,
            value=int(matching_config["category_assisted_min_score"]),
            step=1,
            key=f"cat_min_score_{project['id']}",
        )
    with cset4:
        include_low_conf = st.checkbox(
            "Include low-confidence candidates",
            value=bool(matching_config["include_low_confidence_candidates"]),
            key=f"include_low_conf_{project['id']}",
            help="When enabled, adds lower-similarity name matches inside the same prefix bucket.",
        )
    with cset5:
        low_conf_threshold = st.number_input(
            "Low-confidence name threshold",
            min_value=50,
            max_value=100,
            value=int(matching_config["low_confidence_compare_threshold"]),
            step=1,
            key=f"low_conf_threshold_{project['id']}",
            help="Lower values include more low-confidence candidates.",
        )
    matching_config = {
        "min_score_to_show": int(min_score_to_show),
        "allow_category_assisted_low_confidence": bool(allow_category_assisted_low),
        "category_assisted_min_score": int(category_assisted_min_score),
        "include_low_confidence_candidates": bool(include_low_conf),
        "low_confidence_compare_threshold": int(low_conf_threshold),
    }

    if st.button("Save project settings", key=f"save_project_{project['id']}"):
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    update projects
                    set name = %s,
                        notes = %s,
                        matching_config = %s
                    where id = %s
                    """,
                    (
                        (edit_name or "").strip() or str(project.get("name") or ""),
                        (edit_notes or "").strip() or None,
                        json.dumps(parse_matching_config(matching_config)),
                        str(project["id"]),
                    ),
                )
            conn.commit()
        st.success("Project settings saved.")

    token_text = st.text_area(
        "Removable tokens/phrases for compare_norm",
        value="\n".join(removable_tokens),
        height=220,
        key=f"tokens_{project['id']}",
    )

    confirm = st.checkbox(
        "Confirm regenerate (this clears existing decisions for this project)",
        key=f"regen_confirm_{project['id']}",
    )
    if st.button("Regenerate candidates with these tokens", key=f"regen_{project['id']}"):
        if not confirm:
            st.error("Please confirm regeneration before proceeding.")
            return
        parsed = parse_tokens_input(token_text)
        with st.spinner("Regenerating compare_norm values and candidates..."):
            count = regenerate_project_candidates(str(project["id"]), parsed, matching_config)
        st.success(f"Candidates regenerated: {count}")

    st.markdown("---")
    st.subheader("Delete Project")
    st.caption("This permanently deletes project brands, candidates, and decisions.")
    confirm_name = st.text_input(
        f"Type project name to confirm delete: {project['name']}",
        key=f"delete_confirm_{project['id']}",
    )
    if st.button("Delete project", key=f"delete_project_{project['id']}", type="primary"):
        if confirm_name.strip() != str(project["name"]):
            st.error("Project name does not match. Delete canceled.")
            return
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("delete from projects where id = %s", (str(project["id"]),))
            conn.commit()
        st.success("Project deleted.")
        st.rerun()


def render_admin_queue(project: dict[str, Any], reviewer_name: str) -> None:
    st.header("Review Queue")
    st.caption("Admin view with filters and manual candidate selection.")
    if not reviewer_name:
        st.warning("Enter Reviewer name in the sidebar to review candidates.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        score_min = st.slider("Score >=", min_value=0, max_value=100, value=90)
        same_domain_only = st.checkbox("Same domain only")
    with c2:
        missing_url_only = st.checkbox("Missing URL")
        product_count_diff = st.checkbox("product_count differs")
    with c3:
        limit = st.number_input("Next N pending pairs", min_value=1, max_value=200, value=25)
        search = st.text_input("Search brand name/id")

    filters = {
        "score_min": score_min,
        "same_domain_only": same_domain_only,
        "missing_url_only": missing_url_only,
        "product_count_diff": product_count_diff,
        "search": search.strip(),
    }

    queue = queue_query(str(project["id"]), reviewer_name, filters, int(limit))
    if not queue:
        st.info("No matching pending pairs.")
        return

    rows_for_table = []
    for row in queue:
        rows_for_table.append(
            {
                "candidate_id": row["id"],
                "score": row["score"],
                "brand_id_a": row["brand_id_a"],
                "brand_name_a": row["brand_name_a"],
                "brand_id_b": row["brand_id_b"],
                "brand_name_b": row["brand_name_b"],
                "status": row["status"],
                "locked_by": row["locked_by"],
            }
        )

    st.dataframe(pd.DataFrame(rows_for_table), use_container_width=True)
    ids = [str(r["id"]) for r in queue]
    selected_id = st.selectbox("Select candidate", ids)
    selected = next(r for r in queue if str(r["id"]) == selected_id)
    render_candidate_decision(selected, str(project["id"]), reviewer_name)


def fetch_next_candidate_for_reviewer(
    project_id: str,
    reviewer_name: str,
    sort_mode: str,
    skip_candidate_id: str | None = None,
) -> dict[str, Any] | None:
    filters = {
        "score_min": 0,
        "same_domain_only": False,
        "missing_url_only": False,
        "product_count_diff": False,
        "search": "",
    }
    score_order = "asc" if sort_mode == "Lowest confidence" else "desc"
    queue = queue_query(project_id, reviewer_name, filters, limit=150, score_order=score_order)

    if sort_mode == "Most brands in cluster":
        cluster_sizes = cluster_sizes_from_queue_rows(queue)
        ranked: list[tuple[int, int, dict[str, Any]]] = []
        for row in queue:
            size = int(cluster_sizes.get(str(row["id"]), 1))
            ranked.append((size, int(row["score"]), row))
        ranked.sort(key=lambda x: (-x[0], -x[1]))
        queue = [r[2] for r in ranked]

    if skip_candidate_id and len(queue) > 1:
        preferred = [r for r in queue if str(r["id"]) != skip_candidate_id]
        fallback = [r for r in queue if str(r["id"]) == skip_candidate_id]
        if preferred:
            queue = preferred + fallback

    for row in queue:
        if lock_candidate(project_id, str(row["id"]), reviewer_name):
            row["status"] = "locked"
            row["locked_by"] = reviewer_name
            return row
    return None


def handle_sort_change(project_id: str, reviewer_name: str) -> None:
    current_candidate = st.session_state.get(f"selected_candidate_{project_id}")
    if current_candidate:
        st.session_state[f"skip_candidate_on_next_pick_{project_id}"] = str(current_candidate)
    st.session_state[f"active_candidate_id_{project_id}"] = None
    if reviewer_name:
        release_reviewer_locks(project_id, reviewer_name)
    st.session_state["queue_should_rerun"] = True


def render_candidate_decision(selected: dict[str, Any], project_id: str, reviewer_name: str) -> None:
    selected_id = str(selected["id"])
    selected_state_key = f"selected_candidate_{project_id}"
    lock_state_key = f"locked_candidate_{project_id}"

    previous_selected = st.session_state.get(selected_state_key)
    if previous_selected != selected_id:
        st.session_state[selected_state_key] = selected_id
        st.session_state[lock_state_key] = None

    already_locked_for_me = (
        selected["status"] == "locked" and selected["locked_by"] == reviewer_name
    )
    if st.session_state.get(lock_state_key) != selected_id and not already_locked_for_me:
        if lock_candidate(project_id, selected_id, reviewer_name):
            st.session_state[lock_state_key] = selected_id
            st.info("This candidate is reserved for you for 10 minutes.")
        else:
            st.warning("This candidate was just taken by another reviewer. Loading another one...")
            st.rerun()
            return
    elif already_locked_for_me:
        st.session_state[lock_state_key] = selected_id

    members_df, related_pairs_df = fetch_duplicate_group_context(
        project_id,
        selected["brand_id_a"],
        selected["brand_id_b"],
    )

    st.subheader("Select the Winner")
    member_rows = members_df.to_dict(orient="records")
    default_winner_id, default_reason = winner_default_for_group_rows(member_rows)
    suggested_winner_id = str(default_winner_id)
    member_rows = sorted(
        member_rows,
        key=lambda r: (str(r.get("brand_id")) != suggested_winner_id, to_str(r.get("brand_name"))),
    )
    show_exclude = len(member_rows) > 2
    if show_exclude:
        st.caption("Pick one winner. All other rows merge into that winner unless excluded.")
    else:
        st.caption("Pick one winner. The other row will merge into that winner.")

    brand_ids = [str(r["brand_id"]) for r in member_rows]
    label_by_id = {str(r["brand_id"]): f"{r.get('brand_name') or r['brand_id']} (ID {r['brand_id']})" for r in member_rows}

    existing_winner = str(st.session_state.get(f"group_winner_{selected_id}") or "")
    if existing_winner not in brand_ids:
        existing_winner = str(default_winner_id)

    chosen_winner = existing_winner

    existing_excluded_ids = set()
    if show_exclude:
        existing_excluded_ids = {
            str(x)
            for x in st.session_state.get(f"group_excluded_ids_{selected_id}", [])
        }

    if show_exclude:
        col_widths = [1.0, 1.0, 2.0, 2.0, 1.2, 2.4, 1.6, 1.2]
        headers = ["Winner", "Logo", "Brand", "Website", "Products", "Categories", "Domain", "Exclude"]
    else:
        col_widths = [1.0, 1.0, 2.0, 2.0, 1.2, 2.4, 1.6]
        headers = ["Winner", "Logo", "Brand", "Website", "Products", "Categories", "Domain"]
    header_cols = st.columns(col_widths, vertical_alignment="center")
    for col, label in zip(header_cols, headers):
        with col:
            st.caption(label)

    for row in member_rows:
        bid = str(row["brand_id"])
        row_cols = st.columns(col_widths, vertical_alignment="center")
        with row_cols[0]:
            if bid == chosen_winner:
                st.markdown("🔘")
            else:
                if st.button(
                    "◯",
                    key=f"winner_pick_{selected_id}_{bid}",
                    help=f"Set {label_by_id.get(bid, bid)} as winner",
                    type="tertiary",
                ):
                    st.session_state[f"group_winner_{selected_id}"] = bid
                    st.session_state[f"group_exclude_{selected_id}_{bid}"] = False
                    st.rerun()
        with row_cols[1]:
            logo_url = to_str(row.get("logo_url"))
            if logo_url:
                st.image(logo_url, width=72)
            else:
                st.caption("None")
        with row_cols[2]:
            st.markdown(f"**{to_str(row.get('brand_name')) or '(blank)'}**")
            st.caption(f"ID {bid}")
        with row_cols[3]:
            st.write(to_str(row.get("website_url")) or "None")
        with row_cols[4]:
            st.write(to_int(row.get("product_count")) or 0)
        with row_cols[5]:
            category_text = to_str(row.get("category_norm")) or "None"
            preview_chars = 20
            category_expand_key = f"category_expand_{selected_id}_{bid}"
            if category_expand_key not in st.session_state:
                st.session_state[category_expand_key] = False
            if category_text != "None" and len(category_text) > preview_chars:
                if st.session_state[category_expand_key]:
                    st.write(category_text)
                    if st.button("[collapse]", key=f"{category_expand_key}_collapse", type="tertiary"):
                        st.session_state[category_expand_key] = False
                        st.rerun()
                else:
                    preview = category_text[:preview_chars].rstrip()
                    if st.button(
                        f"{preview}... [expand]",
                        key=f"{category_expand_key}_expand",
                        type="tertiary",
                    ):
                        st.session_state[category_expand_key] = True
                        st.rerun()
            else:
                st.write(category_text)
        with row_cols[6]:
            st.write(to_str(row.get("domain_norm")) or "None")
        if show_exclude:
            with row_cols[7]:
                exclude_key = f"group_exclude_{selected_id}_{bid}"
                if exclude_key not in st.session_state:
                    st.session_state[exclude_key] = bid in existing_excluded_ids and bid != chosen_winner
                if bid == chosen_winner:
                    st.session_state[exclude_key] = False
                st.checkbox(
                    "Exclude",
                    key=exclude_key,
                    disabled=(bid == chosen_winner),
                    label_visibility="collapsed",
                )
        st.divider()

    if show_exclude:
        chosen_losers = [
            str(bid)
            for bid in brand_ids
            if str(bid) != chosen_winner and not to_bool(st.session_state.get(f"group_exclude_{selected_id}_{bid}", False))
        ]
        chosen_excluded = [
            str(bid)
            for bid in brand_ids
            if str(bid) != chosen_winner and to_bool(st.session_state.get(f"group_exclude_{selected_id}_{bid}", False))
        ]
    else:
        chosen_losers = [str(bid) for bid in brand_ids if str(bid) != chosen_winner]
        chosen_excluded = []

    st.session_state[f"group_winner_{selected_id}"] = chosen_winner
    st.session_state[f"group_loser_ids_{selected_id}"] = chosen_losers
    st.session_state[f"group_excluded_ids_{selected_id}"] = chosen_excluded

    if not chosen_losers:
        st.warning("No loser rows selected. Uncheck Exclude on at least one row to merge.")

    winner_lookup = {str(r["brand_id"]): r for r in member_rows}
    if chosen_winner and chosen_winner in winner_lookup:
        winner_name = str(winner_lookup[chosen_winner].get("brand_name") or chosen_winner)
        winner_url = str(winner_lookup[chosen_winner].get("website_url") or "")
        if chosen_winner == str(default_winner_id):
            winner_reason_readable = f"{winner_name} (ID {chosen_winner}) - {default_reason}"
        else:
            winner_reason_readable = f"{winner_name} (ID {chosen_winner}) - reviewer selected"
        if f"group_updated_winner_name_{selected_id}" not in st.session_state:
            st.session_state[f"group_updated_winner_name_{selected_id}"] = winner_name
        if f"group_updated_winner_url_{selected_id}" not in st.session_state:
            st.session_state[f"group_updated_winner_url_{selected_id}"] = winner_url
    else:
        winner_reason_readable = "Select exactly one winner"
        winner_name = ""
        winner_url = ""

    current_pair_score = int(selected.get("score") or 0)
    cluster_score = current_pair_score
    cluster_min = current_pair_score
    cluster_max = current_pair_score
    if not related_pairs_df.empty and "score" in related_pairs_df.columns:
        try:
            score_values = [int(x) for x in related_pairs_df["score"].tolist()]
            cluster_min = min(score_values)
            cluster_max = max(score_values)
            cluster_score = current_pair_score
        except Exception:
            cluster_score = current_pair_score
            cluster_min = current_pair_score
            cluster_max = current_pair_score
    confidence_label, confidence_help = confidence_from_score(cluster_score)
    if cluster_score >= 95:
        confidence_color = "#1f9d55"
        confidence_bg = "#e7f6ec"
    elif cluster_score >= 90:
        confidence_color = "#a16207"
        confidence_bg = "#fef9e7"
    elif cluster_score >= 85:
        confidence_color = "#c2410c"
        confidence_bg = "#fff7ed"
    else:
        confidence_color = "#4b5563"
        confidence_bg = "#f3f4f6"

    st.markdown(
        (
            f"<div style='display:inline-block;padding:4px 10px;border-radius:999px;"
            f"background:{confidence_bg};color:{confidence_color};font-size:0.88rem;font-weight:600;'>"
            f"Confidence: {confidence_label} ({cluster_score}/100)"
            f"</div>"
            f"<div style='margin-top:6px;color:#6b7280;font-size:0.84rem;'>"
            f"{confidence_help} Current pair {current_pair_score}/100. Cluster range {cluster_min}-{cluster_max}."
            f"</div>"
        ),
        unsafe_allow_html=True,
    )
    st.caption(f"Suggested winner: {winner_reason_readable}")
    with st.expander("Edit Winner Details (Optional)"):
        st.text_input(
            "Updated winner brand name",
            key=f"group_updated_winner_name_{selected_id}",
            placeholder="Winner brand name",
        )
        st.text_input(
            "Updated winner website URL",
            key=f"group_updated_winner_url_{selected_id}",
            placeholder="https://example.com",
        )

    with st.expander("Add Merge Notes (Optional)"):
        st.text_area("Merge notes", key=f"group_notes_{selected_id}", label_visibility="collapsed")
    flash_key = f"queue_flash_{project_id}"
    st.session_state[f"group_winner_reason_{selected_id}"] = winner_reason_readable
    if not related_pairs_df.empty and "candidate_id" in related_pairs_df.columns:
        st.session_state[f"group_candidate_ids_{selected_id}"] = [str(x) for x in related_pairs_df["candidate_id"].tolist()]
    else:
        st.session_state[f"group_candidate_ids_{selected_id}"] = [str(selected_id)]

    c1, c2, _ = st.columns([2, 1.4, 3.6])
    with c1:
        st.button(
            "Merge selected losers into selected winner",
            key=f"save_group_merge_{selected_id}",
            use_container_width=True,
            disabled=(chosen_winner is None or not chosen_losers),
            on_click=save_group_merge_from_state,
            args=(project_id, selected_id, reviewer_name),
        )
    with c2:
        st.button(
            "No dupes in this cluster",
            key=f"no_dupes_cluster_{selected_id}",
            use_container_width=True,
            on_click=mark_cluster_no_dupes_from_state,
            args=(project_id, selected_id, reviewer_name),
        )

    flash = st.session_state.pop(flash_key, None)
    if flash:
        level, message = flash
        icon = "✅" if level == "success" else "⚠️"
        # Avoid duplicate toast: queue page already displays this notification.
        st.session_state.pop("global_flash", None)
        st.toast(message, icon=icon)
    if st.session_state.pop("queue_should_rerun", False):
        st.rerun()


def render_reviewer_start(projects: list[dict[str, Any]]) -> None:
    st.header("Welcome to Designer Pages Deduper")

    if not projects:
        st.info("No project yet. Ask an admin to upload a CSV first.")
        return

    project_ids = [str(p["id"]) for p in projects]
    labels = [str(p["name"]) for p in projects]
    current_project_id = st.session_state.get("reviewer_project_id")
    default_index = 0
    if current_project_id:
        for i, pid in enumerate(project_ids):
            if pid == current_project_id:
                default_index = i
                break

    selected_index = st.selectbox(
        "Select project",
        options=list(range(len(labels))),
        index=default_index,
        format_func=lambda i: labels[i],
    )
    selected_project_id = project_ids[selected_index]
    name = st.text_input("Reviewer name", key="reviewer_name_entry")
    if st.button("Begin reviewing", type="primary"):
        if not name.strip():
            st.error("Please enter your name.")
            return
        st.session_state["reviewer_name"] = name.strip()
        st.session_state["reviewer_project_id"] = selected_project_id
        st.session_state["review_started"] = True
        st.rerun()


def reset_reviewer_session() -> None:
    st.session_state["review_started"] = False
    st.session_state["reviewer_name"] = ""
    st.session_state["reviewer_project_id"] = ""
    st.session_state.pop("reviewer_name_entry", None)


def render_reviewer_queue(project: dict[str, Any], reviewer_name: str) -> None:
    project_id = str(project["id"])
    if st.query_params.get("exit_reviewer") == "1":
        reset_reviewer_session()
        st.query_params.clear()
        st.rerun()
    st.markdown(f"Reviewer: {reviewer_name} | [Exit](?exit_reviewer=1)")

    completed_clusters, total_clusters, pct = fetch_cluster_progress(project_id)
    if total_clusters > 0:
        st.caption(
            f"Project: {project['name']} | {completed_clusters}/{total_clusters} clusters reviewed ({pct:.0f}%)"
        )
        st.progress(min(max(pct / 100.0, 0.0), 1.0))
    else:
        st.caption(f"Project: {project['name']} | No clusters identified yet")

    sort_mode = st.selectbox(
        "Sort candidates by",
        options=["Highest confidence", "Lowest confidence", "Most brands in cluster"],
        key=f"review_sort_mode_{project['id']}",
        on_change=handle_sort_change,
        args=(project_id, reviewer_name),
    )
    if not reviewer_name:
        st.warning("Enter your name to begin reviewing.")
        return

    filters = {
        "score_min": 0,
        "same_domain_only": False,
        "missing_url_only": False,
        "product_count_diff": False,
        "search": "",
    }
    score_order = "asc" if sort_mode == "Lowest confidence" else "desc"
    queue = queue_query(project_id, reviewer_name, filters, limit=150, score_order=score_order)

    if sort_mode == "Most brands in cluster":
        cluster_sizes = cluster_sizes_from_queue_rows(queue)
        ranked: list[tuple[int, int, dict[str, Any]]] = []
        for row in queue:
            size = int(cluster_sizes.get(str(row["id"]), 1))
            ranked.append((size, int(row["score"]), row))
        ranked.sort(key=lambda x: (-x[0], -x[1]))
        queue = [r[2] for r in ranked]

    active_key = f"active_candidate_id_{project_id}"
    active_candidate_id = st.session_state.get(active_key)
    candidate = None

    # Keep the current cluster pinned while the reviewer edits controls.
    if active_candidate_id:
        candidate = fetch_candidate_by_id(project_id, str(active_candidate_id))
        if candidate:
            status = str(candidate.get("status") or "")
            locked_by = str(candidate.get("locked_by") or "")
            is_open_for_me = (
                status == "pending"
                or (status == "locked" and locked_by == reviewer_name)
            )
            if not is_open_for_me:
                candidate = None
        if candidate:
            already_mine = candidate["status"] == "locked" and candidate["locked_by"] == reviewer_name
            if not already_mine and not lock_candidate(project_id, str(candidate["id"]), reviewer_name):
                candidate = None
            elif not already_mine:
                candidate["status"] = "locked"
                candidate["locked_by"] = reviewer_name
        if candidate is None:
            st.session_state[active_key] = None

    if candidate is None:
        skip_candidate_id = st.session_state.pop(f"skip_candidate_on_next_pick_{project_id}", None)
        if skip_candidate_id and len(queue) > 1:
            preferred = [r for r in queue if str(r["id"]) != skip_candidate_id]
            fallback = [r for r in queue if str(r["id"]) == skip_candidate_id]
            if preferred:
                queue = preferred + fallback

        for row in queue:
            if lock_candidate(project_id, str(row["id"]), reviewer_name):
                row["status"] = "locked"
                row["locked_by"] = reviewer_name
                candidate = row
                st.session_state[active_key] = str(row["id"])
                break

    if not candidate:
        st.session_state[active_key] = None
        st.success("No pending candidates left to review.")
        return

    st.session_state[active_key] = str(candidate["id"])
    render_candidate_decision(candidate, project_id, reviewer_name)




def render_exports(project: dict[str, Any]) -> None:
    st.header("Exports")
    merge_df = fetch_export_merge(str(project["id"]))
    decisions_df = fetch_export_decisions(str(project["id"]))
    approved_count = int((decisions_df["decision"] == "approved").sum()) if not decisions_df.empty else 0
    rejected_count = int((decisions_df["decision"] == "rejected").sum()) if not decisions_df.empty else 0
    skipped_count = int((decisions_df["decision"] == "skipped").sum()) if not decisions_df.empty else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Approved", approved_count)
    c2.metric("Rejected", rejected_count)
    c3.metric("Skipped", skipped_count)

    st.markdown("**merge_instructions.csv**")
    st.download_button(
        "Download merge_instructions.csv",
        data=csv_bytes(merge_df),
        file_name="merge_instructions.csv",
        mime="text/csv",
    )
    st.dataframe(merge_df.head(50), use_container_width=True)

    st.markdown("**decisions.csv**")
    st.download_button(
        "Download decisions.csv",
        data=csv_bytes(decisions_df),
        file_name="decisions.csv",
        mime="text/csv",
    )
    st.dataframe(decisions_df.head(50), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Brand Dedupe", page_icon="🔎", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(
        """
        <style>
        [data-testid="stSidebarCollapsedControl"] button::after {
            content: "Admin Access";
            font-size: 0.82rem;
            font-weight: 500;
            margin-left: 0.35rem;
            color: rgba(49, 51, 63, 0.8);
            white-space: nowrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if "review_started" not in st.session_state:
        st.session_state["review_started"] = False
    if "reviewer_name" not in st.session_state:
        st.session_state["reviewer_name"] = ""
    if "reviewer_project_id" not in st.session_state:
        st.session_state["reviewer_project_id"] = ""
    if "admin_unlocked" not in st.session_state:
        st.session_state["admin_unlocked"] = False

    with st.sidebar:
        configured_admin_code = admin_access_code()
        if not st.session_state["admin_unlocked"]:
            if not configured_admin_code:
                st.error("Admin code is not configured. Set ADMIN_ACCESS_CODE in secrets or env.")
            else:
                st.text_input("Admin access code", type="password", key="admin_code_entry")
                if st.button("Unlock admin"):
                    provided = st.session_state.get("admin_code_entry", "")
                    if provided == configured_admin_code:
                        st.session_state["admin_unlocked"] = True
                        st.session_state.pop("admin_code_entry", None)
                        st.rerun()
                    else:
                        st.error("Invalid admin code.")

        admin_mode = st.session_state["admin_unlocked"]
        if admin_mode:
            st.caption("Admin mode enabled")
            if st.button("Run DB setup (first time)"):
                try:
                    run_schema_setup()
                    st.success("Schema setup complete.")
                except Exception as exc:
                    st.error(f"Schema setup failed: {exc}")
            if st.button("Exit admin mode"):
                st.session_state["admin_unlocked"] = False
                st.rerun()

        # auto-dismissed toasts are shown in main content area

    try:
        projects = fetch_projects()
    except Exception as exc:
        st.error(f"Database connection failed: {exc}")
        st.stop()

    has_queue_flash = any(
        k.startswith("queue_flash_") and st.session_state.get(k) is not None
        for k in st.session_state.keys()
    )
    global_flash = st.session_state.pop("global_flash", None)
    if global_flash and not has_queue_flash:
        level, msg = global_flash
        st.toast(msg, icon=("✅" if level == "success" else "⚠️"))

    project_map = {str(p["id"]): p for p in projects}
    project_by_id = {str(p["id"]): p for p in projects}
    selected_project_id = None

    reviewer_name = st.session_state.get("reviewer_name", "")

    if admin_mode:
        st.title("Brand Dedupe Tool")
        st.caption("Collaborative duplicate review queue with Supabase Postgres persistence.")
        pages = ["Upload CSV", "Project Settings", "Exports", "Diagnostics"]
        page = st.sidebar.radio("Page", pages)
        if projects:
            project_index = st.sidebar.selectbox(
                "Project",
                options=list(range(len(projects))),
                format_func=lambda i: str(projects[i]["name"]),
                index=0,
            )
            selected_project_id = str(projects[project_index]["id"])
        project = project_map[selected_project_id] if selected_project_id else None
    else:
        page = "Review Queue" if st.session_state.get("review_started") else "Start Review"
        selected_project_id = st.session_state.get("reviewer_project_id", "")
        project = project_by_id.get(selected_project_id)
        st.markdown(
            """
            <style>
            .stDeployButton {
                display: none !important;
            }
            [data-testid="stAppDeployButton"] {
                display: none !important;
            }
            [data-testid="stHeaderActionElements"] {
                display: none !important;
            }
            [data-testid="stToolbarActions"] {
                display: none !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    if page == "Start Review":
        render_reviewer_start(projects)
        return

    if page == "Upload CSV":
        upload_user = "admin" if admin_mode else reviewer_name
        render_upload_page(upload_user)
        if projects:
            st.subheader("Existing Projects")
            project_rows = []
            for p in projects:
                project_rows.append(
                    {
                        "name": p["name"],
                        "id": p["id"],
                        "row_count": p["row_count"],
                        "pending": p["pending_count"],
                        "locked": p["locked_count"],
                        "approved": p["approved_count"],
                        "rejected": p["rejected_count"],
                        "skipped": p["skipped_count"],
                        "created_at": p["created_at"],
                    }
                )
            st.dataframe(pd.DataFrame(project_rows), use_container_width=True)
        return

    if not project:
        st.warning("Create a project first on Upload CSV.")
        return

    if page == "Review Queue":
        if not admin_mode:
            render_reviewer_queue(project, reviewer_name)
    elif page == "Project Settings":
        render_project_admin(project)
    elif page == "Exports":
        render_exports(project)
    elif page == "Diagnostics":
        render_diagnostics(project)


if __name__ == "__main__":
    main()
