"""
Corpus composition analysis.
Queries the live database directly — no ORM overhead, raw SQL for speed.
Run: python scripts/corpus_analysis.py
"""
from __future__ import annotations
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import psycopg

# ── connection ─────────────────────────────────────────────────────────────────
DSN = None
for line in open(os.path.join(ROOT, ".env")):
    if line.strip().startswith("DATABASE_URL="):
        raw = line.strip().split("=", 1)[1].strip()
        # psycopg3 uses postgresql://  (strip +psycopg driver hint)
        DSN = raw.replace("postgresql+psycopg://", "postgresql://")
        break
assert DSN, "DATABASE_URL not found"

conn = psycopg.connect(DSN)
cur  = conn.cursor()

def q(sql, params=None):
    cur.execute(sql, params or [])
    return cur.fetchall()

def q1(sql, params=None):
    cur.execute(sql, params or [])
    row = cur.fetchone()
    return row[0] if row else None

SEP  = "─" * 70
SEP2 = "═" * 70

# ── 1. corpus totals ───────────────────────────────────────────────────────────
total_incidents  = q1("SELECT COUNT(*) FROM incidents")
total_raw        = q1("SELECT COUNT(*) FROM raw_documents")
total_sources    = q1("SELECT COUNT(*) FROM incident_sources")
total_embeddings = q1("SELECT COUNT(*) FROM embeddings")
gold_labeled     = q1("SELECT COUNT(*) FROM incidents WHERE is_gold_labeled = true")
with_resolution  = q1("SELECT COUNT(*) FROM incidents WHERE resolution_summary IS NOT NULL AND resolution_summary != ''")

print(SEP2)
print("CORPUS OVERVIEW")
print(SEP2)
print(f"  total incidents          : {total_incidents:,}")
print(f"  total raw documents      : {total_raw:,}")
print(f"  total sources            : {total_sources}")
print(f"  incidents with embedding : {total_embeddings:,}")
print(f"  gold-labeled incidents   : {gold_labeled:,}  ({gold_labeled/max(total_incidents,1)*100:.1f}%)")
print(f"  with resolution summary  : {with_resolution:,}  ({with_resolution/max(total_incidents,1)*100:.1f}%)")

# ── 2. duplicate rate ─────────────────────────────────────────────────────────
# raw_documents that have no corresponding incident = orphaned raws
orphan_raws = q1("""
    SELECT COUNT(*) FROM raw_documents rd
    LEFT JOIN incidents i ON i.raw_document_id = rd.id
    WHERE i.id IS NULL
""")
# incidents whose raw payload_hash appears more than once
dup_payload = q1("""
    SELECT COUNT(*) FROM (
        SELECT payload_hash, COUNT(*) c
        FROM raw_documents
        GROUP BY payload_hash
        HAVING COUNT(*) > 1
    ) t
""")

print(f"\n{SEP}")
print("DUPLICATE / ORPHAN RATE")
print(SEP)
print(f"  raw docs without incident  : {orphan_raws:,}")
print(f"  duplicate payload hashes   : {dup_payload:,}")
print(f"  effective duplicate rate   : {orphan_raws/max(total_raw,1)*100:.1f}%")

# ── 3. per-repository breakdown ───────────────────────────────────────────────
repo_rows = q("""
    SELECT
        COALESCE(owner,'(null)') || '/' || COALESCE(repo,'(null)') AS full_name,
        COUNT(*) AS incidents,
        ROUND(COUNT(*) * 100.0 / NULLIF(SUM(COUNT(*)) OVER (), 0), 1) AS pct,
        SUM(CASE WHEN is_gold_labeled THEN 1 ELSE 0 END) AS gold,
        SUM(CASE WHEN resolution_summary IS NOT NULL AND resolution_summary != '' THEN 1 ELSE 0 END) AS resolved
    FROM incidents
    GROUP BY owner, repo
    ORDER BY incidents DESC
""")

print(f"\n{SEP}")
print("INCIDENTS PER REPOSITORY")
print(SEP)
print(f"{'repository':<40} {'count':>6} {'%':>6} {'gold':>5} {'resolved':>9}")
print(f"{'─'*40} {'─'*6} {'─'*6} {'─'*5} {'─'*9}")
dominant = []
for name, cnt, pct, gold, resolved in repo_rows:
    marker = " ◄ >10%" if pct and float(pct) > 10 else ""
    if pct and float(pct) > 10:
        dominant.append((name, cnt, float(pct)))
    print(f"{name:<40} {cnt:>6,} {float(pct or 0):>5.1f}% {gold:>5,} {resolved:>9,}{marker}")

# ── 4. source diversity ───────────────────────────────────────────────────────
source_rows = q("""
    SELECT s.name, s.source_type, COUNT(i.id) AS incidents
    FROM incident_sources s
    LEFT JOIN raw_documents rd ON rd.source_id = s.id
    LEFT JOIN incidents i ON i.raw_document_id = rd.id
    GROUP BY s.id, s.name, s.source_type
    ORDER BY incidents DESC
""")

print(f"\n{SEP}")
print("SOURCE DIVERSITY")
print(SEP)
print(f"{'source name':<45} {'type':<12} {'incidents':>9}")
print(f"{'─'*45} {'─'*12} {'─'*9}")
for name, stype, cnt in source_rows:
    print(f"{name:<45} {stype:<12} {cnt or 0:>9,}")

# ── 5. incident type distribution ────────────────────────────────────────────
type_rows = q("""
    SELECT incident_type, COUNT(*) AS n,
           ROUND(COUNT(*)*100.0/NULLIF(SUM(COUNT(*)) OVER(),0),1) AS pct
    FROM incidents
    GROUP BY incident_type
    ORDER BY n DESC
""")

print(f"\n{SEP}")
print("INCIDENT TYPE DISTRIBUTION")
print(SEP)
bar_w = 40
for itype, n, pct in type_rows:
    bar = "█" * int(float(pct or 0) / 100 * bar_w)
    print(f"  {itype:<15} {n:>5,}  {float(pct or 0):>5.1f}%  {bar}")

# ── 6. severity distribution ─────────────────────────────────────────────────
sev_rows = q("""
    SELECT severity, COUNT(*) AS n,
           ROUND(COUNT(*)*100.0/NULLIF(SUM(COUNT(*)) OVER(),0),1) AS pct
    FROM incidents
    GROUP BY severity
    ORDER BY n DESC
""")

print(f"\n{SEP}")
print("SEVERITY DISTRIBUTION")
print(SEP)
for sev, n, pct in sev_rows:
    bar = "█" * int(float(pct or 0) / 100 * bar_w)
    print(f"  {sev:<10} {n:>5,}  {float(pct or 0):>5.1f}%  {bar}")

# ── 7. status distribution ───────────────────────────────────────────────────
status_rows = q("""
    SELECT status, COUNT(*) AS n,
           ROUND(COUNT(*)*100.0/NULLIF(SUM(COUNT(*)) OVER(),0),1) AS pct
    FROM incidents
    GROUP BY status
    ORDER BY n DESC
""")

print(f"\n{SEP}")
print("STATUS DISTRIBUTION")
print(SEP)
for st, n, pct in status_rows:
    bar = "█" * int(float(pct or 0) / 100 * bar_w)
    print(f"  {st:<12} {n:>5,}  {float(pct or 0):>5.1f}%  {bar}")

# ── 8. state (open/closed) ────────────────────────────────────────────────────
state_rows = q("""
    SELECT state, COUNT(*) AS n,
           ROUND(COUNT(*)*100.0/NULLIF(SUM(COUNT(*)) OVER(),0),1) AS pct
    FROM incidents
    GROUP BY state
    ORDER BY n DESC
""")

print(f"\n{SEP}")
print("OPEN / CLOSED STATE")
print(SEP)
for st, n, pct in state_rows:
    print(f"  {st:<12} {n:>5,}  {float(pct or 0):>5.1f}%")

# ── 9. top tags ───────────────────────────────────────────────────────────────
tag_rows = q("""
    SELECT tag, COUNT(*) AS n
    FROM incidents, UNNEST(tags) AS tag
    GROUP BY tag
    ORDER BY n DESC
    LIMIT 25
""")

print(f"\n{SEP}")
print("TOP 25 TAGS")
print(SEP)
for tag, n in tag_rows:
    print(f"  {tag:<35} {n:>5,}")

# ── 10. confidence distribution ──────────────────────────────────────────────
conf_rows = q("""
    SELECT
        CASE
            WHEN confidence_score >= 0.85 THEN '0.85–1.00 (high)'
            WHEN confidence_score >= 0.65 THEN '0.65–0.85 (medium-high)'
            WHEN confidence_score >= 0.45 THEN '0.45–0.65 (medium)'
            ELSE                               '0.00–0.45 (low)'
        END AS bucket,
        COUNT(*) AS n,
        ROUND(COUNT(*)*100.0/NULLIF(SUM(COUNT(*)) OVER(),0),1) AS pct
    FROM incidents
    GROUP BY bucket
    ORDER BY MIN(confidence_score) DESC
""")

print(f"\n{SEP}")
print("CONFIDENCE SCORE DISTRIBUTION")
print(SEP)
for bucket, n, pct in conf_rows:
    bar = "█" * int(float(pct or 0) / 100 * bar_w)
    print(f"  {bucket:<25} {n:>5,}  {float(pct or 0):>5.1f}%  {bar}")

# ── 11. temporal spread ───────────────────────────────────────────────────────
temporal = q1("""
    SELECT
        MIN(created_at_source)::date || '  →  ' || MAX(created_at_source)::date
    FROM incidents
    WHERE created_at_source IS NOT NULL
""")
median_age = q1("""
    SELECT created_at_source::date FROM incidents
    WHERE created_at_source IS NOT NULL
    ORDER BY created_at_source
    OFFSET (SELECT COUNT(*)/2 FROM incidents WHERE created_at_source IS NOT NULL)
    LIMIT 1
""")

print(f"\n{SEP}")
print("TEMPORAL SPREAD")
print(SEP)
print(f"  created_at_source range  : {temporal}")
print(f"  median created_at_source : {median_age}")

# ── 12. dominant repos summary ───────────────────────────────────────────────
print(f"\n{SEP2}")
print("REPOSITORIES CONTRIBUTING >10% OF CORPUS")
print(SEP2)
if dominant:
    for name, cnt, pct in dominant:
        print(f"  {name:<40} {cnt:>6,} incidents  ({pct:.1f}%)")
else:
    print("  None — corpus is well-distributed.")

print(f"\n{SEP2}")
print("END OF ANALYSIS")
print(SEP2)

cur.close()
conn.close()
