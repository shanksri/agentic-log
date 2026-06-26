"""
Probe each candidate collection strategy against apache/spark.
No writes, no database. Read-only GitHub API calls only.

Strategies tested:
  A - Search API  (type:issue is:closed repo:apache/spark)
  B - GraphQL issues-only query
  C - REST /issues endpoint with label filter
  D - REST /issues endpoint with date-window filter (created range)
  E - REST /issues endpoint explicitly (current approach, baseline)

Run: python scripts/probe_strategies.py
"""
from __future__ import annotations

import os, sys, time, json, textwrap
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import httpx

# ── load token ────────────────────────────────────────────────────────────────
token: str | None = None
for line in open(os.path.join(ROOT, ".env")):
    if line.strip().startswith("GITHUB_TOKEN="):
        token = line.strip().split("=", 1)[1].strip()
        break
assert token, "GITHUB_TOKEN not found"

OWNER, REPO = "apache", "spark"
HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "Authorization": f"Bearer {token}",
    "User-Agent": "probe-strategies",
}

def get(url, *, params=None):
    r = httpx.get(url, headers=HEADERS, params=params, timeout=30, follow_redirects=True)
    r.raise_for_status()
    return r

def post(url, *, json_body):
    r = httpx.post(url, headers=HEADERS, json=json_body, timeout=30)
    r.raise_for_status()
    return r

print("=" * 70)
print(f"Strategy probes: {OWNER}/{REPO}")
print("=" * 70)

# ── Strategy A: Search API ────────────────────────────────────────────────────
print("\n── A: Search API (type:issue is:closed) ──")
try:
    t0 = time.perf_counter()
    r = get("https://api.github.com/search/issues", params={
        "q": f"repo:{OWNER}/{REPO} type:issue is:closed",
        "per_page": 100,
        "page": 1,
        "sort": "updated",
        "order": "desc",
    })
    elapsed_a = time.perf_counter() - t0
    data = r.json()
    total_count = data.get("total_count", "?")
    page1_items = len(data.get("items", []))
    pr_in_page1 = sum(1 for i in data.get("items", []) if i.get("pull_request"))
    rate_limit_remaining = r.headers.get("x-ratelimit-remaining", "?")
    rate_limit_reset     = r.headers.get("x-ratelimit-reset", "?")
    search_rate_remaining = r.headers.get("x-ratelimit-search-remaining", "?")
    print(f"  total_count (reported by API) : {total_count}")
    print(f"  page 1 items                  : {page1_items}")
    print(f"  PRs in page 1                 : {pr_in_page1}")
    print(f"  rate-limit remaining (search) : {search_rate_remaining}")
    print(f"  rate-limit remaining (core)   : {rate_limit_remaining}")
    print(f"  elapsed                       : {elapsed_a:.2f}s")
    # probe page 10 — max allowed is 10 for search
    r10 = get("https://api.github.com/search/issues", params={
        "q": f"repo:{OWNER}/{REPO} type:issue is:closed",
        "per_page": 100,
        "page": 10,
        "sort": "updated",
        "order": "desc",
    })
    page10_items = len(r10.json().get("items", []))
    print(f"  page 10 items (max page)      : {page10_items}")
    try:
        r11 = get("https://api.github.com/search/issues", params={
            "q": f"repo:{OWNER}/{REPO} type:issue is:closed",
            "per_page": 100,
            "page": 11,
        })
        print(f"  page 11 items                 : {len(r11.json().get('items', []))}")
    except httpx.HTTPStatusError as e:
        print(f"  page 11 response              : HTTP {e.response.status_code} — {e.response.json().get('message','')[:80]}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── Strategy B: GraphQL ───────────────────────────────────────────────────────
print("\n── B: GraphQL (issues only, first 100) ──")
try:
    t0 = time.perf_counter()
    query = """
    query($owner:String!, $repo:String!, $cursor:String) {
      repository(owner:$owner, name:$repo) {
        issues(first:100, after:$cursor, states:CLOSED, orderBy:{field:UPDATED_AT, direction:DESC}) {
          totalCount
          pageInfo { hasNextPage endCursor }
          nodes { number title updatedAt }
        }
        pullRequests(first:1, states:CLOSED) { totalCount }
      }
      rateLimit { remaining resetAt cost }
    }
    """
    r = post("https://api.github.com/graphql", json_body={
        "query": query,
        "variables": {"owner": OWNER, "repo": REPO, "cursor": None}
    })
    elapsed_b = time.perf_counter() - t0
    body = r.json()
    repo_data = body["data"]["repository"]
    issues_data = repo_data["issues"]
    pr_total = repo_data["pullRequests"]["totalCount"]
    rate = body["data"]["rateLimit"]
    page1_nodes = len(issues_data["nodes"])
    print(f"  total closed issues (exact)   : {issues_data['totalCount']}")
    print(f"  total closed PRs              : {pr_total}")
    print(f"  PR:issue ratio                : {pr_total / max(issues_data['totalCount'],1):.1f}:1")
    print(f"  page 1 nodes returned         : {page1_nodes}")
    print(f"  has next page                 : {issues_data['pageInfo']['hasNextPage']}")
    print(f"  graphql cost (points used)    : {rate['cost']}")
    print(f"  rate-limit remaining          : {rate['remaining']}")
    print(f"  elapsed                       : {elapsed_b:.2f}s")
    # walk 5 pages to measure real pace
    cursor = issues_data["pageInfo"]["endCursor"]
    total_collected = page1_nodes
    pages_walked = 1
    for _ in range(4):
        if not cursor:
            break
        r2 = post("https://api.github.com/graphql", json_body={
            "query": query,
            "variables": {"owner": OWNER, "repo": REPO, "cursor": cursor}
        })
        d2 = r2.json()["data"]
        nodes2 = d2["repository"]["issues"]["nodes"]
        total_collected += len(nodes2)
        cursor = d2["repository"]["issues"]["pageInfo"]["endCursor"]
        pages_walked += 1
    rate_after = r2.json()["data"]["rateLimit"]
    print(f"  after {pages_walked} pages: collected {total_collected} issues")
    print(f"  rate-limit remaining after    : {rate_after['remaining']}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── Strategy C: Label filter via REST ────────────────────────────────────────
print("\n── C: REST /issues with labels (bug, enhancement, question) ──")
try:
    for label in ["bug", "enhancement", "question"]:
        t0 = time.perf_counter()
        r = get(f"https://api.github.com/repos/{OWNER}/{REPO}/issues", params={
            "state": "closed", "labels": label,
            "per_page": 100, "page": 1,
            "sort": "updated", "direction": "desc",
        })
        batch = r.json()
        prs = sum(1 for i in batch if "pull_request" in i)
        elapsed_c = time.perf_counter() - t0
        print(f"  label={label!r:15} raw={len(batch):3}  prs={prs:3}  issues={len(batch)-prs:3}  ({elapsed_c:.1f}s)")
    # probe a Spark-specific label
    for label in ["SPARK", "core", "SQL"]:
        try:
            r = get(f"https://api.github.com/repos/{OWNER}/{REPO}/issues", params={
                "state": "closed", "labels": label,
                "per_page": 100, "page": 1,
                "sort": "updated", "direction": "desc",
            })
            batch = r.json()
            prs = sum(1 for i in batch if "pull_request" in i)
            print(f"  label={label!r:15} raw={len(batch):3}  prs={prs:3}  issues={len(batch)-prs:3}")
        except Exception as e:
            print(f"  label={label!r:15} ERROR: {e}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── Strategy D: Date-window filter via REST ───────────────────────────────────
print("\n── D: REST /issues with created: date windows ──")
# The REST /issues endpoint does NOT support since= for created_at,
# only for updated_at. We test `since` (updated_at floor) as a proxy,
# then probe a single year-wide created window via the Search API.
try:
    windows = [
        ("2024-01-01T00:00:00Z", "2024-12-31T23:59:59Z"),
        ("2023-01-01T00:00:00Z", "2023-12-31T23:59:59Z"),
        ("2022-01-01T00:00:00Z", "2022-12-31T23:59:59Z"),
    ]
    for since, until in windows:
        year = since[:4]
        r = get("https://api.github.com/search/issues", params={
            "q": f"repo:{OWNER}/{REPO} type:issue is:closed created:{since[:10]}..{until[:10]}",
            "per_page": 1,
        })
        data = r.json()
        count = data.get("total_count", "error")
        max_via_search = min(count, 1000) if isinstance(count, int) else "?"
        print(f"  year={year}  closed issues created: {count:>6}  "
              f"max collectible via Search: {max_via_search}")
        time.sleep(1.2)   # search rate: 10 req/min unauthenticated; 30/min authed
except Exception as e:
    print(f"  ERROR: {e}")

# ── Strategy E: Baseline reminder ─────────────────────────────────────────────
print("\n── E: REST /issues baseline (page 1 only, reminder) ──")
try:
    r = get(f"https://api.github.com/repos/{OWNER}/{REPO}/issues", params={
        "state": "closed", "per_page": 100, "page": 1,
        "sort": "updated", "direction": "desc",
    })
    batch = r.json()
    prs = sum(1 for i in batch if "pull_request" in i)
    print(f"  raw={len(batch)}  prs={prs}  issues={len(batch)-prs}  yield={((len(batch)-prs)/len(batch)*100):.1f}%")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "=" * 70)
print("Probe complete.")
print("=" * 70)
