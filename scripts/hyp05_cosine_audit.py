"""Audit the cosine similarity between each hyp-05 hypothesis root_cause
and every retrieved incident title, to diagnose the sibling-mismatch failure."""
import math
from app.db.session import SessionLocal
from app.db.models import Incident
from app.services.embedding_service import EmbeddingService

RETRIEVED_IDS = [
    "5dba5df8-402b-449c-a95e-a86812d3c8d1",  # gold: Duplicated OperationID
    "1b2c535b-6d2e-49ea-8c42-7a9423637df1",
    "e1f61e5b-a504-4e89-9a78-aa0950a0d4b2",
    "539b17a1-2ae9-4f68-9f1f-c1e3414907f2",
    "de1208c6-b280-42f1-bee4-83e003a8b9a7",
]

HYPOTHESES = [
    (1, True,  "Improper use of the `@router.api_route()` method for defining multiple HTTP methods on the same endpoint."),
    (2, True,  "Lack of clear documentation on the proper methods to use for defining routes in FastAPI."),
    (3, True,  "The design of FastAPI allowing multiple methods in a single route definition without adequate restrictions."),
    (4, True,  "Inconsistent handling of operation IDs when using semi-internal methods like `api_route` and `add_api_route`."),
    (5, True,  "Potential oversight in the implementation of the FastAPI framework regarding route definitions and operation ID uniqueness."),
]

def cosine(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot/(na*nb) if na and nb else 0.0

db = SessionLocal()
emb = EmbeddingService()

incidents = [(iid, db.get(Incident, iid)) for iid in RETRIEVED_IDS]
title_vecs = [(iid, inc, emb.embed_text(inc.title)) for iid, inc in incidents]

print(f"{'Inc#':<5} {'Title':<60}")
for i, (iid, inc, _) in enumerate(title_vecs, 1):
    gold = " <-- GOLD" if iid == "5dba5df8-402b-449c-a95e-a86812d3c8d1" else ""
    print(f"  {i}  {inc.title[:58]}{gold}")

print()

for rank, is_match, rc in HYPOTHESES:
    rc_vec = emb.embed_text(rc)
    scores = [(cosine(rc_vec, tvec), iid, inc.title) for iid, inc, tvec in title_vecs]
    scores.sort(reverse=True)

    best_iid   = scores[0][1]
    best_score = scores[0][0]
    gold_score = next(s for s,iid,_ in scores if iid == "5dba5df8-402b-449c-a95e-a86812d3c8d1")
    gap        = best_score - gold_score
    chose_gold = best_iid == "5dba5df8-402b-449c-a95e-a86812d3c8d1"

    print(f"rank{rank} (match={is_match})")
    print(f"  rc: {rc[:80]}")
    for score, iid, title in scores:
        marker = ""
        if iid == "5dba5df8-402b-449c-a95e-a86812d3c8d1":
            marker = " <-- GOLD"
        if iid == best_iid and not chose_gold:
            marker += " <-- CHOSEN (wrong)"
        elif iid == best_iid:
            marker += " <-- CHOSEN (correct)"
        print(f"  {score:.4f}  {title[:60]}{marker}")
    print(f"  gap (best - gold): {gap:+.4f}  chose_gold={chose_gold}")
    print()
