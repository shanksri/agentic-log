"""Analysis-only: retrieval behavior study over the 8k mixed corpus."""
from __future__ import annotations
import json
from collections import Counter
from app.db.session import SessionLocal
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.search import IncidentSearchService
from app.services.confidence import classify_confidence

QUERIES = {
 "A": ["triggerer not starting","JS heap out of memory","OutOfMemoryError",
       "NullPointerException during startup","broker not starting","leader election failed",
       "connection refused","timeout waiting for response"],
 "B": ["java.lang.OutOfMemoryError","java.net.SocketTimeoutException","HTTP 500 internal server error",
       "Segmentation fault","panic: runtime error","OOMKilled","disk pressure","node not ready"],
 "C": ["background scheduler refuses to launch","worker process consumes all available memory",
       "service becomes unavailable after deployment","application crashes after upgrade",
       "high memory consumption causing instability","unexpected restart loop",
       "database requests becoming extremely slow","system unable to elect a leader"],
 "D": ["kafka broker crash","zookeeper connection issue","kubernetes pod restart loop",
       "deployment rollback","helm upgrade failure","ingress routing issue",
       "high CPU utilization","memory leak"],
 "E": ["broker startup failure","replica synchronization problem","consumer lag issue",
       "controller failover","cluster instability","data replication error",
       "partition reassignment problem","ISR shrink event"],
 "F": ["triggerer","memory","deployment","timeout","error","failure","issue","bug"],
}
SUBSET = ["triggerer not starting","leader election failed","java.lang.OutOfMemoryError",
          "panic: runtime error","background scheduler refuses to launch",
          "database requests becoming extremely slow","kafka broker crash","memory leak",
          "broker startup failure","ISR shrink event","memory","timeout"]

def lbl(inc):
    if inc.source_type=="github": return f"gh:{inc.repo}"
    return f"jira:{(inc.source_metadata or {}).get('project_key','?')}"

def main():
    db=SessionLocal()
    svc=IncidentSearchService(db, embedding_service=EmbeddingService(), llm_service=LLMService())
    out={}
    print("="*70); print("DENSE PASS (expand=F rerank=F) — all 48 queries"); print("="*70)
    for setname,qs in QUERIES.items():
        print(f"\n--- SET {setname} ---")
        for q in qs:
            res=svc.retrieve(q, limit=10, expand=False, rerank=False)
            if not res:
                print(f"  {q[:34]:34s} | NO RESULTS"); continue
            s=[r.similarity_score for r in res]
            srcs=[r.incident.source_type for r in res]
            top=res[0].incident
            jira=sum(1 for x in srcs if x=="jira")
            conf=classify_confidence(s[0])
            print(f"  {q[:34]:34s} | top1={s[0]:.3f} {conf:6s} | {lbl(top):16s} | "
                  f"s@5={sum(s[:5])/len(s[:5]):.3f} s@10={s[-1]:.3f} | jira{jira}/10 | "
                  f"{top.title[:30]}")
            out[q]={"top1":s[0],"conf":conf,"top_label":lbl(top),"top_title":top.title,
                    "jira_in_top10":jira,"s5":sum(s[:5])/len(s[:5]),"s10":s[-1],
                    "top10_src":dict(Counter(srcs))}
    # 3-config subset
    print("\n"+"="*70); print("3-CONFIG MATRIX — diagnostic subset (top-5)"); print("="*70)
    for q in SUBSET:
        print(f"\n### {q}")
        for tag,(ex,rr) in [("dense",(False,False)),("expand",(True,False)),("rerank",(True,True))]:
            res=svc.retrieve(q, limit=5, expand=ex, rerank=rr)
            s=[r.similarity_score for r in res]
            line=", ".join(f"{lbl(r.incident)}({r.similarity_score:.2f})" for r in res)
            c=classify_confidence(s[0]) if s else "LOW"
            print(f"  {tag:7s} top1={s[0] if s else 0:.3f} {c:6s}: {line}")
    db.close()
    print("\nJSON_DIGEST="+json.dumps(out))

if __name__=="__main__":
    main()
