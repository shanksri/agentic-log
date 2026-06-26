from app.db.session import SessionLocal
from app.db.models import Incident

db = SessionLocal()
rows = [
    ("a9a17361-6af2-4ab1-97b4-e7f1866e6eca", "hyp-03"),
    ("b093992f-7be2-405a-9127-ea3ee4c5c382", "hyp-04"),
    ("0c4aacbc-3946-4b03-ba85-920017ba8b48", "hyp-07"),
]
for iid, label in rows:
    inc = db.get(Incident, iid)
    print(label, "|", inc.title)
    for s in inc.symptoms:
        print("  S:", s.text)
    print("  R:", inc.resolution_summary)
