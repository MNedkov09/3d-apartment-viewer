import os
from starlette.testclient import TestClient
from app import app

client = TestClient(app)

print("Testing /api/health...")
res = client.get("/api/health")
print("Health response:", res.status_code, res.json())

sample_path = os.path.join(os.path.dirname(__file__), "..", "public", "samples", "residential_plan.png")
print(f"\nTesting /api/detect with sample image at {sample_path}...")
with open(sample_path, "rb") as f:
    files = {"file": ("residential_plan.png", f, "image/png")}
    data = {"conf": 0.12, "use_hybrid": "true"}
    res = client.post("/api/detect", files=files, data=data)

print("Detect response status:", res.status_code)
if res.status_code == 200:
    data = res.json()
    print("Stats:", data.get("stats"))
    print("Wall edges count:", len(data.get("wall_edges", [])))
    print("Doors count:", len(data.get("doors", [])))
    print("Windows count:", len(data.get("windows", [])))
    print("Furniture count:", len(data.get("furniture", [])))
    if data.get("doors"):
        print("Sample door:", data["doors"][0])
    if data.get("wall_edges"):
        print("Sample wall edge:", data["wall_edges"][0])
    if data.get("furniture"):
        print("Sample furniture:", data["furniture"][0])
else:
    print("Error:", res.text)
