import os, requests
r = requests.get("https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
for m in r.json()["data"]:
    if "deepseek" in m["id"].lower():
        p = m.get("pricing", {})
        print(f"{m['id']:45s} ctx {m.get('context_length'):>8}  "
              f"in {p.get('prompt')} out {p.get('completion')}")