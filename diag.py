import os, requests, json, time
import prompts_v3

p = prompts_v3.build_prompt("N01_CoastTrain", "A2")
t0 = time.time()
r = requests.post("https://openrouter.ai/api/v1/chat/completions",
    headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
             "Content-Type": "application/json"},
    json={"model": "deepseek/deepseek-v4-pro",
          "messages": [{"role": "user", "content": p}],
                    "temperature": 0.7, "max_tokens": 30000,
          "reasoning": {"enabled": True}},
    timeout=900)

d = r.json()
print("elapsed:", round(time.time()-t0), "s")
print("status :", r.status_code)
print("usage  :", json.dumps(d.get("usage"), indent=2))

ch = d["choices"][0]
print("finish :", ch.get("finish_reason"))
print("keys   :", list(ch["message"].keys()))
print("content:", (ch["message"].get("content") or "NONE")[:300])