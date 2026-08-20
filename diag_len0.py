import os, requests, time, json
import prompts_v3

URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
       "gemini-3.5-flash:generateContent")
P = prompts_v3.build_prompt("N12_ITNightShift", "A2")

for i in range(3):
    t0 = time.time()
    r = requests.post(URL, params={"key": os.environ["GEMINI_API_KEY"]},
        json={"contents": [{"role": "user", "parts": [{"text": P}]}],
              "generationConfig": {"temperature": 0.7,
                                   "maxOutputTokens": 30000,
                                   "thinkingConfig": {"thinkingBudget": 0}}},
        timeout=900)
    d = r.json()
    txt = "".join(p.get("text","") for p in
                  d["candidates"][0]["content"]["parts"])
    wc = prompts_v3.word_count(prompts_v3.strip_preamble(txt))
    ok, _, why = prompts_v3.accept(prompts_v3.strip_preamble(txt))
    print(f"{i+1}: {wc} words, {round(time.time()-t0)}s → "
          f"{'ACCEPT' if ok else why}", flush=True)