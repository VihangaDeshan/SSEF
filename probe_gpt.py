import os, requests, time
import prompts_v3

P = prompts_v3.build_prompt("N12_ITNightShift", "A2")

for model in ["openai/gpt-5.6-luna", "openai/gpt-5.6-terra", "openai/gpt-5.6-sol"]:
    t0 = time.time()
    r = requests.post("https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                 "Content-Type": "application/json"},
        json={"model": model,
              "messages": [{"role": "user", "content": P}],
              "temperature": 0.7, "max_tokens": 30000,
              "reasoning": {"enabled": True}},
        timeout=900)
    d = r.json()
    if "error" in d:
        print(f"{model:28s} ERROR: {str(d['error'])[:150]}\n"); continue
    ch = d["choices"][0]; txt = ch["message"].get("content")
    u = d.get("usage", {})
    rt = u.get("completion_tokens_details", {}).get("reasoning_tokens")
    wc = prompts_v3.word_count(prompts_v3.strip_preamble(txt)) if txt else 0
    q  = (txt or "").count('"') + (txt or "").count("\u201c") + (txt or "").count("\u201d")
    print(f"{model:28s} {round(time.time()-t0):>4}s  finish={ch.get('finish_reason'):8s} "
          f"reasoning={rt}  words={wc}  quotes={q}  ${u.get('cost')}")
    if txt:
        open(f"probe_{model.split('/')[1]}.txt", "w", encoding="utf-8").write(txt)