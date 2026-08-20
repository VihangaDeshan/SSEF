import os, requests
import prompts_v3

P = prompts_v3.build_prompt("N04_Avurudu", "A2")

r = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
             "Content-Type": "application/json"},
    json={"model": "openai/gpt-5.6-sol",
          "messages": [{"role": "user", "content": P}],
          "temperature": 0.7, "max_tokens": 30000,
          "reasoning": {"enabled": True}},
    timeout=900)

t = r.json()["choices"][0]["message"]["content"]
open("probe_sol_2.txt", "w", encoding="utf-8").write(t)
print("words:", prompts_v3.word_count(prompts_v3.strip_preamble(t)))