#!/usr/bin/env python3
"""
diag_gemini.py — does Gemini's thinkingBudget actually work?
============================================================

WHY THIS MATTERS
----------------
DEC-027 ("all three models run in reasoning mode") was forced by Kimi, which
ignored every attempt to disable reasoning. Kimi is now rejected (DEC-030), so
that constraint is gone and the reasoning question is REOPENED.

If thinkingBudget=0 works on Gemini, non-reasoning becomes available: faster,
cheaper, more predictable, and a much wider field of candidate third models.
If it does not work, all-reasoning stands and the third model must also reason.

WHAT THIS TESTS
---------------
Three settings, same prompt, same topic, same temperature:
    A  thinkingBudget omitted   (provider default — the current baseline)
    B  thinkingBudget = 0       (attempt to disable)
    C  thinkingBudget = 512     (small explicit budget)

READ thoughtsTokenCount. That single number is the answer.
    A ~5000, B 0 or absent  -> the parameter WORKS
    A ~5000, B ~5000        -> IGNORED, same as Kimi

⚠️ This is n=1 per setting. It answers "does the parameter take effect",
   which is a yes/no mechanism question. It does NOT establish word-count
   behaviour under the new setting — that needs its own samples afterwards.

RUN
---
    $env:GEMINI_API_KEY="..."          # PowerShell
    python diag_gemini.py

    ~2-6 minutes total, a few cents.
"""

import os
import sys
import json
import time

import requests

import prompts_v3

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    sys.exit("GEMINI_API_KEY is not set.\n"
             "  PowerShell: $env:GEMINI_API_KEY='...'\n"
             "  bash      : export GEMINI_API_KEY='...'")

MODEL = "gemini-3.5-flash"      # responds reliably; 3.7-flash was 503-ing
URL = (f"https://generativelanguage.googleapis.com/v1beta/models/"
       f"{MODEL}:generateContent")

TOPIC = "N12_ITNightShift"      # same topic used for every earlier probe
PROMPT = prompts_v3.build_prompt(TOPIC, "A2")

TESTS = [
    ("A  default (no thinkingConfig)", None),
    ("B  thinkingBudget = 0",          0),
    ("C  thinkingBudget = 512",        512),
]

results = []

print(f"model  : {MODEL}")
print(f"topic  : {TOPIC}")
print(f"prompt : {len(PROMPT.split())} words\n")

for label, budget in TESTS:
    cfg = {"temperature": 0.7, "maxOutputTokens": 30000}
    if budget is not None:
        cfg["thinkingConfig"] = {"thinkingBudget": budget}

    body = {"contents": [{"role": "user", "parts": [{"text": PROMPT}]}],
            "generationConfig": cfg}

    print(f"--- {label} ---", flush=True)
    t0 = time.time()
    try:
        r = requests.post(URL, params={"key": API_KEY}, json=body, timeout=900)
        d = r.json()
    except Exception as e:
        print(f"    request failed: {e}\n", flush=True)
        results.append({"test": label, "error": str(e)[:200]})
        continue

    secs = round(time.time() - t0)

    # An error here is itself informative: some models reject thinkingBudget
    # outright rather than ignoring it. That is a different finding.
    if "error" in d:
        msg = str(d["error"])[:300]
        print(f"    {secs}s  API ERROR: {msg}\n", flush=True)
        results.append({"test": label, "seconds": secs, "api_error": msg})
        continue

    cands = d.get("candidates", [])
    if not cands:
        pf = d.get("promptFeedback", {})
        print(f"    {secs}s  no candidate. blockReason="
              f"{pf.get('blockReason')}\n", flush=True)
        results.append({"test": label, "seconds": secs,
                        "block": pf.get("blockReason")})
        continue

    c = cands[0]
    u = d.get("usageMetadata", {})
    thoughts = u.get("thoughtsTokenCount")
    parts = c.get("content", {}).get("parts", [])
    txt = "".join(p.get("text", "") for p in parts)
    clean = prompts_v3.strip_preamble(txt) if txt else ""
    wc = prompts_v3.word_count(clean) if clean else 0
    quotes = clean.count('"') + clean.count("\u201c") + clean.count("\u201d")
    first = clean.split("\n")[0][:90] if clean else "(no content)"

    row = {
        "test": label, "seconds": secs,
        "finish": c.get("finishReason"),
        "thoughts_tokens": thoughts,
        "output_tokens": u.get("candidatesTokenCount"),
        "words": wc, "quote_chars": quotes,
        "first_line": first,
    }
    results.append(row)

    print(f"    seconds  : {secs}")
    print(f"    finish   : {row['finish']}")
    print(f"    THOUGHTS : {thoughts}      <-- the answer")
    print(f"    output   : {row['output_tokens']} tokens")
    print(f"    words    : {wc}   gate {prompts_v3.LENGTH_GATE}")
    print(f"    quotes   : {quotes}   (0 = no dialogue, DARC has no input)")
    print(f"    starts   : {first}")
    print(flush=True)

    if clean:
        fn = f"gemini_think_{'default' if budget is None else budget}.txt"
        open(fn, "w", encoding="utf-8").write(clean)
        print(f"    saved    : {fn}\n", flush=True)

# ---------------------------------------------------------------------------

json.dump(results, open("diag_gemini_results.json", "w"),
          ensure_ascii=False, indent=2)

print("=" * 60)
print("VERDICT")
print("=" * 60)

base = next((r for r in results if r["test"].startswith("A")), {})
zero = next((r for r in results if r["test"].startswith("B")), {})
bt, zt = base.get("thoughts_tokens"), zero.get("thoughts_tokens")

if bt and (zt in (0, None)) and zero.get("words", 0) > 0:
    print("thinkingBudget WORKS. Non-reasoning generation is available.")
    print("  -> DEC-027 can be revisited. A non-reasoning corpus is possible,")
    print("     and the third-model field opens up considerably.")
    print("  -> Word-count behaviour under this setting is NOT yet known.")
    print("     Run 3 samples before changing the gate or the design.")
elif bt and zt and zt > 100:
    print("thinkingBudget IGNORED — same failure mode as Kimi.")
    print("  -> DEC-027 stands: all three models run in reasoning mode.")
    print("  -> The third model must also be a reasoning model.")
elif zero.get("api_error"):
    print("thinkingBudget REJECTED by the API (see the error above).")
    print("  -> Not the same as being ignored. Check the parameter name")
    print("     against current Google documentation before concluding.")
else:
    print("Inconclusive. Read the rows above and do not guess.")

print("\nRemember: n=1 per setting. This answers whether the parameter takes")
print("effect. It establishes nothing about output quality or length.")
print("Saved: diag_gemini_results.json")
