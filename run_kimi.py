#!/usr/bin/env python3
"""
run_kimi.py — SSEF Phase 1, Kimi generation run
===============================================

Project : A Quality Evaluation Framework for Sinhala Story Generation (SSEF)
Version : 2.0  (supersedes 1.0)
Updated : 2026-08-19
Requires: prompts_v3.py in the same directory  (LENGTH_GATE must be (720,880)
          and MAX_ATTEMPTS must be 4 — verify before running)
Install : pip install requests

CHANGES FROM v1.0 — all from measurements taken 2026-08-19
----------------------------------------------------------
* MODEL_ID set to moonshotai/kimi-k2.6 (kimi-k3 REJECTED, see below)
* REQUEST_TIMEOUT 180 -> 900. Observed generations took 391-658 s; 180 s
  returned None and masked a timeout as a data error.
* Attempt log now written after EVERY attempt, not at the end. A crash or a
  closed laptop no longer destroys the rejection record.
* generation_id captured in the manifest. kimi-k2.6 is not a date-pinned
  identifier, so generation IDs + access date are the reproducibility record.
* max_tokens overridden locally to 9000 (see MAX_TOKENS below).
* Tokenizer threshold text corrected — the go/no-go heuristic is WITHDRAWN.
* Retries now cover 5xx, not only 429.

⚠️ THIS RUNS KIMI ONLY. Ten Kimi stories are NOT the corpus. The corpus is 30
   stories, three models, the SAME prompt, the SAME gate. Do not begin any
   analysis on a one-model partial set — RQ4 needs all three or none.

SETUP (Windows PowerShell)
--------------------------
    $env:OPENROUTER_API_KEY="sk-or-v1-..."
    python run_kimi.py generate 2>&1 | Tee-Object -FilePath run_log.txt

SETUP (macOS / Linux)
---------------------
    export OPENROUTER_API_KEY="sk-or-v1-..."
    python run_kimi.py generate 2>&1 | tee run_log.txt

NEVER hardcode the key. Never commit it. Never paste it into a chat.

STAGES
------
    python run_kimi.py models      # list Kimi model IDs on OpenRouter
    python run_kimi.py tokenize    # tokens/word — INFORMATIONAL ONLY
    python run_kimi.py refusal     # 10 truncated calls, pennies
    python run_kimi.py generate    # full run, 10 stories, ~2 h, ~$1.30
"""

import os
import sys
import json
import csv
import time
import hashlib
import datetime as dt
from pathlib import Path

import requests

from prompts_v3 import (
    TOPICS, build_prompt, GEN_PARAMS, LENGTH_GATE, MAX_ATTEMPTS,
    word_count, strip_preamble, accept,
)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"

# SELECTED 2026-08-19. Read from GET /api/v1/models on that date.
#
# REJECTED — moonshotai/kimi-k3: reasoning-by-default. 9,182 reasoning tokens
#   consumed the entire budget, content returned None, $0.135 per call. See
#   KT §2.2.
# NOTE — kimi-k2.6 also reasons and ignores reasoning:{"enabled":false}, but
#   it does return story content. Reasoning mode is therefore held constant
#   at "on" across all three models (DEC-027).
MODEL_ID = "moonshotai/kimi-k2.6"
MODEL_ID_READ_ON = "2026-08-19"

PROMPT_VARIANT = "A2"      # A2 = register NOT instructed. "A1" only for the
                           # pilot comparison.

# LOCAL max_tokens OVERRIDE — a documented deviation from GEN_PARAMS.
# Kimi generates acceptable stories with finish_reason "stop" at 9000.
# Gemini requires ~30000 because thinking tokens count against its ceiling.
# This is a NON-BINDING CEILING in both cases (DEC-021): length is enforced
# post-hoc by the word gate, so differing ceilings do not affect the output
# as long as finish_reason is never "length". WATCH FOR "length" IN THE LOG.
MAX_TOKENS = 9000

OUT = Path("outputs")
STORY_DIR = OUT / "stories" / "kimi"
RUN_ID = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# PARAMETERS — every one of these goes in the methodology
# ---------------------------------------------------------------------------
#
#   temperature   0.7    Fixed across all three models. Run-to-run variation
#                        of 60-80 words on an identical prompt is a direct
#                        consequence; this is what forced the gate widening.
#
#   max_tokens    9000   NON-BINDING CEILING (DEC-021), not a length control.
#
#   seed          fixed  OpenRouter passes it through only for providers that
#                        support it; it may be silently ignored. The Gemini
#                        route exposes no seed at all — a documented asymmetry.
#
#   length gate   720-880 words (800 +/-10%), 4 attempts (DEC-029).
#                 Evidence: n=7 across two models, mean 766, range 663-859.
#                 The previous +/-5% gate (760-840) accepted only 2 of 7 and
#                 was narrower than the run-to-run variance. REJECTIONS ARE
#                 DATA — they answer "did the model comply with the length
#                 instruction?" for RQ4.
#
#   top_p, frequency_penalty, presence_penalty: NOT SET. Provider defaults,
#   deliberately — setting them would add uncontrolled variables whose
#   meaning differs across providers.

REQUEST_TIMEOUT = 900      # was 180. Observed: 391-658 s per generation.
RETRY_ON_NETWORK = 3
SLEEP_BETWEEN = 2.0


def _headers():
    if not API_KEY:
        sys.exit(
            "OPENROUTER_API_KEY is not set.\n"
            "  PowerShell: $env:OPENROUTER_API_KEY='sk-or-v1-...'\n"
            "  bash      : export OPENROUTER_API_KEY='sk-or-v1-...'\n"
            "Do not hardcode it in this file."
        )
    return {"Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"}


def sha256(text):
    """Prompt hash. Proves at write-up time that every model received a
    byte-identical prompt — the whole basis of the RQ4 comparison."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _preflight():
    """Fail loudly on stale settings rather than silently running the wrong
    configuration for two hours."""
    problems = []
    if not MODEL_ID:
        problems.append("MODEL_ID is None")
    if LENGTH_GATE != (720, 880):
        problems.append(f"LENGTH_GATE is {LENGTH_GATE}, expected (720, 880) "
                        f"— prompts_v3.py is stale (DEC-029)")
    if MAX_ATTEMPTS < 4:
        problems.append(f"MAX_ATTEMPTS is {MAX_ATTEMPTS}, expected 4 "
                        f"— prompts_v3.py is stale (DEC-029)")
    if REQUEST_TIMEOUT < 900:
        problems.append(f"REQUEST_TIMEOUT is {REQUEST_TIMEOUT}, expected 900")
    if problems:
        print("PREFLIGHT FAILED:")
        for p in problems:
            print("  -", p)
        sys.exit("Fix the above before running. Do not proceed.")
    print(f"Preflight OK — {MODEL_ID}, gate {LENGTH_GATE}, "
          f"{MAX_ATTEMPTS} attempts, timeout {REQUEST_TIMEOUT}s\n")


# ---------------------------------------------------------------------------
# STAGE 0a — MODEL DISCOVERY
# ---------------------------------------------------------------------------

def list_models():
    r = requests.get(f"{BASE_URL}/models", headers=_headers(),
                     timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    models = r.json().get("data", [])
    hits = [m for m in models
            if "kimi" in m.get("id", "").lower()
            or "moonshot" in m.get("id", "").lower()]

    if not hits:
        print("No Kimi/Moonshot models found. First 40 available:")
        for m in models[:40]:
            print("  ", m.get("id"))
        return

    print(f"Kimi / Moonshot models on {dt.date.today().isoformat()}:\n")
    for m in hits:
        pr = m.get("pricing", {})
        alias = "  WARNING: moving alias, do not use" if m.get("id", "").startswith("~") else ""
        print(f"  ID      : {m.get('id')}{alias}")
        print(f"  Name    : {m.get('name')}")
        print(f"  Context : {m.get('context_length')}")
        print(f"  $/Mtok  : in {pr.get('prompt')}  out {pr.get('completion')}")
        print()


# ---------------------------------------------------------------------------
# STAGE 0b — TOKENIZER SCREEN — INFORMATIONAL ONLY
# ---------------------------------------------------------------------------

def tokenizer_screen():
    """Measure tokens-per-word on Sinhala via the API's own usage counter.

    ⚠️ THIS IS NOT A GO/NO-GO GATE. The v1.0 threshold heuristic
    (<=4 usable, 5-6 marginal, 7+ degenerate) was built from TWO data points
    (SinBERT 4.1, Llama-3.1-8B 11.0), presented as a gate, and FALSIFIED the
    same day: kimi-k2.6 measured 8.01 tok/word and then produced coherent
    Sinhala prose with correct dialogue punctuation and no repetition loop.
    The threshold claim is WITHDRAWN (KT §2.1).

    What this figure is still good for: budget estimation and choosing
    max_tokens. Model capability is decided by READING A GENERATED STORY.

    Reference points: gemini-3.5-flash 3.54 · SinBERT ~4.1 · kimi-k2.6 8.01
    (all wrote fine) · Llama-3.1-8B 11.0 (degenerate).

    PROBE TEXT: the researcher's own topic descriptions. Not generated.
    """
    _require_model()
    probes = list(TOPICS.items())[:3]
    ratios = []

    print(f"Tokenizer screen — {MODEL_ID}  (informational, NOT a gate)\n")
    for tid, text in probes:
        body = {"model": MODEL_ID,
                "messages": [{"role": "user", "content": text}],
                "max_tokens": 1}
        r = requests.post(f"{BASE_URL}/chat/completions", headers=_headers(),
                          json=body, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        ptok = r.json().get("usage", {}).get("prompt_tokens")
        wc = word_count(text)
        if not ptok:
            print(f"  {tid}: no usage returned — cannot measure.")
            return
        ratio = (ptok - 6) / max(1, wc)   # small chat-template allowance
        ratios.append(ratio)
        print(f"  {tid}: {ptok} tokens / {wc} words = {ratio:.2f} tok/word")
        time.sleep(SLEEP_BETWEEN)

    mean = sum(ratios) / len(ratios)
    rec = int(900 * mean * 1.5)
    print(f"\n  MEAN: {mean:.2f} tokens/word")
    print(f"  Suggested max_tokens floor: {max(MAX_TOKENS, rec)}")
    print(f"  Capability is NOT decided here. Generate a story and read it.")

    _write_json(OUT / f"tokenizer_screen_{RUN_ID}.json", {
        "run_id": RUN_ID, "model_id": MODEL_ID,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "per_probe": [{"topic_id": t, "ratio": round(r, 3)}
                      for (t, _), r in zip(probes, ratios)],
        "mean_tokens_per_word": round(mean, 3),
        "note": "Informational only. Threshold heuristic withdrawn, KT 2.1.",
    })


# ---------------------------------------------------------------------------
# STAGE 1 — REFUSAL SCREEN
# ---------------------------------------------------------------------------

def refusal_screen():
    """10 topics x 1 truncated call. Refusals surface in the first few tokens.

    CATCHES : refusal, safety preamble, non-Sinhala output, title/preamble.
    MISSES  : hedging — a model that complies but quietly sanitises the
              premise produces an off-topic story without announcing it. Only
              reading full-length output catches that.

    `refused` and `sinhala_script` are BLANK on purpose. You fill them by
    reading. An automated refusal classifier would be a second unvalidated
    instrument in a project that already has one.
    """
    _require_model()
    rows = []
    print(f"Refusal screen — {MODEL_ID}, variant {PROMPT_VARIANT}\n")
    for tid in TOPICS:
        prompt = build_prompt(tid, PROMPT_VARIANT)
        head = _call(prompt, max_tokens=200) or ""
        print(f"  {tid}\n    {head[:200]}\n", flush=True)
        rows.append({
            "run_id": RUN_ID, "model_id": MODEL_ID, "topic_id": tid,
            "variant": PROMPT_VARIANT, "prompt_sha": sha256(prompt),
            "head": head[:500],
            "refused": "", "sinhala_script": "", "notes": "",
        })
        _write_csv(OUT / f"refusal_screen_{RUN_ID}.csv", rows)
        time.sleep(SLEEP_BETWEEN)

    print(f"Saved outputs/refusal_screen_{RUN_ID}.csv — read every row.")
    print("Apply REFUSAL_RULE from prompts_v3.py before changing any topic.")


# ---------------------------------------------------------------------------
# STAGE 3 — GENERATION
# ---------------------------------------------------------------------------

def generate():
    """10 topics, length gate, up to MAX_ATTEMPTS each.

    SAVES (all written incrementally — a crash loses at most one attempt)
      outputs/stories/kimi/{topic_id}.txt   accepted story, UTF-8
      outputs/manifest_{run}.jsonl          one line per accepted story
      outputs/attempts_{run}.csv            EVERY attempt, rejections included
      outputs/raw/{topic}_a{n}.txt          raw response, pre-strip

    Raw responses are kept because strip_preamble() is a crude guard, not a
    parser. If it strips something it should not have, the raw file is the
    only way to find out.
    """
    _preflight()
    STORY_DIR.mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)

    attempts, manifest = [], []
    accepted_n = 0
    t_start = time.time()

    for idx, tid in enumerate(TOPICS, 1):
        prompt = build_prompt(tid, PROMPT_VARIANT)
        p_sha = sha256(prompt)
        story = None
        print(f"\n[{idx}/{len(TOPICS)}] {tid}", flush=True)

        for n in range(1, MAX_ATTEMPTS + 1):
            t0 = time.time()
            raw, meta = _call(prompt, max_tokens=MAX_TOKENS, with_meta=True)
            secs = round(time.time() - t0)

            if raw is None:
                attempts.append({
                    "run_id": RUN_ID, "topic_id": tid, "attempt": n,
                    "seconds": secs, "word_count": 0, "accepted": False,
                    "reason": "api_error", "prompt_sha": p_sha,
                    "finish_reason": meta.get("finish_reason", ""),
                    "completion_tokens": "", "generation_id": "",
                })
                _write_csv(OUT / f"attempts_{RUN_ID}.csv", attempts)
                print(f"   x  attempt {n}: no content ({secs}s)", flush=True)
                time.sleep(SLEEP_BETWEEN)
                continue

            (OUT / "raw" / f"{tid}_a{n}.txt").write_text(raw, encoding="utf-8")
            text = strip_preamble(raw)
            ok, wc, reason = accept(text)

            attempts.append({
                "run_id": RUN_ID, "topic_id": tid, "attempt": n,
                "seconds": secs, "word_count": wc, "accepted": ok,
                "reason": reason or "", "prompt_sha": p_sha,
                "finish_reason": meta.get("finish_reason", ""),
                "completion_tokens": meta.get("completion_tokens", ""),
                "generation_id": meta.get("generation_id", ""),
            })
            _write_csv(OUT / f"attempts_{RUN_ID}.csv", attempts)

            print(f"   {'OK ' if ok else 'REJ'} attempt {n}: {wc} words, "
                  f"{secs}s ({meta.get('finish_reason')})"
                  f"{'' if ok else ' — ' + str(reason)}", flush=True)

            # "length" means max_tokens truncated the story. That is a
            # PARAMETER fault, not a model fault — raise MAX_TOKENS and re-run.
            # Do NOT report it as a length-compliance failure.
            if meta.get("finish_reason") == "length":
                print("       WARNING: TRUNCATED by max_tokens. Raise "
                      "MAX_TOKENS and re-run this topic.", flush=True)

            if ok:
                story = text
                break
            time.sleep(SLEEP_BETWEEN)

        if story is None:
            print(f"   FAILED {tid}: no accepted generation in "
                  f"{MAX_ATTEMPTS} attempts. This is a FINDING about the "
                  f"model on this topic — report the rate.", flush=True)
            continue

        (STORY_DIR / f"{tid}.txt").write_text(story, encoding="utf-8")
        accepted_n += 1
        manifest.append({
            "run_id": RUN_ID,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "model_id": MODEL_ID, "model_id_read_on": MODEL_ID_READ_ON,
            "provider": "openrouter",
            "topic_id": tid, "variant": PROMPT_VARIANT,
            "prompt_sha256_16": p_sha,
            "temperature": GEN_PARAMS["temperature"],
            "max_tokens": MAX_TOKENS,
            "seed": GEN_PARAMS.get("seed"),
            "length_gate": list(LENGTH_GATE),
            "max_attempts": MAX_ATTEMPTS,
            "word_count": word_count(story),
            "generation_id": next((a["generation_id"] for a in reversed(attempts)
                                   if a["topic_id"] == tid and a["accepted"]), ""),
            "file": str(STORY_DIR / f"{tid}.txt"),
        })
        _write_jsonl(OUT / f"manifest_{RUN_ID}.jsonl", manifest)

    rej = sum(1 for a in attempts if not a["accepted"])
    wcs = [a["word_count"] for a in attempts if a["word_count"]]
    mins = round((time.time() - t_start) / 60)

    print("\n" + "=" * 55)
    print(f"  Accepted   : {accepted_n} / {len(TOPICS)}")
    print(f"  Attempts   : {len(attempts)}  (rejections: {rej})")
    if wcs:
        print(f"  Word range : {min(wcs)} - {max(wcs)}, "
              f"mean {round(sum(wcs)/len(wcs))}")
    print(f"  Elapsed    : {mins} min")
    print(f"\n  Append to methodology.md §10 with today's date: accepted "
          f"{accepted_n}/{len(TOPICS)}, {rej} rejections, word range, "
          f"rejection reasons, model ID and access date.")
    print(f"  Ten Kimi stories are ONE THIRD of a corpus, not a result.")


# ---------------------------------------------------------------------------
# PLUMBING
# ---------------------------------------------------------------------------

def _require_model():
    if not MODEL_ID:
        sys.exit("MODEL_ID is None. Run `python run_kimi.py models` first.")


def _call(prompt, max_tokens, with_meta=False):
    body = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": GEN_PARAMS["temperature"],
        "max_tokens": max_tokens,
    }
    if GEN_PARAMS.get("seed") is not None:
        body["seed"] = GEN_PARAMS["seed"]

    for attempt in range(RETRY_ON_NETWORK):
        try:
            r = requests.post(f"{BASE_URL}/chat/completions",
                              headers=_headers(), json=body,
                              timeout=REQUEST_TIMEOUT)
            # 5xx are transient server-capacity errors, not fatal.
            if r.status_code in (429, 500, 502, 503, 504):
                wait = 10 * (attempt + 1)
                print(f"       HTTP {r.status_code}, waiting {wait}s",
                      flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                print(f"       API error: {str(data['error'])[:200]}",
                      flush=True)
                return (None, {"finish_reason": "api_error"}) if with_meta else None
            choice = data["choices"][0]
            text = choice["message"].get("content")
            if with_meta:
                u = data.get("usage", {})
                return text, {
                    "finish_reason": choice.get("finish_reason"),
                    "completion_tokens": u.get("completion_tokens"),
                    "reasoning_tokens": u.get("completion_tokens_details", {})
                                         .get("reasoning_tokens"),
                    "cost": u.get("cost"),
                    "generation_id": data.get("id"),
                }
            return text
        except Exception as e:
            print(f"       network error ({e}); retry {attempt+1}", flush=True)
            time.sleep(5 * (attempt + 1))

    return (None, {"finish_reason": "network_fail"}) if with_meta else None


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                    encoding="utf-8")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    OUT.mkdir(parents=True, exist_ok=True)
    {"models": list_models,
     "tokenize": tokenizer_screen,
     "refusal": refusal_screen,
     "generate": generate}.get(cmd, lambda: print(__doc__))()
