#!/usr/bin/env python3
"""
run_gemini_or.py — SSEF Phase 1, Gemini generation VIA OPENROUTER
==================================================================

Project : A Quality Evaluation Framework for Sinhala Story Generation (SSEF)
Version : 2.0  (supersedes 1.0 — SEED BUG FIXED)
Created : 2026-08-19
Requires: prompts_v3.py with "seed": None  (the script enforces this)
Install : pip install requests

⚠️ WHAT WENT WRONG IN v1.0 — THE SEED BUG (DEC-033)
----------------------------------------------------
Run of 2026-08-19 produced 6/10 with these rejections:

    N02: 667, 667, 667, 700 words
    N03: 704, 704, 704, 887
    N09: 602, 616, 602, 602
    N14: 984, 682, 984, 984

IDENTICAL WORD COUNTS ON REPEATED ATTEMPTS = IDENTICAL STORIES.

Cause: GEN_PARAMS carried "seed": 20260819. OpenRouter passes the seed
through and Gemini honours it. Same prompt + same seed = same output, so
every retry regenerated the story that had already been rejected. A topic
that missed the gate on attempt 1 was stuck outside it for all four attempts.
The regeneration protocol was inert.

DeepSeek apparently IGNORES the seed — which is why its attempts varied and
it reached 10/10 under the same code. Two models, same parameter, different
behaviour: itself worth recording.

FIX: seed removed entirely. Preflight refuses to run if it is still set.

COST OF THE FIX — state this in the methodology:
    Generation is no longer bit-reproducible. An exact replication cannot be
    produced from the seed. What IS recorded for each story: the full prompt,
    its SHA-256 hash, model ID, access date, temperature, max_tokens,
    reasoning setting, length gate, attempt number, and the provider
    generation_id. This is weaker than a seed but it is honest, and the seed
    was never giving reproducibility across models anyway — DeepSeek ignored
    it, so the corpus was already only half-seeded.

WHY REASONING IS ON (DEC-027, confirmed on evidence)
-----------------------------------------------------
gemini-3.5-flash, identical prompt, n=4 per condition:
    reasoning on  : 714, 824, 822, 781   mean 785   3/4 inside the gate
    reasoning off : 644, 599, 542, 650   mean 609   0/4 inside the gate
No overlap. Mode must be identical across models or it confounds RQ4.
Gemini reasons by default, DeepSeek does not, so BOTH are set EXPLICITLY.

ACCESS ROUTE (DEC-032)
----------------------
OpenRouter for both models. The AI Studio free tier caps at 20 requests/day
(observed RPD 22/20, HTTP 429 from topic 5, run abandoned at 4 stories) and
cannot produce a single model's corpus at a ~40% acceptance rate. Unifying on
OpenRouter also REMOVES the access-route asymmetry rather than working around
it: same provider defaults, same parameter semantics, both models.

COST NOTE
---------
google/gemini-3.5-flash is $1.50/$9.00 per M tokens; reasoning tokens are
charged at the OUTPUT rate. Observed: $1.837 for 27 attempts.
DeepSeek v4-pro: $0.404 for 26 attempts. Gemini is ~4.5x more expensive.
google/gemini-3.7-flash is $0.375/$1.875 — roughly 4x cheaper than 3.5 —
and UNTESTED on this route. It returned HTTP 503 on AI Studio's free tier,
which says nothing about how OpenRouter routes it. Probing it is optional:

    python run_gemini_or.py probe --model google/gemini-3.7-flash

SETUP
-----
    $env:OPENROUTER_API_KEY="sk-or-v1-..."                     # PowerShell
    python run_gemini_or.py generate 2>&1 | Tee-Object -FilePath run_log_gem2.txt

STAGES
------
    python run_gemini_or.py probe      # ONE story — read it
    python run_gemini_or.py refusal    # 10 truncated calls
    python run_gemini_or.py generate   # full run, ~20 min
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

MODEL_ID = "google/gemini-3.5-flash"
MODEL_ID_READ_ON = "2026-08-19"
# Excluded: ~google/gemini-flash-latest, ~google/gemini-pro-latest (moving
# aliases destroy reproducibility); *-preview (can change mid-study);
# *-image, *-lite (wrong task or smaller); :batch (different latency profile).

# Override from the command line: --model google/gemini-3.7-flash
for _i, _a in enumerate(sys.argv):
    if _a == "--model" and _i + 1 < len(sys.argv):
        MODEL_ID = sys.argv[_i + 1]

PROMPT_VARIANT = "A2"      # A2 = register NOT instructed

# Explicit, not a default. See header.
REASONING = {"enabled": True}

# NON-BINDING CEILING (DEC-021). Gemini's reasoning tokens count against the
# output ceiling; 3,357 reasoning tokens truncated a story at 9,000. Length is
# enforced post-hoc by the word gate, so ceilings may differ across models
# PROVIDED finish_reason is never "length". WATCH THE LOG FOR IT.
MAX_TOKENS = 30000

# SEED IS DELIBERATELY NOT SENT. See the seed-bug note in the header.
# Do not reintroduce it without also varying it per attempt.
SEND_SEED = False

OUT = Path("outputs")
STORY_DIR = OUT / "stories" / "gemini"     # one folder per model; identical
                                           # filenames across folders make the
                                           # crossed design visible on disk
RAW_DIR = OUT / "raw_gemini"
RUN_ID = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

REQUEST_TIMEOUT = 900
RETRY_ON_NETWORK = 3
SLEEP_BETWEEN = 2.0


def _headers():
    if not API_KEY:
        sys.exit(
            "OPENROUTER_API_KEY is not set.\n"
            "  PowerShell: $env:OPENROUTER_API_KEY='sk-or-v1-...'\n"
            "Do not hardcode it in this file."
        )
    return {"Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"}


def sha256(text):
    """Prompt hash. Proves at write-up time that every model received a
    byte-identical prompt — the whole basis of the RQ4 comparison. With the
    seed gone, this hash plus the generation_id is the provenance record."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _preflight():
    """Fail loudly on stale settings. Every check here exists because the
    corresponding mistake actually happened on 2026-08-19."""
    problems = []
    if LENGTH_GATE != (720, 880):
        problems.append(f"LENGTH_GATE is {LENGTH_GATE}, expected (720, 880) "
                        f"— prompts_v3.py is stale (DEC-029)")
    if MAX_ATTEMPTS < 4:
        problems.append(f"MAX_ATTEMPTS is {MAX_ATTEMPTS}, expected 4 (DEC-029)")
    if not REASONING.get("enabled"):
        problems.append("REASONING is not enabled — DEC-027 requires all "
                        "models to run in reasoning mode")
    if MAX_TOKENS < 20000:
        problems.append(f"MAX_TOKENS is {MAX_TOKENS}; reasoning tokens count "
                        f"against this ceiling and will truncate the story")
    if GEN_PARAMS.get("seed") is not None and SEND_SEED:
        problems.append("A seed is set AND being sent. This makes every "
                        "retry identical and defeats the length gate "
                        "(DEC-033). Set \"seed\": None in prompts_v3.py")

    stale = list(STORY_DIR.glob("*.txt"))
    if stale:
        problems.append(
            f"{len(stale)} story file(s) already in {STORY_DIR}. Those came "
            f"from a run with a different access route or a broken retry "
            f"protocol. Delete them:  Remove-Item {STORY_DIR}\\*.txt")

    if problems:
        print("PREFLIGHT FAILED:")
        for p in problems:
            print("  -", p)
        sys.exit("Fix the above before running. Do not proceed.")
    print(f"Preflight OK — {MODEL_ID}, gate {LENGTH_GATE}, "
          f"{MAX_ATTEMPTS} attempts, reasoning ON, max_tokens {MAX_TOKENS}, "
          f"seed OFF\n")


# ---------------------------------------------------------------------------
# CORE CALL
# ---------------------------------------------------------------------------

def _call(prompt, max_tokens, with_meta=False):
    body = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": GEN_PARAMS["temperature"],
        "max_tokens": max_tokens,
        "reasoning": REASONING,
    }
    # No seed. Sending one makes retries identical — see DEC-033.

    for attempt in range(RETRY_ON_NETWORK):
        try:
            r = requests.post(f"{BASE_URL}/chat/completions",
                              headers=_headers(), json=body,
                              timeout=REQUEST_TIMEOUT)
            if r.status_code in (429, 500, 502, 503, 504):
                wait = 10 * (attempt + 1)
                print(f"       HTTP {r.status_code}, waiting {wait}s", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                print(f"       API error: {str(data['error'])[:200]}", flush=True)
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


# ---------------------------------------------------------------------------
# PROBE
# ---------------------------------------------------------------------------

def probe(topic_id="N12_ITNightShift"):
    """ONE story, same topic every other model was probed on.

    ⚠️ DEMONSTRATION, NOT EVALUATION. One story per model shows the pipeline
    produces Sinhala prose. It shows NOTHING about which model is better.
    """
    prompt = build_prompt(topic_id, PROMPT_VARIANT)
    t0 = time.time()
    txt, meta = _call(prompt, MAX_TOKENS, with_meta=True)
    secs = round(time.time() - t0)

    print(f"model      : {MODEL_ID}")
    print(f"elapsed    : {secs} s")
    print(f"finish     : {meta.get('finish_reason')}")
    print(f"reasoning  : {meta.get('reasoning_tokens')}   <-- >0, NOT the "
          f"whole budget")
    print(f"completion : {meta.get('completion_tokens')}")
    print(f"cost       : ${meta.get('cost')}")

    if not txt:
        print("\ncontent: None. If reasoning_tokens equals max_tokens this is "
              "the Kimi failure mode (DEC-030) — STOP, raising the ceiling "
              "makes it worse.")
        return None

    clean = strip_preamble(txt)
    w = clean.split()
    ok, wc, reason = accept(clean)
    print(f"words      : {wc}   gate {LENGTH_GATE} -> "
          f"{'ACCEPT' if ok else 'REJECT: ' + str(reason)}")
    print(f"unique     : {round(len(set(w))/max(1,len(w)), 3)}   "
          f"(<0.25 = repetition loop)")
    print(f"quotes     : {clean.count(chr(34)) + clean.count(chr(8220)) + clean.count(chr(8221))}"
          f"   (0 = no dialogue, DARC has no input)")

    (OUT / "probe").mkdir(parents=True, exist_ok=True)
    safe = MODEL_ID.replace("/", "_")
    (OUT / "probe" / f"{safe}_{topic_id}.txt").write_text(clean, encoding="utf-8")
    print("\n" + "=" * 60 + "\n")
    print(clean)
    print("\n" + "=" * 60)
    print("\nNOW READ IT. The checks above are proxies. Whether the Sinhala "
          "is good is a native-speaker judgement and no script can make it.")
    return clean


# ---------------------------------------------------------------------------
# REFUSAL SCREEN
# ---------------------------------------------------------------------------

def refusal_screen():
    """10 topics, truncated calls.

    CATCHES : refusal, safety preamble, non-Sinhala output, title/preamble.
    MISSES  : hedging — a model that complies but sanitises the premise
              produces an off-topic story without announcing it. Only reading
              full-length output catches that.
    """
    rows = []
    print(f"Refusal screen — {MODEL_ID}, variant {PROMPT_VARIANT}\n")
    for tid in TOPICS:
        prompt = build_prompt(tid, PROMPT_VARIANT)
        head = _call(prompt, 300) or ""
        print(f"  {tid}\n    {head[:200]}\n", flush=True)
        rows.append({
            "run_id": RUN_ID, "model_id": MODEL_ID, "topic_id": tid,
            "variant": PROMPT_VARIANT, "prompt_sha": sha256(prompt),
            "head": head[:500],
            "refused": "", "sinhala_script": "", "notes": "",
        })
        _write_csv(OUT / f"refusal_gemini_{RUN_ID}.csv", rows)
        time.sleep(SLEEP_BETWEEN)

    print(f"Saved outputs/refusal_gemini_{RUN_ID}.csv — read every row.")


# ---------------------------------------------------------------------------
# GENERATION
# ---------------------------------------------------------------------------

def generate():
    """10 topics, length gate, up to MAX_ATTEMPTS each.

    SAVES (incrementally — a crash loses at most one attempt)
      outputs/stories/gemini/{topic_id}.txt   accepted story, UTF-8
      outputs/manifest_gemini_{run}.jsonl     one line per accepted story
      outputs/attempts_gemini_{run}.csv       EVERY attempt, rejections too
      outputs/raw_gemini/{topic}_a{n}.txt     raw response, pre-strip

    A DUPLICATE-OUTPUT WARNING fires if two attempts on the same topic return
    the same word count. That is the signature of the seed bug (DEC-033) and
    means the retry protocol is inert — stop and investigate rather than
    letting the run finish.
    """
    _preflight()
    STORY_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    attempts, manifest = [], []
    accepted_n = 0
    dup_warnings = 0
    t_start = time.time()

    for idx, tid in enumerate(TOPICS, 1):
        prompt = build_prompt(tid, PROMPT_VARIANT)
        p_sha = sha256(prompt)
        story = None
        seen_counts = []
        print(f"\n[{idx}/{len(TOPICS)}] {tid}", flush=True)

        for n in range(1, MAX_ATTEMPTS + 1):
            t0 = time.time()
            raw, meta = _call(prompt, MAX_TOKENS, with_meta=True)
            secs = round(time.time() - t0)

            if raw is None:
                attempts.append({
                    "run_id": RUN_ID, "topic_id": tid, "attempt": n,
                    "seconds": secs, "word_count": 0, "accepted": False,
                    "reason": "no_content", "prompt_sha": p_sha,
                    "finish_reason": meta.get("finish_reason", ""),
                    "reasoning_tokens": meta.get("reasoning_tokens", ""),
                    "completion_tokens": meta.get("completion_tokens", ""),
                    "cost": meta.get("cost", ""), "generation_id": "",
                })
                _write_csv(OUT / f"attempts_gemini_{RUN_ID}.csv", attempts)
                print(f"   x  attempt {n}: no content "
                      f"({meta.get('finish_reason')}, {secs}s)", flush=True)
                time.sleep(SLEEP_BETWEEN)
                continue

            (RAW_DIR / f"{tid}_a{n}.txt").write_text(raw, encoding="utf-8")
            text = strip_preamble(raw)
            ok, wc, reason = accept(text)

            if wc in seen_counts:
                dup_warnings += 1
                print(f"       WARNING: word count {wc} already seen on this "
                      f"topic. Retries may be returning IDENTICAL output "
                      f"(DEC-033). Check that no seed is being sent.",
                      flush=True)
            seen_counts.append(wc)

            attempts.append({
                "run_id": RUN_ID, "topic_id": tid, "attempt": n,
                "seconds": secs, "word_count": wc, "accepted": ok,
                "reason": reason or "", "prompt_sha": p_sha,
                "finish_reason": meta.get("finish_reason", ""),
                "reasoning_tokens": meta.get("reasoning_tokens", ""),
                "completion_tokens": meta.get("completion_tokens", ""),
                "cost": meta.get("cost", ""),
                "generation_id": meta.get("generation_id", ""),
            })
            _write_csv(OUT / f"attempts_gemini_{RUN_ID}.csv", attempts)

            print(f"   {'OK ' if ok else 'REJ'} attempt {n}: {wc} words, "
                  f"{secs}s ({meta.get('finish_reason')})"
                  f"{'' if ok else ' — ' + str(reason)}", flush=True)

            if meta.get("finish_reason") == "length":
                print("       WARNING: TRUNCATED by max_tokens. PARAMETER "
                      "fault, not a model fault — raise MAX_TOKENS and re-run "
                      "this topic. Do not report as a length failure.",
                      flush=True)

            if ok:
                story = text
                break
            time.sleep(SLEEP_BETWEEN)

        if story is None:
            print(f"   FAILED {tid}: no accepted generation in {MAX_ATTEMPTS} "
                  f"attempts. FINDING — report the rate.", flush=True)
            continue

        (STORY_DIR / f"{tid}.txt").write_text(story, encoding="utf-8")
        accepted_n += 1
        manifest.append({
            "run_id": RUN_ID,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "model_id": MODEL_ID, "model_id_read_on": MODEL_ID_READ_ON,
            "provider": "openrouter",
            "reasoning_enabled": True,
            "topic_id": tid, "variant": PROMPT_VARIANT,
            "prompt_sha256_16": p_sha,
            "temperature": GEN_PARAMS["temperature"],
            "max_tokens": MAX_TOKENS,
            "seed": None,                     # deliberately not sent, DEC-033
            "length_gate": list(LENGTH_GATE),
            "max_attempts": MAX_ATTEMPTS,
            "word_count": word_count(story),
            "generation_id": next((a["generation_id"] for a in reversed(attempts)
                                   if a["topic_id"] == tid and a["accepted"]), ""),
            "file": str(STORY_DIR / f"{tid}.txt"),
        })
        _write_jsonl(OUT / f"manifest_gemini_{RUN_ID}.jsonl", manifest)

    rej = sum(1 for a in attempts if not a["accepted"])
    wcs = [a["word_count"] for a in attempts if a["word_count"]]
    costs = [a["cost"] for a in attempts if isinstance(a.get("cost"), (int, float))]
    mins = round((time.time() - t_start) / 60)

    print("\n" + "=" * 55)
    print(f"  Model      : {MODEL_ID}")
    print(f"  Accepted   : {accepted_n} / {len(TOPICS)}")
    print(f"  Attempts   : {len(attempts)}  (rejections: {rej})")
    if wcs:
        print(f"  Word range : {min(wcs)} - {max(wcs)}, "
              f"mean {round(sum(wcs)/len(wcs))}")
    if costs:
        print(f"  Total cost : ${round(sum(costs), 3)}")
    print(f"  Elapsed    : {mins} min")
    if dup_warnings:
        print(f"\n  ⚠️ {dup_warnings} DUPLICATE WORD COUNTS. Retries may be "
              f"returning identical output. Investigate before trusting "
              f"this run (DEC-033).")
    print(f"\n  Append to methodology.md §10 with today's date: model ID, "
          f"access date, accepted {accepted_n}/{len(TOPICS)}, {rej} "
          f"rejections, word range, rejection reasons, cost.")
    print(f"  DeepSeek reference: 10/10, 26 attempts, 16 rejections, "
          f"489-1035, mean 768, $0.404, 42 min.")


# ---------------------------------------------------------------------------
# PLUMBING
# ---------------------------------------------------------------------------

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


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    OUT.mkdir(parents=True, exist_ok=True)
    {"probe": probe,
     "refusal": refusal_screen,
     "generate": generate}.get(cmd, lambda: print(__doc__))()
