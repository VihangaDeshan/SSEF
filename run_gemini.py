#!/usr/bin/env python3
"""
run_gemini.py — SSEF Phase 1, Gemini generation run
===================================================

Project : A Quality Evaluation Framework for Sinhala Story Generation (SSEF)
Version : 2.0  (supersedes 1.0)
Updated : 2026-08-19
Requires: prompts_v3.py in the same directory (gate (720,880), attempts 4)
Install : pip install requests

⚠️ TWO BUGS FIXED FROM v1.0 — BOTH WOULD HAVE RUINED THE RUN
-------------------------------------------------------------
1. THINKING_BUDGET was left at 0 after the 2026-08-19 diagnostic. That
   setting produced 644, 599, 542, 650 words — 0 of 4 inside the gate. It is
   reverted to None (provider default = reasoning ON).
2. max_tokens was read from GEN_PARAMS, which is now 9000 (a Kimi-era value).
   Gemini's thinking tokens count against its output ceiling: 3,357 thinking
   tokens truncated a story at 9000 with finish_reason MAX_TOKENS. A LOCAL
   override of 30000 is used instead.

WHY REASONING IS ON (DEC-027, confirmed on evidence)
-----------------------------------------------------
gemini-3.5-flash, identical prompt, n=4 per condition:
    thinkingBudget default : 714, 824, 822, 781   mean 785   3/4 in gate
    thinkingBudget = 0     : 644, 599, 542, 650   mean 609   0/4 in gate
No overlap. Disabling reasoning shortened output by roughly 22%. Mode must
therefore be identical across models or it confounds RQ4.

Retained for later analysis: the four non-reasoning stories. Whether
disabling reasoning also changes register or coherence is UNTESTED.

ACCESS ROUTE ASYMMETRY — a documented limitation
-------------------------------------------------
Gemini runs on Google AI Studio; DeepSeek runs on OpenRouter. Different
provider defaults, different hidden system instructions, different parameter
semantics. A component of any RQ4 difference is attributable to the access
path rather than the model. Two specific asymmetries:
  - NO SEED on this route. OpenRouter accepts one; AI Studio does not.
  - Safety settings NOT modified. Provider defaults on both sides. Safety
    blocks are logged as refusal data rather than routed around.

MODEL SELECTION
---------------
gemini-3.5-flash, read 2026-08-19. Rule applied to every vendor: the most
recent generally-available text model that is serving traffic, excluding
preview builds and moving "-latest" aliases.
REJECTED — gemini-3.7-flash: HTTP 503 UNAVAILABLE on four consecutive
attempts across two runs, three with backoff. An availability finding, not a
capability one.

SETUP
-----
    $env:GEMINI_API_KEY="..."                                  # PowerShell
    python run_gemini.py generate 2>&1 | Tee-Object -FilePath run_log_gem.txt

    export GEMINI_API_KEY="..."                                # bash
    python run_gemini.py generate 2>&1 | tee run_log_gem.txt

STAGES
------
    python run_gemini.py models      # list available models
    python run_gemini.py probe       # ONE story — read it first
    python run_gemini.py refusal     # 10 truncated calls
    python run_gemini.py generate    # full run, ~15 min
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

API_KEY = os.environ.get("GEMINI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
API_VERSION = "v1beta"          # record this — v1 and v1beta differ

MODEL_ID = "gemini-3.5-flash"
MODEL_ID_READ_ON = "2026-08-19"

PROMPT_VARIANT = "A2"

# None = provider default = reasoning ON. DO NOT SET THIS TO 0.
# thinkingBudget=0 is honoured by this model and produces stories ~22% short
# (0 of 4 inside the gate). Kept as None deliberately, not by omission.
THINKING_BUDGET = None

# LOCAL OVERRIDE — do not read max_tokens from GEN_PARAMS (9000, a Kimi-era
# value). Gemini's thinking tokens count against the output ceiling; observed
# 3,116-5,390 thinking tokens plus ~2,500 output. 30000 is a NON-BINDING
# CEILING (DEC-021). Model output limit is 65536, so there is headroom.
MAX_TOKENS = 30000

OUT = Path("outputs")
STORY_DIR = OUT / "stories" / "gemini"     # one folder per model; identical
                                           # filenames across folders make the
                                           # crossed design visible on disk
RAW_DIR = OUT / "raw_gemini"
RUN_ID = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

REQUEST_TIMEOUT = 900
RETRY_ON_NETWORK = 3
SLEEP_BETWEEN = 2.0


def _key():
    if not API_KEY:
        sys.exit("GEMINI_API_KEY is not set.\n"
                 "  PowerShell: $env:GEMINI_API_KEY='...'\n"
                 "  bash      : export GEMINI_API_KEY='...'\n"
                 "Do not hardcode it.")
    return API_KEY


def sha256(text):
    """Prompt hash. Proves at write-up time that every model received a
    byte-identical prompt — the whole basis of the RQ4 comparison."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _preflight():
    """Fail loudly on stale settings. This check exists because a stale
    max_tokens cost a full night of generation on 2026-08-19."""
    problems = []
    if LENGTH_GATE != (720, 880):
        problems.append(f"LENGTH_GATE is {LENGTH_GATE}, expected (720, 880) "
                        f"— prompts_v3.py is stale (DEC-029)")
    if MAX_ATTEMPTS < 4:
        problems.append(f"MAX_ATTEMPTS is {MAX_ATTEMPTS}, expected 4 (DEC-029)")
    if THINKING_BUDGET == 0:
        problems.append("THINKING_BUDGET is 0 — this produces stories ~22% "
                        "short (0/4 in gate). DEC-027 requires reasoning ON. "
                        "Set it to None.")
    if MAX_TOKENS < 20000:
        problems.append(f"MAX_TOKENS is {MAX_TOKENS}; thinking tokens count "
                        f"against this ceiling and will truncate the story.")
    if problems:
        print("PREFLIGHT FAILED:")
        for p in problems:
            print("  -", p)
        sys.exit("Fix the above before running. Do not proceed.")
    print(f"Preflight OK — {MODEL_ID}, gate {LENGTH_GATE}, "
          f"{MAX_ATTEMPTS} attempts, thinking default (ON), "
          f"max_tokens {MAX_TOKENS}\n")


# ---------------------------------------------------------------------------
# MODEL DISCOVERY
# ---------------------------------------------------------------------------

def list_models():
    r = requests.get(f"{BASE_URL}/models", params={"key": _key()},
                     timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    print(f"Gemini models on {dt.date.today().isoformat()} ({API_VERSION}):\n")
    for m in r.json().get("models", []):
        if "generateContent" not in m.get("supportedGenerationMethods", []):
            continue
        name = m.get("name", "").replace("models/", "")
        alias = "   WARNING: moving alias, do not use" if name.endswith("latest") else ""
        print(f"  {name}{alias}")
        print(f"    display : {m.get('displayName')}")
        print(f"    in/out  : {m.get('inputTokenLimit')} / "
              f"{m.get('outputTokenLimit')}\n")


# ---------------------------------------------------------------------------
# CORE CALL
# ---------------------------------------------------------------------------

def _call(prompt, max_tokens, with_meta=False):
    """Returns (text|None, meta). text is None on error, safety block, or
    empty candidate. meta always carries whatever the API disclosed."""
    gen_cfg = {"temperature": GEN_PARAMS["temperature"],
               "maxOutputTokens": max_tokens}
    if THINKING_BUDGET is not None:
        gen_cfg["thinkingConfig"] = {"thinkingBudget": THINKING_BUDGET}

    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": gen_cfg}
    # No "seed": this API does not expose one. Documented asymmetry with the
    # OpenRouter route. No "safetySettings": provider defaults, deliberately.

    url = f"{BASE_URL}/models/{MODEL_ID}:generateContent"

    for attempt in range(RETRY_ON_NETWORK):
        try:
            r = requests.post(url, params={"key": _key()}, json=body,
                              timeout=REQUEST_TIMEOUT)
            if r.status_code in (429, 500, 502, 503, 504):
                wait = 15 * (attempt + 1)
                print(f"       HTTP {r.status_code}, waiting {wait}s", flush=True)
                time.sleep(wait)
                continue
            data = r.json()

            if "error" in data:
                meta = {"finish_reason": "api_error",
                        "error": str(data["error"])[:300]}
                print(f"       API error: {meta['error']}", flush=True)
                return (None, meta) if with_meta else None

            pf = data.get("promptFeedback", {})
            if pf.get("blockReason"):
                meta = {"finish_reason": "PROMPT_BLOCKED",
                        "block_reason": pf.get("blockReason")}
                print(f"       PROMPT BLOCKED: {pf.get('blockReason')} "
                      f"— REFUSAL DATA, log it", flush=True)
                return (None, meta) if with_meta else None

            cands = data.get("candidates", [])
            if not cands:
                meta = {"finish_reason": "NO_CANDIDATE"}
                return (None, meta) if with_meta else None

            c = cands[0]
            u = data.get("usageMetadata", {})
            meta = {
                "finish_reason": c.get("finishReason"),
                "prompt_tokens": u.get("promptTokenCount"),
                "completion_tokens": u.get("candidatesTokenCount"),
                "thoughts_tokens": u.get("thoughtsTokenCount"),
                "total_tokens": u.get("totalTokenCount"),
                "model_version": data.get("modelVersion"),
                "response_id": data.get("responseId"),
            }

            parts = c.get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts) or None

            if text is None and c.get("finishReason") == "SAFETY":
                print("       SAFETY BLOCK — REFUSAL DATA", flush=True)

            return (text, meta) if with_meta else text

        except Exception as e:
            print(f"       network error ({e}); retry {attempt+1}", flush=True)
            time.sleep(5 * (attempt + 1))

    return (None, {"finish_reason": "network_fail"}) if with_meta else None


# ---------------------------------------------------------------------------
# PROBE
# ---------------------------------------------------------------------------

def probe(topic_id="N12_ITNightShift"):
    """ONE story. DEMONSTRATION, NOT EVALUATION — one story per model shows
    the pipeline produces Sinhala prose and nothing about which is better."""
    prompt = build_prompt(topic_id, PROMPT_VARIANT)
    t0 = time.time()
    txt, meta = _call(prompt, MAX_TOKENS, with_meta=True)
    secs = round(time.time() - t0)

    print(f"model      : {MODEL_ID}")
    print(f"version    : {meta.get('model_version')}")
    print(f"elapsed    : {secs} s")
    print(f"finish     : {meta.get('finish_reason')}")
    print(f"thoughts   : {meta.get('thoughts_tokens')}   <-- should be >0")
    print(f"completion : {meta.get('completion_tokens')}")

    if not txt:
        print("\ncontent: None. MAX_TOKENS -> raise it. SAFETY/PROMPT_BLOCKED "
              "-> refusal data, log it.")
        return None

    clean = strip_preamble(txt)
    w = clean.split()
    ok, wc, reason = accept(clean)
    print(f"words      : {wc}   gate {LENGTH_GATE} -> "
          f"{'ACCEPT' if ok else 'REJECT: ' + str(reason)}")
    print(f"unique     : {round(len(set(w))/max(1,len(w)), 3)}")
    print(f"quotes     : {clean.count(chr(34)) + clean.count(chr(8220)) + clean.count(chr(8221))}")

    (OUT / "probe").mkdir(parents=True, exist_ok=True)
    (OUT / "probe" / f"gemini_{topic_id}.txt").write_text(clean, encoding="utf-8")
    print("\n" + "=" * 60 + "\n")
    print(clean)
    print("\n" + "=" * 60)
    print("\nNOW READ IT. The checks above are proxies.")
    return clean


# ---------------------------------------------------------------------------
# REFUSAL SCREEN
# ---------------------------------------------------------------------------

def refusal_screen():
    """Gemini's safety filter is the specific risk: romance topics can trigger
    PROMPT_BLOCKED or a SAFETY finishReason.

    MISSES hedging — a model that complies but sanitises the premise produces
    an off-topic story without announcing it. Only reading catches that.
    """
    rows = []
    print(f"Refusal screen — {MODEL_ID}, variant {PROMPT_VARIANT}\n")
    for tid in TOPICS:
        prompt = build_prompt(tid, PROMPT_VARIANT)
        txt, meta = _call(prompt, 300, with_meta=True)
        print(f"  {tid}  [{meta.get('finish_reason')}]\n"
              f"    {(txt or '')[:200]}\n", flush=True)
        rows.append({
            "run_id": RUN_ID, "model_id": MODEL_ID, "topic_id": tid,
            "variant": PROMPT_VARIANT, "prompt_sha": sha256(prompt),
            "finish_reason": meta.get("finish_reason"),
            "block_reason": meta.get("block_reason", ""),
            "head": (txt or "")[:500],
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
    """
    _preflight()
    STORY_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

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
            raw, meta = _call(prompt, MAX_TOKENS, with_meta=True)
            secs = round(time.time() - t0)

            if raw is None:
                attempts.append({
                    "run_id": RUN_ID, "topic_id": tid, "attempt": n,
                    "seconds": secs, "word_count": 0, "accepted": False,
                    "reason": meta.get("finish_reason", "no_content"),
                    "prompt_sha": p_sha,
                    "finish_reason": meta.get("finish_reason", ""),
                    "thoughts_tokens": meta.get("thoughts_tokens", ""),
                    "completion_tokens": meta.get("completion_tokens", ""),
                })
                _write_csv(OUT / f"attempts_gemini_{RUN_ID}.csv", attempts)
                print(f"   x  attempt {n}: no content "
                      f"({meta.get('finish_reason')}, {secs}s)", flush=True)
                time.sleep(SLEEP_BETWEEN)
                continue

            (RAW_DIR / f"{tid}_a{n}.txt").write_text(raw, encoding="utf-8")
            text = strip_preamble(raw)
            ok, wc, reason = accept(text)

            attempts.append({
                "run_id": RUN_ID, "topic_id": tid, "attempt": n,
                "seconds": secs, "word_count": wc, "accepted": ok,
                "reason": reason or "", "prompt_sha": p_sha,
                "finish_reason": meta.get("finish_reason", ""),
                "thoughts_tokens": meta.get("thoughts_tokens", ""),
                "completion_tokens": meta.get("completion_tokens", ""),
            })
            _write_csv(OUT / f"attempts_gemini_{RUN_ID}.csv", attempts)

            print(f"   {'OK ' if ok else 'REJ'} attempt {n}: {wc} words, "
                  f"{secs}s ({meta.get('finish_reason')})"
                  f"{'' if ok else ' — ' + str(reason)}", flush=True)

            if meta.get("finish_reason") == "MAX_TOKENS":
                print("       WARNING: TRUNCATED. Raise MAX_TOKENS and re-run "
                      "this topic. PARAMETER fault, not a model fault — do "
                      "not report as a length failure.", flush=True)

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
            "provider": "google_ai_studio", "api_version": API_VERSION,
            "thinking_budget": THINKING_BUDGET,
            "reasoning_enabled": True,
            "safety_settings": "provider_defaults_unmodified",
            "topic_id": tid, "variant": PROMPT_VARIANT,
            "prompt_sha256_16": p_sha,
            "temperature": GEN_PARAMS["temperature"],
            "max_tokens": MAX_TOKENS,
            "seed": None,               # not exposed by this API
            "length_gate": list(LENGTH_GATE),
            "max_attempts": MAX_ATTEMPTS,
            "word_count": word_count(story),
            "file": str(STORY_DIR / f"{tid}.txt"),
        })
        _write_jsonl(OUT / f"manifest_gemini_{RUN_ID}.jsonl", manifest)

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
    print(f"\n  Append to methodology.md §10 with today's date.")
    print(f"  Ten Gemini stories are ONE THIRD of a corpus, not a result.")


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
    {"models": list_models,
     "probe": probe,
     "refusal": refusal_screen,
     "generate": generate}.get(cmd, lambda: print(__doc__))()
