#!/usr/bin/env python3
"""
run_gpt.py — SSEF Phase 1, GPT generation VIA OPENROUTER
=========================================================

Project : A Quality Evaluation Framework for Sinhala Story Generation (SSEF)
Version : 1.0
Created : 2026-08-19
Requires: prompts_v3.py with LENGTH_GATE (720,880), MAX_ATTEMPTS 4,
          "seed": None   (preflight enforces all three)
Install : pip install requests

THIRD MODEL SELECTION (DEC-034)
--------------------------------
CHOSEN: openai/gpt-5.6-sol

Three general-availability gpt-5.6 variants were probed on N12_ITNightShift,
variant A2, 2026-08-19:

    luna    822 words   29 s   $0.0035   reasoning 166
    terra  1026 words   92 s   $0.043    reasoning  41
    sol     774 words  151 s   $0.041    reasoning  34

REJECTED - terra: 1,026 words, well outside the 720-880 gate.

REJECTED - luna: its dialogue uses the form "oba" throughout. Gemini,
DeepSeek and sol all use "oya". Native-speaker judgement (researcher) is that
"oya" is the normal conversational form for this context. Since DARC measures
exactly this distinction, a luna-vs-others score difference could reflect
this lexical choice rather than register competence. Register consistency
across models outweighs luna being 12x cheaper and 5x faster.

WITHDRAWN CLAIM: an earlier assessment held that sol produced more
orthographic word-boundary errors than luna. A second sol generation
(N04_Avurudu) contained none, and comparable errors were then found in BOTH
the Gemini and DeepSeek corpora. The claim rested on n=1 and does not stand.

*** REASONING TOKENS ARE LOW - A LIMITATION TO STATE ***
    Gemini 3,116  |  DeepSeek 4,293  |  gpt-5.6-sol 34

DEC-027 requires all models to run in reasoning mode, and sol satisfies that
literally (reasoning explicitly enabled, reasoning_tokens > 0). But 34 is two
orders of magnitude below the other two. Whether the three models perform
comparable deliberation is UNKNOWN. Do not write that all three ran "in the
same reasoning mode" without qualifying it - report the token counts.

*** CORPUS-WIDE ISSUE - ORTHOGRAPHIC WORD BOUNDARIES ***
All three models occasionally join two words with no space:
    Gemini   ~6-7 per story
    DeepSeek ~3-4 per story
    sol       0-4 per story

This corrupts word_count(), inflates MTLD (a joined pair reads as a novel
type), and can hide register markers from the DARC segmenter.

NOT DETECTABLE BY REGEX - an attempted pattern returned 1,263 false positives
on normal Sinhala conjuncts. Counting requires native-speaker reading.

RECOMMENDED HANDLING: measure and report; do not silently correct. Counting
errors per model is itself an RQ4 finding, and it is a defect no automated
metric in this project catches but a human reader spots immediately - which
is directly relevant to the research problem. Hand-correcting the corpus
would mean the stories are no longer model output, and that must be declared.

SETUP
-----
    $env:OPENROUTER_API_KEY="sk-or-v1-..."                  # PowerShell
    python run_gpt.py generate 2>&1 | Tee-Object -FilePath run_log_gpt.txt

STAGES
------
    python run_gpt.py probe      # ONE story - read it
    python run_gpt.py refusal    # 10 truncated calls
    python run_gpt.py generate   # full run, ~45 min, ~$0.70
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

# Read from GET /api/v1/models on 2026-08-19.
# Excluded: ~openai/gpt-latest, ~openai/gpt-mini-latest (moving aliases —
# destroy reproducibility); *-codex (code-tuned); *-image, *-audio (wrong
# modality); :batch (asynchronous); o1/o3/o4 (previous reasoning line);
# gpt-4*, gpt-3.5 (previous generations); *-mini, *-nano (smaller tier);
# gpt-oss-* (open-weights, different family); terra and luna (see header).
MODEL_ID = "openai/gpt-5.6-sol"
MODEL_ID_READ_ON = "2026-08-19"
# NOTE: gpt-5.6-sol is NOT date-pinned. Access date + generation_id are the
# provenance record (audit defect D8).

PROMPT_VARIANT = "A2"      # A2 = register NOT instructed. "A1" only for the
                           # pilot comparison.

# Explicit, not a default. See the header note.
REASONING = {"enabled": True}

# LOCAL max_tokens OVERRIDE — a documented deviation from GEN_PARAMS.
# Observed usage with reasoning on: 7,021 completion tokens. 30,000 is a
# NON-BINDING CEILING (DEC-021) with generous headroom. Length is enforced
# post-hoc by the word gate, so ceilings may differ across models PROVIDED
# finish_reason is never "length". WATCH THE LOG FOR IT.
MAX_TOKENS = 30000

OUT = Path("outputs")
STORY_DIR = OUT / "stories" / "gpt"   # one folder per model; identical
                                           # filenames across folders make the
                                           # crossed design visible on disk
RAW_DIR = OUT / "raw_gpt"
RUN_ID = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

REQUEST_TIMEOUT = 900      # observed 102-137 s; generous margin
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
    configuration. This check exists because a stale MAX_TOKENS cost a full
    night of generation on 2026-08-19."""
    problems = []
    if LENGTH_GATE != (720, 880):
        problems.append(f"LENGTH_GATE is {LENGTH_GATE}, expected (720, 880) "
                        f"— prompts_v3.py is stale (DEC-029)")
    if MAX_ATTEMPTS < 4:
        problems.append(f"MAX_ATTEMPTS is {MAX_ATTEMPTS}, expected 4 (DEC-029)")
    if GEN_PARAMS.get("seed") is not None:
        problems.append('GEN_PARAMS["seed"] is set. A fixed seed makes every '
                        'retry return the IDENTICAL story and defeats the '
                        'length gate (DEC-033). Set it to None.')
    if not REASONING.get("enabled"):
        problems.append("REASONING is not enabled — DEC-027 requires all "
                        "models to run in reasoning mode")
    if problems:
        print("PREFLIGHT FAILED:")
        for p in problems:
            print("  -", p)
        sys.exit("Fix the above before running. Do not proceed.")
    print(f"Preflight OK — {MODEL_ID}, gate {LENGTH_GATE}, "
          f"{MAX_ATTEMPTS} attempts, reasoning ON, "
          f"max_tokens {MAX_TOKENS}, seed OFF\n")


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
    if GEN_PARAMS.get("seed") is not None:
        body["seed"] = GEN_PARAMS["seed"]

    for attempt in range(RETRY_ON_NETWORK):
        try:
            r = requests.post(f"{BASE_URL}/chat/completions",
                              headers=_headers(), json=body,
                              timeout=REQUEST_TIMEOUT)
            # 5xx are transient capacity errors, not fatal. gemini-3.7-flash
            # was rejected on 2026-08-19 for returning 503 four times running.
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
# STAGE 1 — SINGLE PROBE
# ---------------------------------------------------------------------------

def probe(topic_id="N12_ITNightShift"):
    """ONE story, same topic the other models were probed on.

    ⚠️ DEMONSTRATION, NOT EVALUATION. One story per model shows the pipeline
    produces Sinhala prose. It shows NOTHING about which model is better.
    Do not compare quality at n=1.
    """
    prompt = build_prompt(topic_id, PROMPT_VARIANT)
    t0 = time.time()
    txt, meta = _call(prompt, MAX_TOKENS, with_meta=True)
    secs = round(time.time() - t0)

    print(f"model      : {MODEL_ID}")
    print(f"elapsed    : {secs} s")
    print(f"finish     : {meta.get('finish_reason')}")
    print(f"reasoning  : {meta.get('reasoning_tokens')}   <-- must be >0 and "
          f"NOT the whole budget")
    print(f"completion : {meta.get('completion_tokens')}")
    print(f"cost       : ${meta.get('cost')}")

    if not txt:
        print("\ncontent: None. If reasoning_tokens equals max_tokens, this is "
              "the Kimi failure mode (DEC-030) — STOP, do not raise the "
              "ceiling, it makes it worse.")
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
    (OUT / "probe" / f"gpt_{topic_id}.txt").write_text(clean, encoding="utf-8")
    print("\n" + "=" * 60 + "\n")
    print(clean)
    print("\n" + "=" * 60)
    print("\nNOW READ IT. The checks above are proxies. Whether the Sinhala is "
          "good is a native-speaker judgement and no script can make it.")
    return clean


# ---------------------------------------------------------------------------
# STAGE 2 — REFUSAL SCREEN
# ---------------------------------------------------------------------------

def refusal_screen():
    """10 topics, truncated calls.

    CATCHES : refusal, safety preamble, non-Sinhala output, title/preamble.
    MISSES  : hedging — a model that complies but sanitises the premise
              produces an off-topic story without announcing it. Only reading
              full-length output catches that.

    `refused` and `sinhala_script` are BLANK on purpose. Fill them by reading.
    An automated refusal classifier would be a second unvalidated instrument
    in a project that already has one.
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
        _write_csv(OUT / f"refusal_gpt_{RUN_ID}.csv", rows)
        time.sleep(SLEEP_BETWEEN)

    print(f"Saved outputs/refusal_gpt_{RUN_ID}.csv — read every row.")
    print("Apply REFUSAL_RULE from prompts_v3.py before changing any topic.")


# ---------------------------------------------------------------------------
# STAGE 3 — GENERATION
# ---------------------------------------------------------------------------

def generate():
    """10 topics, length gate, up to MAX_ATTEMPTS each.

    SAVES (written incrementally — a crash loses at most one attempt)
      outputs/stories/deepseek/{topic_id}.txt   accepted story, UTF-8
      outputs/manifest_gpt_{run}.jsonl     one line per accepted story
      outputs/attempts_gpt_{run}.csv       EVERY attempt, rejections too
      outputs/raw_gpt/{topic}_a{n}.txt     raw response, pre-strip

    Raw responses are kept because strip_preamble() is a crude guard, not a
    parser. If it strips something it should not have, the raw file is the
    only way to find out.
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
                    "reason": "no_content", "prompt_sha": p_sha,
                    "finish_reason": meta.get("finish_reason", ""),
                    "reasoning_tokens": meta.get("reasoning_tokens", ""),
                    "completion_tokens": meta.get("completion_tokens", ""),
                    "cost": meta.get("cost", ""), "generation_id": "",
                })
                _write_csv(OUT / f"attempts_gpt_{RUN_ID}.csv", attempts)
                print(f"   x  attempt {n}: no content ({secs}s)", flush=True)
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
                "reasoning_tokens": meta.get("reasoning_tokens", ""),
                "completion_tokens": meta.get("completion_tokens", ""),
                "cost": meta.get("cost", ""),
                "generation_id": meta.get("generation_id", ""),
            })
            _write_csv(OUT / f"attempts_gpt_{RUN_ID}.csv", attempts)

            print(f"   {'OK ' if ok else 'REJ'} attempt {n}: {wc} words, "
                  f"{secs}s ({meta.get('finish_reason')})"
                  f"{'' if ok else ' — ' + str(reason)}", flush=True)

            # "length" means max_tokens truncated the story. PARAMETER fault,
            # not a model fault — raise MAX_TOKENS and re-run that topic.
            # Do NOT report it as a length-compliance failure.
            if meta.get("finish_reason") == "length":
                print("       WARNING: TRUNCATED by max_tokens. Raise "
                      "MAX_TOKENS and re-run this topic.", flush=True)

            if ok:
                story = text
                break
            time.sleep(SLEEP_BETWEEN)

        if story is None:
            print(f"   FAILED {tid}: no accepted generation in {MAX_ATTEMPTS} "
                  f"attempts. This is a FINDING about the model on this topic "
                  f"— report the rate.", flush=True)
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
            "seed": GEN_PARAMS.get("seed"),
            "length_gate": list(LENGTH_GATE),
            "max_attempts": MAX_ATTEMPTS,
            "word_count": word_count(story),
            "generation_id": next((a["generation_id"] for a in reversed(attempts)
                                   if a["topic_id"] == tid and a["accepted"]), ""),
            "file": str(STORY_DIR / f"{tid}.txt"),
        })
        _write_jsonl(OUT / f"manifest_gpt_{RUN_ID}.jsonl", manifest)

    rej = sum(1 for a in attempts if not a["accepted"])
    wcs = [a["word_count"] for a in attempts if a["word_count"]]
    costs = [a["cost"] for a in attempts if isinstance(a.get("cost"), (int, float))]
    mins = round((time.time() - t_start) / 60)

    print("\n" + "=" * 55)
    print(f"  Accepted   : {accepted_n} / {len(TOPICS)}")
    print(f"  Attempts   : {len(attempts)}  (rejections: {rej})")
    if wcs:
        print(f"  Word range : {min(wcs)} - {max(wcs)}, "
              f"mean {round(sum(wcs)/len(wcs))}")
    if costs:
        print(f"  Total cost : ${round(sum(costs), 3)}")
    print(f"  Elapsed    : {mins} min")
    print(f"\n  Append to methodology.md §10 with today's date: accepted "
          f"{accepted_n}/{len(TOPICS)}, {rej} rejections, word range, "
          f"rejection reasons, model ID and access date.")
    print(f"\n  Reference runs, same prompt and gate, 2026-08-19:")
    print(f"    gemini-3.5-flash  10/10, 14 attempts,  4 rej, 645-899,  $0.921")
    print(f"    deepseek-v4-pro   10/10, 26 attempts, 16 rej, 489-1035, $0.404")
    print(f"\n  With all three models this is 30 stories and the AI corpus is")
    print(f"  COMPLETE. The HUMAN REFERENCE CORPUS (D9) is still missing, and")
    print(f"  without it no DARC score can be interpreted as good or bad.")


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
