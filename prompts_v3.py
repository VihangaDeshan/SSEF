"""
prompts_v3.py — SSEF Phase 1 story generation prompts
=====================================================

Project : A Quality Evaluation Framework for Sinhala Story Generation (SSEF)
Version : 3.1
Created : 2026-08-19
Updated : 2026-08-19 (evening — measured corrections)
Supersedes: prompts_v3.0, v2.0, v1.0

CHANGES IN 3.1 — all from measurements taken 2026-08-19
--------------------------------------------------------
* LENGTH_GATE  (760, 840) -> (720, 880).   DEC-029
* MAX_ATTEMPTS 3 -> 4.                      DEC-029
* max_tokens   3500 -> 9000.                measured, see GEN_PARAMS
* MODEL_IDS filled in for gemini and kimi; kimi-k3 recorded as rejected.
* STAGES text corrected: the tokenizer screen is NOT a go/no-go gate.
  That heuristic is WITHDRAWN (see TOKENIZER_NOTE).
* tokens_per_word() docstring corrected — it was still asserting the
  falsified threshold.

┌─────────────────────────────────────────────────────────────────────────┐
│ PROVENANCE — RECORD THIS IN THE METHODOLOGY                             │
│ The revised topic list appears to be model-generated and subsequently   │
│ curated by the researcher (a native speaker). If so, state it plainly:  │
│ topics were LLM-generated and human-curated. Do not present them as     │
│ researcher-authored. An LLM-designed topic set for an LLM benchmark is  │
│ defensible with curation, and indefensible if concealed.                │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ SPECIFICATION LEVEL — A KNOWN UNCONTROLLED VARIABLE                     │
│ The revised topics specify characters, named locations, season, weather │
│ and atmosphere; the retired topics specified a relationship premise     │
│ only. Consequences:                                                     │
│ (a) UNEVEN LENGTH ACROSS TOPICS confounds topic with specification      │
│     level.                                                              │
│ (b) SENSORY/SCENIC DETAIL PUSHES MODELS TOWARD DESCRIPTIVE NARRATION,   │
│     depressing the DARC Register Balance sub-score for reasons          │
│     unrelated to model competence. Identical across models, so RQ4 is   │
│     unaffected; absolute DARC values ARE shifted, which matters for the │
│     DEC-022 z-score composite.                                          │
│ MITIGATION NOT APPLIED: normalisation to 25-40 words was specified but  │
│ not carried out. TOPICS_RAW is in use. THIS IS AN ACCEPTED LIMITATION,  │
│ NOT AN OVERSIGHT — state it in one sentence in the limitations.         │
└─────────────────────────────────────────────────────────────────────────┘
"""

# ---------------------------------------------------------------------------
# SPEC — the eleven points expressed by the instruction block
# ---------------------------------------------------------------------------
SPEC = """
 1. Role: an experienced Sinhala short-story writer.
 2. Task: write ONE complete love story in Sinhala.
 3. Length: approximately 800 words.
 4. Point of view: third person.
 5. Structure: complete narrative arc — opening, romantic conflict, resolution.
 6. Setting: as given in the topic.
 7. Dialogue: several exchanges between the characters.
    NO quantity. NO ratio. NO percentage. Instructing the ratio would make the
    DARC Register Balance sub-score circular.
 8. All dialogue enclosed in double quotation marks "…".
    Non-negotiable: the DARC segmenter cannot see em-dash or unquoted dialogue.
 9. Output entirely in Sinhala, Sinhala Unicode script.
10. Output the story text ONLY — no title, preamble, commentary, word count.
11. A1 VARIANT ONLY: literary register for narration, spoken register for
    character dialogue. OMITTED IN A2.
"""

TOKENIZER_NOTE = """
WITHDRAWN CLAIM — tokens-per-word as a capability gate.

The v3.0 heuristic (<=4 usable, 5-6 marginal, 7+ expect degenerate output)
was built from TWO data points — SinBERT ~4.1 and Llama-3.1-8B 11.0 — was
presented as a go/no-go gate, and was FALSIFIED the same day:

    gemini-3.5-flash   3.54 tok/word   wrote fine prose
    SinBERT           ~4.10            reference point
    kimi-k2.6          8.01            WROTE FINE PROSE
    Llama-3.1-8B      11.00            degenerate: 93% repetition, 29-token
                                       loop, zero dialogue

Inefficient tokenisation and inability to write Sinhala are separate
properties. A large model tolerates the former. The figure remains useful for
budget estimation and max_tokens selection ONLY. Capability is decided by
generating a story and reading it.

Report the instrument, its falsification, and its revision. Hevner Guideline 6
rewards a visible search, including one that corrected itself.
"""

NORMALISATION = """
NOT APPLIED — TOPICS_RAW is in use. Recorded as a limitation.
If ever applied:
  TARGET: 25-40 words per topic, uniform across all ten.
  KEEP   : character names; the relationship; one line of setting.
  CUT    : weather, sensory atmosphere, emotional framing, parenthetical
           English glosses.
  WRITTEN BY: the researcher. Not generated.
"""


# ---------------------------------------------------------------------------
# SELECTED TOPICS (10)
# ---------------------------------------------------------------------------
# Selection criteria:
#   (a) peer relationships — no employer/subordinate, no teacher/student, no
#       formal deference. Removes the register confound the retired set
#       carried (claim still UNVERIFIED; see RETIRED_TOPICS).
#   (b) diversity of setting and social stratum
#   (c) no within-set duplication
#   (d) no characters who read as under 18

TOPICS_RAW = {
    "N01_CoastTrain": "කොළඹ කාර්යාල නිමවී ගාලු මුහුදුබඩ දුම්රියේ දිනපතා එකම අසුන බෙදාගන්නා කසුන් සහ තිළිණි අතර නිහඬව ගොඩනැගෙන බැඳීමක්. දුම්රිය ජනේලයෙන් එන මුහුදු සුළඟ, හැන්දෑවේ අලුත්කඩේ තේ කෝප්පයක් සහ නාගරික තනිකම මැද උපදින ආදරය.",
    "N02_Peradeniya": "පේරාදෙණිය විශ්වවිද්‍යාල පරිශ්‍රයේ කලා මාවත සහ සරසවි පුස්තකාලය වටා දිවෙන සසිඳු සහ දිනීෂාගේ කතාවක්. හන්තාන කඳුවැටියෙන් ඇදහැලෙන අනෝරා වැස්සකදී එකම කුඩයක් යට තෙමෙමින්, අවසන් වසරේ විභාග බිය සහ අනාගතය පිළිබඳ අවිනිශ්චිතතාව මැද රැඳෙන සදාතනික බැඳීම.",
    "N03_GalleFort": "ගාලු කොටු පවුර අසල ඓතිහාසික ගොඩනැගිල්ලක පිහිටි කලාගාරයක සහ කැෆේ එකක හමුවන චිත්‍ර ශිල්පියෙකු වන නිමේෂ් සහ සංරක්ෂණ නිලධාරිනියක් වන අමාලි. පැරණි ඕලන්ද ගෘහ නිර්මාණ ශිල්පය, මුහුදු රළ හඬ සහ අතීත මතකයන් පාදක කරගත් ගැඹුරු සංවාදශීලී ප්‍රේමය.",
    "N04_Avurudu": "ඈත ගමක අස්වනු නෙළන කාලය සහ සිංහල අලුත් අවුරුදු උත්සව පසුබිම් කරගත් කවිඳු සහ සඳුනිගේ කතාව. ගමේ චාරිත්‍ර, ඔන්චිලි වාරම් සහ කුඹුරු නියරවල් මැදින් හුවමාරු වන බැල්මවල් මතින් ඇරඹෙන පාරම්පරික හා ගැමි සුවඳ රැඳි අහිංසක ආදරය.",
    "N07_Rajarata": "අනුරාධපුරය නුවර කලාවියේ මහා වැවක් අසල සන්ධ්‍යා යාමයේ රුවන්ත සහ තක්ෂිලාගේ කතාව. නටබුන් සහ පුරාවිද්‍යා ගවේෂණ කටයුතු අතරතුර, වැව් දිය මත දිලිසෙන හිරු එළිය සාක්ෂි දරද්දී ගොඩනැගෙන පරිණත, තැන්පත් ආදර කතාවක්.",
    "N09_Vesak": "වෙසක් සමයේ ගමේ පන්සලේ තොරණ සහ පහන් කූඩු නිර්මාණ කටයුතු පසුබිම් කරගත් ලහිරු සහ නෙත්මිගේ ආදරයක්. උණ බම්බු කැපීම, සව්කොළ ඇලවීම සහ පහන් දැල්වීම අතරතුර හදවත් තුළ දැල්වෙන නොනිමෙන පහනක් වන් හැඟීමක්.",
    "N10_Mirissa": "දකුණු වෙරළ තීරයේ සර්ෆින් පුහුණුකරුවෙකු වන චතුර සහ නිවාඩුවට පැමිණි නාගරික තරුණියක වන මල්ෂි. සාගරයේ රළ පහරවල්, සැඳෑ හිරු බැසයාම සහ නිදහස් වෙරළ සංස්කෘතිය මැද උපදින අනපේක්ෂිත ආදරය.",
    "N12_ITNightShift": "කොළඹ නගරයේ තොරතුරු තාක්ෂණ ආයතනයක රාත්‍රී සේවා මුරයේ යෙදෙන තරින්දු සහ රශ්මි. ව්‍යාපෘති අවසන් කිරීමේ පීඩනය, කාර්යාලීය කෝපි මැෂිම අසල කෙටි කතාබහ සහ කොළඹ නිහඬ රැය තුළ හුදෙකලාව දුරලන ආදරය.",
    "N14_FloodRelief": "රත්නපුරය ප්‍රදේශයේ ගංවතුර ආපදාවකදී සහන සැලසීමට එක්වන ස්වේච්ඡා තරුණයෙකු වන ප්‍රමෝද් සහ වෛද්‍ය ශිෂ්‍යාවක. අන් අයට උදව් කිරීමේ මානුෂීය මෙහෙයුම අතරතුර කැපවීම සහ දයාව මත පදනම්ව ගොඩනැගෙන උතුම් ආදරයක්.",
    "N15_Airport": "රට රැකියාවකට පිටත්ව යාම සහ වසර ගණනාවකට පසු නැවත පැමිණීම පසුබිම් කරගත් දසුන් සහ මනුෂිගේ කතාව. කටුනායක ගුවන් තොටුපළේ පැමිණීමේ පර්යන්තයේදී කඳුළු සහ සිනහව මැද යළි එක්වන සැබෑ ආදරයේ ඉවසීම.",
}

# NAME ASYMMETRY: N14's medical student is unnamed (අමාලි collided with N03).
# Nine topics name two characters; N14 names one. Minor uncontrolled variation
# — the model will invent a name. Record it; do not fix mid-corpus.

TOPICS_NORM = {k: "" for k in TOPICS_RAW}   # not written; see NORMALISATION
TOPICS = TOPICS_RAW


# ---------------------------------------------------------------------------
# RESERVE TOPICS — each carries a specific, named problem
# ---------------------------------------------------------------------------
RESERVES = """
#8  Yal Devi — Sinhala man / Tamil woman, Colombo to Jaffna.
    STRONGEST TOPIC SCIENTIFICALLY, RISKIEST OPERATIONALLY.
    (a) elevated hedging risk;
    (b) TAMIL-LANGUAGE CONTAMINATION — if a model renders the Tamil
        character's dialogue in Tamil, the tokeniser, the DARC segmenter and
        SinBERT all receive input none of them assumes. Every component in
        the pipeline is Sinhala-only.
    Include only as a deliberate decision with (b) planned for.

#5  Tea planter + environmentalist — occupational status gap.
#11 Kandy dancer + dance teacher — teacher-to-student. This IS the deference
    confound, not merely a risk of it.

#6  A/L tuition students and #13 Big Match schoolgirl — EXCLUDED, not
    reserved. Characters read as under 18.
"""


# ---------------------------------------------------------------------------
# RETIRED TOPICS — kept, not deleted. The swap may need reversing.
# ---------------------------------------------------------------------------
# The register-confound argument that motivated retiring these is STILL
# UNVERIFIED. If a second native speaker confirms that deferential Sinhala
# speech shifts only in address terms and honorifics — NOT in verb morphology
# — the confound does not exist and the swap was unnecessary.
#
# COST OF THE SWAP, to be stated either way: the retired set covered a
# three-wheel driver, a garment worker, a landless farmer, a divorced woman,
# an arranged marriage. The new set is largely middle-class and picturesque.
# Social range NARROWED — weakens the benchmark-dataset claim in Objective 3.

RETIRED_TOPICS = {
    "T02_Office": "රජයේ කාර්යාලයක සේවය කරන ඉහළ නිලධාරියෙක් සහ ඔහු යටතේ සේවය කරන සේවිකාවක් අතර ඇතිවන ප්‍රේම සබඳතාවක්",
    "T03_Farmer": "ග්‍රාමීය දරිද්‍රතාවෙන් පෙලෙන පරිසරයක ජීවත් වන ගොවිතැන් කටයුතු කරන තරුණයෙකු සහ පියා අහිමි දුප්පත් පවුලක තරුණියක් අතර ඇතිවන ප්‍රේම සබඳතාවක්",
    "T04_ThreeWheel": "ත්‍රී රෝද රථ රියදුරෙක් සහ නිමි ඇඳුම් කර්මාන්තශාලා සේවිකාවක් අතර ඇතිවන ප්‍රේම සබඳතාවක්",
    "T06_AgeGap": "වයසින් මුහුකුරාගිය විවාහයකින් වෙන්වූ කාන්තාවක් සහ ඇයට වඩා වයසින් බාල තරුණයෙකු අතර ඇතිවන පෙම් සබඳතාවක්",
    "T07_AICompanion": "පෙම්වතෙක් ලෙස සලකමින් කෘතිම බුද්ධිය සමග ස්වකීය ජීවිතය බෙදාගන්නා තරුණියකගේ ප්‍රේමයක්",
    "T08_Arranged": "ධනවත් කුලවත් පවුලක මැදිහත්වීමෙන් විවාහ යෝජනාවක් ලෙස සම්බන්ධ කෙරූ උසස් පවුල් දෙකක තරුණයෙකු හා තරුණියකගේ ප්‍රේම සබඳතාවක්",
}


# ---------------------------------------------------------------------------
# FORK B — RESOLVED: English instruction block + Sinhala topic (hybrid)
# ---------------------------------------------------------------------------
FORK_B_RECORD = """
CHOSEN: English instruction block + Sinhala topic description.
  English has no diglossia, so the instruction block cannot prime literary or
  spoken Sinhala forms. Removes the register-priming risk a Sinhala
  instruction block would introduce into the experiment measuring register.

REJECTED: full Sinhala instruction block — instructional Sinhala prose is
  necessarily literary register and sits at the point of maximum priming
  influence. A constant downward shift on Dialogue Register cancels across
  models (RQ4 safe) but contaminates absolute DARC values (DEC-022 unsafe).

REJECTED: full English prompt — translates the theme, adding a step the study
  does not measure.

CARRIED RISK — UNVERIFIED: code-switched prompts. Models may handle the
  language switch unevenly. sources.md Part C.
"""


# ---------------------------------------------------------------------------
# INSTRUCTION BLOCKS
# ---------------------------------------------------------------------------

ENGLISH_INSTRUCTION_A2 = """You are an experienced Sinhala short-story writer. Write one complete love story in Sinhala.

Requirements:
- Length: approximately 800 words.
- Third-person point of view.
- Both principal characters are adults.
- A complete narrative arc: an opening, a romantic conflict, and a resolution.
- Include several exchanges of dialogue between the characters.
- Enclose all dialogue in double quotation marks ("...").
- Write entirely in Sinhala, in Sinhala Unicode script.
- Output the story text only. Do not include a title, an introduction, a commentary, or a word count.

Theme: {topic}"""


ENGLISH_INSTRUCTION_A1 = """You are an experienced Sinhala short-story writer. Write one complete love story in Sinhala.

Requirements:
- Length: approximately 800 words.
- Third-person point of view.
- Both principal characters are adults.
- A complete narrative arc: an opening, a romantic conflict, and a resolution.
- Include several exchanges of dialogue between the characters.
- Enclose all dialogue in double quotation marks ("...").
- Use the literary register for narration and the spoken register for character dialogue.
- Write entirely in Sinhala, in Sinhala Unicode script.
- Output the story text only. Do not include a title, an introduction, a commentary, or a word count.

Theme: {topic}"""

# The v2 line "Set the story in either a rural or an urban setting" was REMOVED
# in v3.0 — the revised topics specify their own settings and the line would
# contradict the topic slot. v2 and v3 prompts are NOT interchangeable and
# their outputs CANNOT be pooled.

SINHALA_INSTRUCTION_A1 = ""  # REJECTED variant — retained as design-search record
SINHALA_INSTRUCTION_A2 = ""  # REJECTED variant — retained as design-search record

_BLOCKS = {"A1": ENGLISH_INSTRUCTION_A1, "A2": ENGLISH_INSTRUCTION_A2}


# ---------------------------------------------------------------------------
# BUILDER
# ---------------------------------------------------------------------------

def build_prompt(topic_id, variant="A2", topics=None):
    topics = topics or TOPICS
    if topic_id not in topics:
        raise KeyError(f"Unknown topic_id {topic_id!r}. Known: {list(topics)}")
    if not topics[topic_id].strip():
        raise ValueError(f"Topic {topic_id!r} has no description written yet.")
    if variant not in _BLOCKS:
        raise KeyError(f"Unknown variant {variant!r}. Use 'A1' or 'A2'.")
    return _BLOCKS[variant].format(topic=topics[topic_id])


def all_prompts(variant="A2", topics=None):
    topics = topics or TOPICS
    return {tid: build_prompt(tid, variant, topics) for tid in topics}


# ---------------------------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------------------------

GEN_PARAMS = {
    "temperature": 0.7,
    # 9000: NON-BINDING CEILING (DEC-021), not a length control.
    # Measured 2026-08-19 — kimi-k2.6 finishes with "stop" at 9000.
    # ⚠️ GEMINI NEEDS ~30000, because its thinking tokens count against the
    # output ceiling (observed: 3357 thinking tokens truncated a story at
    # 9000). run_gemini.py must override this locally. Differing ceilings do
    # not affect output PROVIDED finish_reason is never "length"/"MAX_TOKENS".
    # Watch the attempt log for it.
    "max_tokens": 9000,
    # OpenRouter passes seed through only for providers that support it.
    # The Google AI Studio route exposes NO seed at all — a documented
    # asymmetry between the Kimi and Gemini runs.
     #SEED REMOVED — DEC-033. Sending a fixed seed made every retry return the
    # IDENTICAL story, so a topic that missed the gate on attempt 1 was stuck
    # outside it for all four attempts (N02: 667, 667, 667 · N09: 602, 616,
    # 602, 602). The regeneration protocol was inert; only 6/10 accepted.
    # Gemini honours the seed; DeepSeek ignores it — so the corpus was never
    # fully seeded anyway.
    # COST: generation is no longer bit-reproducible. Recorded instead:
    # prompt text, prompt SHA-256, model ID, access date, temperature,
    # max_tokens, reasoning setting, gate, attempt number, generation_id.
    "seed": None,
}

MODEL_IDS = {
    "chatgpt": None,                       # [GAP — must be a REASONING model
                                           #  per DEC-027, or the design breaks]
    "gemini":  "gemini-3.5-flash",         # Google AI Studio, read 2026-08-19.
                                           # thoughtsTokenCount 3357 -> reasons.
                                           # gemini-3.7-flash REJECTED: HTTP 503
                                           # on four consecutive attempts.
    "kimi":    "moonshotai/kimi-k2.6",     # OpenRouter, read 2026-08-19.
                                           # kimi-k3 REJECTED: reasoning-only,
                                           # content None, $0.135/call.
}

# ---------------------------------------------------------------------------
# LENGTH GATE — DEC-029, measured
# ---------------------------------------------------------------------------
# 800 +/- 10%. Evidence, n=7 across two models on an identical prompt:
#     kimi-k2.6        663, 728, 859, 751   (mean 750)
#     gemini-3.5-flash 714, 824, 822        (mean 787)
#     pooled mean 766, range 663-859
# The previous +/-5% gate (760, 840) accepted only 2 of 7. It was NARROWER
# THAN THE RUN-TO-RUN VARIANCE at temperature 0.7, so it was unreachable by
# construction rather than because the models write badly.
#
# COST OF WIDENING: length uniformity loosens. Length uniformity is what keeps
# PPPL and coherence comparable across stories; MTLD is designed to be
# length-independent. State this as a limitation in one sentence.
#
# The human reference corpus MUST be selected in the SAME band, or the DEC-022
# composite compares constrained AI output against unconstrained human output.
LENGTH_GATE = (720, 880)
MAX_ATTEMPTS = 4


# ---------------------------------------------------------------------------
# STAGE ORDER
# ---------------------------------------------------------------------------
STAGES = """
0. Tokenizer measurement.  INFORMATIONAL ONLY — not a gate. See TOKENIZER_NOTE.
1. Refusal screen: 10 topics x 3 models, max_tokens=200. ~30 calls, pennies.
   Ratify REFUSAL_RULE BEFORE running, not after seeing which topic failed.
2. Prompt pilot: A1 vs A2, 2 topics x 3 models x 2 variants = 12 generations.
   PILOT STORIES DO NOT ENTER THE CORPUS.
3. Production: 10 topics x 3 models = 30 stories.

Estimated production cost, from measurements 2026-08-19:
   kimi-k2.6        ~7 min/attempt, ~$0.075/attempt  -> ~2 h, ~$1.30
   gemini-3.5-flash ~2 min/attempt                   -> well under 1 h
"""

PILOT_TOPICS = ["N12_ITNightShift", "N14_FloodRelief"]
# N12 = office setting, closest analogue to the retired T02 — shows whether
#       removing the power asymmetry mattered.
# N14 = disaster/humanitarian, emotionally heavier — a plausible hedging case.

REFUSAL_RULE = """
RECOMMENDED DEFAULT — ratify or override in DEC-026 BEFORE running Stage 1.

The design is CROSSED (same ten topics x three models); the pairing is what
allows score differences to be attributed to the model rather than to topic.

  1-2 refusals from one model:
      KEEP the topic. Report the refusal as an RQ4 content-limitation finding.
      Corpus becomes N=29 or N=28. State n at every reporting point; do NOT
      silently substitute.

  3+ refusals from one model:
      A systematic content limitation, not noise. Report it as a finding in
      its own right, AND swap the offending topics for reserves. Record which
      topics moved and why.

  Refused by ALL THREE models:
      Topic unusable. Drop for all three and replace. Pairing preserved.

RATIONALE: replacing topics costs generation time and re-runs; reporting a
refusal costs a sentence and is a genuine result. Bias toward reporting.
"""


# ---------------------------------------------------------------------------
# LENGTH GATE FUNCTIONS
# ---------------------------------------------------------------------------

def word_count(text):
    """Whitespace count. Sinhala is space-delimited, adequate for the gate.
    The SAME method must produce every word-count figure in the dissertation,
    including for the human reference corpus. State it in the methodology."""
    return len(text.split())


def strip_preamble(text):
    """Minimal guard. Preamble contaminates word count, MTLD, PPPL, and lands
    in the DARC narrative segment.

    NOT A PARSER. It trims blank lines only — it cannot tell planning prose
    from story prose. Observed 2026-08-19: a TRUNCATED Gemini generation began
    with English planning text ("Let's rewrite the flow directly in
    Sinhala...") which this function would NOT have removed. On a COMPLETE
    generation the leak did not occur. Inspect the first and last line of
    every generation; the raw/ files exist for exactly this."""
    lines = text.strip().splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()


def accept(text, gate=LENGTH_GATE):
    wc = word_count(text)
    lo, hi = gate
    if wc < lo:
        return False, wc, f"too_short ({wc} < {lo})"
    if wc > hi:
        return False, wc, f"too_long ({wc} > {hi})"
    return True, wc, None


def generate_with_gate(call_model, prompt, meta, log, max_attempts=MAX_ATTEMPTS):
    """REJECTIONS ARE DATA — they answer 'did the model comply with the length
    instruction?' for RQ4. Publish the rejection log with the dataset.

    Returning None after MAX_ATTEMPTS is a FINDING about that model on that
    topic, not an error to retry once more.

    NOTE: each attempt is an INDEPENDENT call with the SAME prompt and no
    conversation history. The model is never told a previous attempt was too
    short. Telling it so would be a DIFFERENT PROMPT, and the three models
    would no longer have received identical input — RQ4 would break."""
    for attempt in range(1, max_attempts + 1):
        text = strip_preamble(call_model(prompt))
        ok, wc, reason = accept(text)
        log.append({**meta, "attempt": attempt, "word_count": wc,
                    "accepted": ok, "reason": reason})
        if ok:
            return text, log
    return None, log


def tokens_per_word(tokenizer, text):
    """INFORMATIONAL ONLY. See TOKENIZER_NOTE — the capability threshold built
    on this measure was falsified 2026-08-19 and is withdrawn. Useful for
    budget and max_tokens selection, not for accepting or rejecting a model."""
    return len(tokenizer.encode(text)) / max(1, word_count(text))


if __name__ == "__main__":
    print(SPEC)
    print(TOKENIZER_NOTE)
    print(STAGES)
    print(f"Topics      : {len(TOPICS)} selected, {len(RETIRED_TOPICS)} retired")
    print(f"Length gate : {LENGTH_GATE}, {MAX_ATTEMPTS} attempts")
    print(f"Params      : {GEN_PARAMS}\n")
    print("--- N12_ITNightShift / A2 ---\n")
    print(build_prompt("N12_ITNightShift", "A2"))
