# Response Plan for Model-Related Reviewer Comments

**Manuscript:** "A Soft Wearable System for Real-time Respiratory Disease Diagnosis and Vibrotactile Intervention" (Science Advances)
**Prepared by:** Yun Tae Hwang — 2026-08-14 (rev. 3; scope narrowed to genuinely model-technical items, verified against the final code over two internal audit rounds)

## Executive Summary

No new data collection and no new clinical experiments are required to address the model-related comments. Six comments fall within the genuinely model-technical scope — the questions that can only be answered from the training pipeline and the classifier itself — and each of them can be resolved either by literature-backed justification, by re-analysis of data we already have (computation only), or by clarifying the Methods text. The remaining comments that mention the model only in passing (sound-versus-diagnosis framing, prevention claims, labeling procedure) can be answered generally without model work; for those I provide one-line technical inputs where useful, and they are otherwise outside this plan.

The proposed timeline is **two weeks** from confirmation of the finalized code set. Two prerequisites — one external, one internal — should be settled early in Week 1: the clinical team's one-line confirmation of the annotation procedure (needed for the labeling answer, whoever writes it), and an internal decision to unify the real-time update interval between the manuscript and the app (needed before the latency answer is written; details in the internal memo). Week 1 covers the code-set verification, the per-class re-evaluation, the alert-level analysis, and the latency bench measurement; Week 2 covers the point-by-point responses and revised manuscript text, with internal review by Dr. Oh before delivery to Prof. Yeo.

---

## Core Model-Technical Items (our scope)

### R2-9 — Why 2-s windows and a 0.5-s hop; what about split events and padding

The reviewer worries that fixed windows may divide a respiratory event between segments or miss short events, and that padding shorter segments may introduce artificial patterns. This is precisely the "why did you set it up this way" class of question, and the answer is a single grounded argument rather than a new experiment.

The window length is defensible from physiology and from our own annotations. An adult respiratory rate of 12–20 breaths per minute corresponds to a 3–5-s cycle, which at a typical 1:2 inspiratory-to-expiratory ratio gives roughly 1–2 s per phase, so a 2-s window is long enough to encompass a complete inspiratory or expiratory phase together with the transient adventitious events inside it. Our annotated data agree: 88.2% of breath phases are shorter than 2 s with a median around 1.2 s — shorter than the adult theoretical bound, consistent with the faster respiratory rates of the pediatric asthma participants. The choice is also in line with the 1–3-s analysis windows commonly used in the lung-sound classification literature, and we will attach references to that effect. For faster breathing, where a window may span more than one phase, the 0.5-s hop provides 4× overlap in the analysis pipeline, so each event is represented across several overlapping windows rather than at a single boundary.

On padding, the Methods wording needs a clarification: upon further investigation of the final implementation, residual sub-2-s tails at segment boundaries are discarded rather than zero-padded, and the short events they might contain are still captured by the neighboring overlapping windows. We will revise the Methods sentence accordingly and note that no artificial zero-padded patterns enter the model input. The response can close by pointing to the Discussion's existing statement that adaptive segmentation is a direction for future work.

### R2-6 — Per-class sensitivity, specificity, precision, recall, F1

The reviewer is right that a single accuracy figure hides per-class behavior. We will re-evaluate the already-extracted features with patient-grouped repeated evaluation — recordings from the same participant never split across training and evaluation, repeated so that variability can be reported — and provide per-class sensitivity, specificity, precision, F1, and the full confusion matrix as mean ± SD. The class-decision thresholds will be re-derived within the same run with an explicit finite-value guard for classes that are sparse in an evaluation split. This is computation on existing features; no new recordings are involved. The specific confusion figures that Reviewer 3 quotes (non-breathing misread as crackle, wheezing, or healthy) will be re-derived in the same run so that every number in the response is mutually consistent.

### R2-5 — Exact participants and recordings per data split

This looks like a writing item but the numbers must come from the pipeline, so it rides on the R2-6 run. We will produce an explicit table — participants, recordings, and analysis windows per split and per class — directly from the re-evaluation output, and align the Methods text with it, including the distinction between enrolled participants, participants whose recordings entered model training, and the per-split breakdown. The dataset-description sentences will be updated to match the table exactly.

### R3-Major2 — What the non-breathing class is, and what its confusion means for alerts

Two things are being asked: a definition and a consequence. The definition is precise in the pipeline: non-breathing is the complement of the physician-annotated breath phases within each recording — inter-breath silence and ambient background. Speech, motion artifact, and breath-holding were not separately curated, and we will say so explicitly as a limitation; a class that is heterogeneous by construction is expected to show lower single-window separability.

The consequence is about alerts, not windows, and that is where the answer gains ground: actuation decisions are not made per window. With the aggregation rule described under R1-4, transient window-level confusions are suppressed unless they persist across consecutive windows. We will quantify this by re-running the existing recordings offline and reporting alert-level false-positive and false-negative rates at the recording level — the clinically relevant quantity for actuation decisions — alongside the window-level confusion matrix from R2-6.

### R1-4 — Aggregating per-window predictions into clinically meaningful alerts

The reviewer's concern is over-alerting and habituation if every abnormal 2-s window triggers feedback. We will define a simple aggregation rule — an alert fires only when a majority of recent windows (for example K of the last N) agree on the same abnormal class — and demonstrate its effect by re-running existing recordings offline and reporting alert counts before and after aggregation. This is a simulation on data we already have. The clinical meaning of each tactile cue is the actuation team's part of this comment and is handled with R3-Major3.

### R2-8 & R3-Minor3 — Processing latency

Once the update interval is unified between the manuscript and the app (the internal prerequisite above), we will instrument the existing real-time app with timing probes at each stage and report mean and maximum latency from bench playback. Structurally, end-to-end latency is dominated by the 2-s analysis window plus the update interval; the feature-extraction and inference path is a single 64-dimensional vector through the same scaler and ensemble for every class, so there is no class-dependent delay, and we will state that explicitly since the reviewer asked. This is a bench measurement, not a clinical experiment.

---

## Items We Only Support (general answers, no model work)

Three comments touch the model but do not require model work, and per our discussion they can be answered generally. For the sound-versus-diagnosis comments (R1-1 first part, R2-1), the technical input from my side is one sentence: the classifier outputs sound-class probabilities, not disease labels, so the terminology revision ("diagnosis" → "respiratory sound classification") is accurate to the implementation. The same claim-softening pass covers the prevention-claims comment (R2-2). For the labeling-procedure comment (R2-3), the clinical team must confirm the number of annotating physicians and the adjudication procedure; the one model-side fact worth including in that answer is that labels were assigned per recording (dominant adventitious sound with the clinical diagnosis) and propagated to all breathing windows of that recording during segmentation — individual windows were not re-auscultated, which directly answers the reviewer's dominant-sound/any-sound/diagnosis question.

## Out of Scope (other co-authors)

R1-1 second part (behavioral/clinical benefit of feedback), R1-2 (HR/BR gold-standard validation), R1-3 (power/battery), R1-5–7 and R3-Major3/4 (tactile-cue rationale, human-subject validation), R2-4 (demographics), R2-7 (introduction references), R2-10 (patient response to alerts), R3-Minor1 (MEMS spec), R3-Minor2 (cavity simulation), R3-Minor4 (long-term home use). R3-Major1 (sensor placement) is a clinical-protocol item; its generalizability implication connects to the existing Discussion limitation on user-specific variability.

## Summary for the Meeting

| Question | Answer |
|---|---|
| New experiments needed? | **No.** Two computational re-analyses on existing data (R2-6 metrics, R1-4/R3-Major2 alert-level simulation) plus one bench measurement (latency); everything else is literature-backed writing and Methods clarification. |
| Timeline | **2 weeks** from code-set confirmation, with two Week-1 prerequisites: CNUH annotation-procedure confirmation (external) and update-interval unification (internal). |
