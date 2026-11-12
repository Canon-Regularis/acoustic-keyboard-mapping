# Acoustic Keystroke Inference — Project Plan & Literature

> **Goal:** Build a system that recovers which keys a person is typing from the *sound* their
> keyboard makes (a keyboard acoustic side‑channel attack, "ASCA"). This document (1) analyses an
> existing open‑source project, **SneaKey**, (2) lays out what it would take to build something
> more intricate and higher quality than it, and (3) collects the research literature on the topic.

This is a **defensive/academic security** project. Keystroke acoustic emanations are a classic,
well‑published side‑channel. Treat recordings as sensitive, only record people who have consented,
and keep everything local by default.

---

## Table of contents

1. [What the attack is](#1-what-the-attack-is)
2. [Analysis of the SneaKey repo](#2-analysis-of-the-sneakey-repo)
3. [What SneaKey does well / where it stops](#3-what-sneakey-does-well--where-it-stops)
4. [How to go beyond SneaKey — the build plan](#4-how-to-go-beyond-sneakey--the-build-plan)
5. [Suggested target architecture](#5-suggested-target-architecture)
6. [Evaluation protocol you should hold yourself to](#6-evaluation-protocol-you-should-hold-yourself-to)
7. [Full research‑paper bibliography](#7-full-research-paper-bibliography)

---

## 1. What the attack is

Every key, when pressed, radiates a short (~few ms) broadband "click". Because keys sit in different
positions, are struck by different fingers, and excite the keyboard plate differently, the clicks
differ slightly key‑to‑key. The pipeline for every system in the literature is some version of:

```
audio ─► detect/segment keystrokes ─► extract features ─► classify key ─► use language to clean up text
```

The interesting difficulty is not "can you tell two very different keys apart" (you can) — it is
doing it **across keyboards, microphones, rooms, typists, and noise**, with **realistic capture**
(a phone on the desk, a Zoom call), and turning noisy per‑key guesses into **correct text**.

---

## 2. Analysis of the SneaKey repo

Repository: `https://github.com/hypRe1/SneaKey` (analysed from a fresh clone of the default branch).

SneaKey is a **local, consent‑based research prototype** built as a single FastAPI server
(`run.py`, binds `127.0.0.1:8000`) plus a no‑build‑step single‑page web UI (`web/`, ES modules). It
is a clean re‑engineering of an earlier project (referred to internally as "v6" / "the original
project"), with the numerics of the core models deliberately preserved and verified by parity tests.

### 2.1 The four‑stage prediction pipeline

The whole thing is **audio‑only at prediction time** — nothing reads the keystroke log when
predicting (`sneakey/pipeline.py`). The chain is:

```
audio ─► detector (v6) ─► press classifier (v6) ─► CNN key classifier ─► dictionary word decoder
         spectral flux     gradient boosting        log-mel, 49 keys      English words
```

**Stage 1 — Candidate detection** (`sneakey/detection.py`)
- High‑passes the audio at 180 Hz, computes **spectral‑flux novelty** via STFT (`nperseg=384`,
  one value per millisecond), and divides by a 251 ms running median so steady background noise
  maps to ~1 and transients stand out.
- Peaks above a threshold (`flux_threshold`, default ≈1.5) with prominence/distance constraints
  become candidates, each snapped to the loudest millisecond of the 1.2 kHz high‑passed energy.

**Stage 2 — Press vs. release/noise** (`sneakey/detection.py`)
- A **gradient‑boosting classifier** over **13 hand‑engineered features** (log peak/attack/before/
  after/tail energies, attack ratios, crest factor, 3 spectral‑band ratios, and log previous/next
  inter‑event gaps) keeps only candidates that sound like key *presses* (prob ≥ `press_threshold`,
  default 0.35), discarding key *releases* and noise.

**Stage 3 — Which key** (`sneakey/classifier.py`)
- A **small CNN** (`KeyNet`): 3 conv blocks (32→64→128 channels, each = 2×[Conv3×3‑BN‑ReLU] +
  MaxPool), global average pool, dropout 0.3, linear head over ~49 keys.
- Input is a **64‑band log‑mel spectrogram** of the 700 Hz high‑passed audio from 15 ms before to
  45 ms after the press (1 ms hop), **per‑example normalised** (mean/std), and the prediction is
  **averaged over ±2 ms test‑time shifts**. Each press is filtered inside its own 250 ms context so
  a live window and a whole recording give identical results.

**Stage 4 — Words from keys** (`sneakey/decoder.py`)
- A **dictionary word decoder**. Space/punctuation presses split the stream into runs of letters;
  within a run every dictionary word is aligned to the presses with a DP that allows **missed
  presses** and **spurious presses** (fixed penalties), scored by the summed log‑probability of the
  word's letters at the aligned presses **plus the word's Zipf frequency in English**. Runs can hold
  multiple words (missed space), and un‑matchable stretches keep their raw top‑1 letters (so names
  survive). Weights are tuned on validation recordings (`training/tune_decoder.py`).

### 2.2 Ground truth, training, evaluation

- **Ground truth** (`sneakey/groundtruth.py`): the recorder cannot see the mic input delay
  (30–60 ms, varies per session), so it is measured from the audio by aligning logged key‑downs to
  click onsets; each event is snapped to its click. Those snapped positions are the training/eval
  labels.
- **Retraining in‑browser** (`sneakey/training.py`, Train page): retrains the *key* step on your own
  recordings as an **SVM** or **LDA** over 464 hand‑made features (64 band energies + a 10×40 log‑mel
  patch) in seconds. **CNN training is disabled by default** (too slow on a laptop); the shipped CNN
  is still used for prediction. Models live in a registry (`sneakey/models.py`) with manifests of
  which sessions they were trained on, and an active‑model pointer.
- **Evaluation** (`sneakey/evaluation.py`): detection recall/precision (one‑to‑one match within
  20 ms), key accuracy on detected presses (end‑to‑end) and at logged timing (classifier alone, with
  top‑3 and balanced/mean‑per‑key), and CER/WER. Every recording carries a **role badge**
  (`unseen` / `trained` / `validation` / `test` / `locked`) so optimistic scores can't be misread,
  with a `splits.json` and locked sets to protect a final test set.

### 2.3 Reported numbers (their shipped baseline)

On the original project's held‑out test session (**one typist, one laptop, one evening**):

| Metric | Value |
|---|---|
| Detection recall / precision | 77.0% / 85.0% |
| Key accuracy (detected presses, end‑to‑end) | 83.9% |
| Key accuracy (logged timing) | 77.4% (top‑3 90.0%) |
| CER keys only → after dictionary | 38.7% → **29.5%** |
| WER after dictionary | 67.9% |

Model comparison (true press times, held‑out): CNN led at ~60% on ~20k presses vs ~53% for SVM/LDA;
with less data LDA/SVM can match or beat it.

### 2.4 Nice touches most student projects skip

- **Phone‑capture demo** (`sneakey/phone.py`): audio comes **only from a phone** on the same Wi‑Fi,
  so a correct prediction cannot leak from the keystroke log. Cross‑device clock sync via repeated
  server‑clock pings (fastest round trip per 5 s window), then the phone's mic delay + audio‑clock
  drift are recovered from the audio itself. Self‑signed TLS (`sneakey/tls.py`). Honestly reports
  that a laptop‑trained detector finds only ~12% of presses on phone audio vs ~60% after retraining.
- **Procedural sentence generator** (`sneakey/sentences.py`): a probabilistic grammar (12 frames,
  tense/modality/agreement variation, trigram anti‑repetition) so the same key sequences don't recur
  across sessions and a language model can't trivially guess the text.
- **An honest LLM experiment** (`sneakey/llm_decoder.py`): tried replacing the dictionary decoder
  with DeepSeek; it did **worse** (CER 55%+ at best vs 29.5%) because the LLM invented fluent text
  instead of doing evidence alignment. Kept the dictionary decoder. Good negative result to learn
  from.
- Analyse view with per‑stage spectrograms, novelty curve, per‑press inspector (exact CNN input,
  probabilities), per‑word probability distributions, confusion matrix, pooled metrics.

---

## 3. What SneaKey does well / where it stops

**Strengths to keep / match:** audio‑only prediction, phone‑capture so labels can't leak, honest
roles/locked test sets, honest reporting of failure (phone transfer, LLM), reproducibility
(deterministic filtering, parity tests, saved generator seeds), privacy‑by‑default (localhost).

**Where it stops — these are your openings:**

| Limitation in SneaKey | Why it matters |
|---|---|
| **One typist, one laptop, one room** | Numbers don't transfer; it says so. Generalisation is the open problem. |
| **Classic small CNN on log‑mel** | SOTA is **CoAtNet / Swin / MOAT (conv + self‑attention)** reaching ~95%. |
| **Dictionary decoder only** | SOTA uses **char‑level BERT + LLM error correction** to fix noisy predictions. |
| **Lowercase only; no Shift/Caps; digits & punctuation starved** | Can't recover real passwords, casing, symbols. |
| **Detection misses ~20% of presses, adds spurious ones** | Their own note: most error is **detection/segmentation**, not classification. |
| **No noise robustness** | Breaks with speech, music, fans, overlapping keystrokes. |
| **Supervised, labelled, per‑keyboard** | Needs target‑keyboard labels; **unsupervised/self‑supervised** attacks don't. |
| **No defences / countermeasures studied** | A high‑quality project evaluates mitigations too. |
| **Single capture channel (local mic / one phone)** | No Zoom/VoIP, no mic array, no through‑wall, no cross‑device transfer study. |

---

## 4. How to go beyond SneaKey — the build plan

Ordered roughly by impact. Each item says **what SneaKey did** → **what "more intricate" looks like**.

### 4.1 Data — the single biggest lever
- [ ] **Multi‑keyboard, multi‑mic, multi‑room, multi‑typist corpus.** SneaKey: one of each. Build a
  capture protocol covering membrane, scissor‑switch (laptop) and mechanical (various switches:
  linear/tactile/clicky) keyboards, several mics (laptop, phone, headset, cheap USB, mic array),
  several rooms, and ≥10 typists (hunt‑and‑peck vs touch‑typists).
- [ ] **Full key set + casing.** Add digits, punctuation, Shift/Caps‑Lock reconstruction, modifiers,
  Backspace semantics. Record enough per‑key examples for rare keys.
- [ ] **Realistic capture channels.** Nearby phone (SneaKey has this), **Zoom/VoIP** (compression +
  noise suppression change the signal), far‑field, and **through‑wall / adjacent‑room** audio.
- [ ] **Automatic ground‑truth labelling at scale.** SneaKey hand‑configures sessions. Borrow the
  2026 **OCR + video** idea: record screen/video, OCR the typed text, align to audio to auto‑label —
  removes the labelling bottleneck and lets you scale to hours of data.
- [ ] **Published, versioned dataset** with per‑recording metadata (keyboard, mic, room, typist,
  SNR). Compare against / extend public sets (MKA, the Harrison phone/Zoom set).
- [ ] **Augmentation:** time/frequency masking (SpecAugment), random time‑shift, room impulse
  responses (reverb), additive noise at controlled SNRs, mixup of overlapping keystrokes.

### 4.2 Detection & segmentation (SneaKey's weakest link, by its own admission)
- [ ] Replace the fixed spectral‑flux + gradient‑boosting detector with a **learned onset/segmentation
  model** (e.g. a small temporal CNN / CRNN / U‑Net over the spectrogram) trained to find press
  onsets and separate **overlapping keystrokes** during fast typing.
- [ ] Model **press vs. release** jointly with timing, and use the **release** sound as extra signal
  (most projects throw it away).
- [ ] Joint **detect‑and‑classify** (CTC‑style or detection‑transformer) so segmentation errors stop
  dominating end‑to‑end error.
- [ ] Robust onset detection under noise (median/RPCA background subtraction, adaptive thresholds).

### 4.3 Features / representation
- [ ] SneaKey: 64‑band log‑mel + hand features. Add/compare **MFCCs, cross‑correlation features**
  (shown to beat MFCC/FFT for clustering), **CWT/scalogram**, learnable front‑ends (SincNet,
  LEAF), and **224×224 mel images** for transformer backbones.
- [ ] **TDoA / multi‑microphone** features: with ≥2 mics you get key *position* via time‑difference‑
  of‑arrival — a strong, typist‑independent cue SneaKey's single mic can't use.
- [ ] Per‑keystroke **two‑time‑scale** features (coarse whole‑event + fine peak‑level), as in recent
  RF/acoustic work.

### 4.4 Key‑classification model
- [ ] Swap the small CNN for **CoAtNet / Swin Transformer / MOAT** (conv + self‑attention hybrids) —
  the current SOTA for this exact task (~95% clean, 95.23% reported on the standard benchmark).
- [ ] **Keyboard‑geometry‑aware loss** (penalise confusions by physical key distance — "revisited"
  idea) to exploit that nearby keys sound alike.
- [ ] **Calibrated probabilities** (temperature scaling) so the language stage gets honest
  uncertainty — SneaKey already feeds probabilities to the decoder; make them well‑calibrated.
- [ ] **Multi‑task:** predict key + finger + hand + row simultaneously for regularisation.

### 4.5 Language / decoding (turn noisy keys into correct text)
- [ ] Keep the dictionary decoder as a baseline, but add an **n‑gram / neural character LM** beam
  search over the per‑press distributions (not top‑1).
- [ ] **Character‑level BERT** that consumes *ambiguity‑aware* embeddings (the full per‑key
  distribution, not a hard decision) — the 2026 self‑supervised approach, before any hard commit.
- [ ] **LLM error‑correction as a post‑processor** ("typo correction"), done right: give it the
  candidate text + alternatives and ask it to correct, not to free‑write. SneaKey's LLM failed
  because it invented text; the WOOT'25 framing (few‑shot "fix the typos", or **LoRA‑fine‑tuned
  small LLM**) is the method that works and matches GPT‑4o with a model 67× smaller.
- [ ] **Iterative acoustic↔language loop:** high‑confidence LM decisions feed back to refine the
  cluster→character mapping (self‑supervised, label‑free).

### 4.6 Generalisation & threat‑model realism (the research contribution)
- [ ] **Cross‑keyboard / cross‑mic / cross‑typist transfer learning + domain adaptation.** Report
  *unseen‑device* numbers, not just same‑device. This is where SneaKey explicitly doesn't go.
- [ ] **Unsupervised / self‑supervised attack** that needs **no labelled data from the target
  keyboard** — cluster keystrokes, resolve with language context. Recent work hits >99% on English
  text with ~100+ keystrokes and no target labels; ~90% through‑wall / over a call.
- [ ] **Noise robustness** as a first‑class metric: evaluate at a range of SNRs with speech/music/
  fan/overlapping‑typing noise, not just clean audio.
- [ ] **Password / passphrase recovery** mode (no language model to lean on) with dictionary +
  brute‑force‑beating search, and **PIN‑pad / numeric** mode.
- [ ] **VoIP channel modelling:** run audio through Zoom/Opus codecs + noise suppression and measure
  the hit.

### 4.7 Defences / countermeasures (what makes it "high quality", not just an attack)
- [ ] Implement and *evaluate* mitigations: random **fake‑keystroke injection**, **masking sounds**/
  active noise, keystroke‑sound **randomisation/warping**, and typing‑style changes (touch‑typing).
- [ ] Quantify how much each defence lowers attacker accuracy — a proper attack↔defence study.

### 4.8 Evaluation & engineering rigour (keep SneaKey's good habits, add more)
- [ ] Keep roles/locked test sets; add **cross‑validation across the generalisation axes** above.
- [ ] Report **per‑key recall + confusion matrices + balanced accuracy** (SneaKey does) **per
  device/typist**, plus **top‑k**, **CER/WER**, and **password success‑within‑N‑guesses**.
- [ ] Pre‑register splits; publish seeds, configs, and the dataset; parity/regression tests (SneaKey
  has these — match them).
- [ ] Statistical significance across typists/sessions (confidence intervals, not single numbers).
- [ ] Reproducible training (containerised), model cards + dataset datasheet, and a written
  **ethics/consent** section.

### 4.9 Stretch goals
- [ ] **Multi‑modal fusion:** combine acoustic with video (screen/finger), EM, Wi‑Fi CSI, or motion
  sensors for a stronger or label‑free attack.
- [ ] **Real‑time streaming** inference with latency budget.
- [ ] **Cross‑lingual** text (SneaKey is English‑only via its dictionary).

---

## 5. Suggested target architecture

```
                 ┌─────────────── capture ───────────────┐
                 │ laptop mic │ phone │ Zoom/VoIP │ array │
                 └───────────────────┬────────────────────┘
                                     ▼
            ┌──────────────── pre‑processing ───────────────┐
            │ resample · band‑pass · denoise/RPCA · (TDoA if │
            │ multi‑mic) · augmentation (noise/reverb/mask)  │
            └───────────────────────┬────────────────────────┘
                                    ▼
            learned onset/segmentation (CRNN/U‑Net)  ──►  overlapping‑press handling
                                    ▼
            mel / MFCC / cross‑corr  ──►  CoAtNet / Swin backbone (+ geometry‑aware loss)
                                    ▼
            calibrated per‑press key distributions
                                    ▼
            ambiguity‑aware char‑BERT  ──►  beam search over LM  ──►  LLM (LoRA) typo‑correction
                                    ▼
                               recovered text
                                    │
            ┌───────────────── evaluation harness ──────────────────┐
            │ cross‑device / cross‑typist / SNR sweeps · CER/WER ·   │
            │ per‑key confusion · password success@N · defences     │
            └────────────────────────────────────────────────────────┘
```

---

## 6. Evaluation protocol you should hold yourself to

1. **Never test on the training keyboard/typist and call it generalisation.** Report both.
2. Report **CER and WER**, detection recall/precision, per‑key balanced accuracy, and top‑k.
3. Sweep **SNR** with realistic noise; report a curve, not one number.
4. For passwords: **success within N guesses** (N = 1, 10, 50) with and without a dictionary.
5. Hold out **locked** sessions (SneaKey‑style) touched only once at the very end.
6. Confidence intervals across typists/sessions.
7. Measure each **defence's** effect on attacker accuracy.

---

## 7. Full research‑paper bibliography

> Compiled from the two field surveys (Taheritajar et al. 2023; Abdullahi et al. 2026) plus direct
> search of the recent literature. The surveys are the canonical aggregators — start there if you
> want to be exhaustive. Acoustic‑keystroke papers are listed first; closely related non‑acoustic
> keystroke side‑channels and background are grouped at the end and clearly marked. No automatically
> compiled list can be guaranteed 100% complete, but this covers the foundational work through the
> 2025–2026 state of the art.

### 7.1 Surveys & datasets (read these first)

- **Taheritajar, A., Harris, Z. M., Rahaeimehr, R.** "A Survey on Acoustic Side Channel Attacks on
  Keyboards." arXiv:2309.11012, 2023. *(The canonical survey — its bibliography is the backbone of
  this list.)*
- **Abdullahi et al.** "A Survey on Acoustic Side-Channel Attacks: An Artificial Intelligence
  Perspective." *Journal of Cybersecurity and Privacy* (MDPI) 6(1):6, 2026. doi:10.3390/jcp6010006.
- **Hama Rawf, K. M., et al.** "Multi-datasets for different keyboard key sound recognition
  (Multi‑Keyboard Acoustic / MKA Datasets)." *Data in Brief*, 2024. doi:10.1016/j.dib.2024.110949.
- **Monaco, J. V.** "SoK: Keylogging Side Channels." *IEEE Symposium on Security and Privacy (S&P)*,
  2018. *(Systematisation across acoustic/EM/timing/motion.)*

### 7.2 Foundational acoustic keystroke attacks

- **Asonov, D., Agrawal, R.** "Keyboard Acoustic Emanations." *IEEE Symposium on Security and
  Privacy (S&P)*, 2004. doi:10.1109/SECPRI.2004.1301311. *(The origin — neural net, FFT features.)*
- **Zhuang, L., Zhou, F., Tygar, J. D.** "Keyboard Acoustic Emanations Revisited." *ACM CCS 2005*;
  journal version *ACM TISSEC* 13(1), 2009. *(Unsupervised + language model; recovered up to ~96%.)*
- **Berger, Y., Wool, A., Yeredor, A.** "Dictionary Attacks Using Keyboard Acoustic Emanations."
  *ACM CCS*, 2006. *(Cross‑correlation + dictionary; no labelled training data.)*
- **Fiona (Au Hiu Yan).** "Keyboard Acoustic Triangulation Attack." CUHK bachelor's thesis, 2006.
- **Neale, et al.** "Investigating the Feasibility of Keyboard Acoustic Attacks." Univ. of Auckland,
  2006.
- **Kelly, A.** "Cracking Passwords using Keyboard Acoustics and Language Modeling." MSc thesis,
  University of Edinburgh, 2010.
- **Halevi, T., Saxena, N.** "A Closer Look at Keyboard Acoustic Emanations: Random Passwords, Typing
  Styles and Decoding Techniques." *ACM ASIACCS*, 2012.
- **Halevi, T., Saxena, N.** "Keyboard Acoustic Side Channel Attacks: Exploring Realistic and
  Security‑Sensitive Scenarios." *International Journal of Information Security* 14(5), 2015.
- **Zhu, T., Ma, Q., Zhang, S., Liu, Y.** "Context‑free Attacks Using Keyboard Acoustic Emanations."
  *ACM CCS*, 2014. *(Uses inter‑keystroke timing + TDoA; no per‑key training.)*
- **Ponnam, et al.** "Keyboard Acoustic Emanations Attack: An Empirical Study." 2013.
- **Wit, de, & Houtenbos.** "All Your Keystrokes Are Belong To Us." 2014.

### 7.3 Deep‑learning era (CNN / transformer / LLM)

- **Martinasek, Z., Clupek, V., Trasy, K.** "Acoustic Attack on Keyboard Using Spectrogram and Neural
  Network." *TSP (Telecommunications and Signal Processing)*, 2015.
- **Giallanza, T., et al.** "Keyboard Snooping from Mobile Phone Arrays with Mixed Convolutional and
  Recurrent Neural Networks." *ACM IMWUT* 3(2), 2019. *(CNN detect + RNN identify.)*
- **Slater, D., et al.** "Robust Keystroke Transcription from the Acoustic Side‑Channel." *ACSAC*,
  2019. *(End‑to‑end segmentation + classification; CER 7.41% known / 15.41% unknown typist.)*
- **Harrison, J., Toreini, E., Mehrnezhad, M.** "A Practical Deep Learning‑Based Acoustic Side
  Channel Attack on Keyboards." *IEEE EuroS&PW (SiLM'23)*, 2023. arXiv:2308.01074.
  doi:10.1109/EuroSPW59978.2023.00034. *(CoAtNet on mel‑spectrograms; 95% phone, 93% Zoom — the
  headline modern result and the standard benchmark dataset.)*
- **Taheritajar, A., Rahaeimehr, R.** "Acoustic Side Channel Attack on Keyboards Based on Typing
  Patterns." arXiv:2403.08740, 2024; *Springer* (conference volume), 2026. *(Incorporates typing
  style as a parameter.)*
- **Park, J. H., Ayati, S. A., Cai, Y.** "Improving Acoustic Side‑Channel Attacks on Keyboards Using
  Transformers and Large Language Models." arXiv:2502.09782, 2025. *(Vision transformers + LLMs;
  CoAtNet SOTA + LoRA‑tuned small LLM. Note: arXiv version was **withdrawn** over a preprocessing
  error in its noise analysis — read alongside the WOOT'25 paper below, which supersedes it.)*
- **Ayati, S. A., Park, J. H., Cai, Y., Botacin, M.** "Making Acoustic Side‑Channel Attacks on Noisy
  Keyboards Viable with LLM‑Assisted Spectrograms' 'Typo' Correction." *19th USENIX WOOT*, 2025.
  arXiv:2504.11622. *(Vision transformers for classification + LLM 'typo' correction for noise; LoRA
  small LLM ≈ GPT‑4o.)*
- **Zychowski, et al.** "Acoustic Side‑Channel Attacks (robustness/generalisation of CoAtNet, MOAT,
  Swin)." *IJCNN*, 2026. *(New SOTA 95.23% on the standard benchmark; LLM semantic/syntactic
  post‑correction toward near‑zero error under noise.)*
- **(Authors per arXiv)** "Transforming Keystroke Noise to Text: Self‑Supervised Acoustic
  Eavesdropping Attacks on Keyboards." arXiv:2607.22094, 2026. *(Label‑free: cluster + ambiguity‑
  aware char‑level BERT + LLM correction + iterative feedback; >99% on English text from ~100+
  keystrokes, ~90% through‑wall / over a call.)*
- **Fürst, D., Aßmuth, A.** "Practical Acoustic Eavesdropping On Typed Passphrases."
  arXiv:2503.16719, 2025. *(Unsupervised; cross‑correlation beats MFCC/FFT for clustering; dictionary
  attack on passphrases.)*
- **Vranješ, D., Stančić, I., Bugarić, M., Perković, T.** "Automated Acoustic Side‑Channel Attack on
  Keyboard Inputs via Combined Video–Audio Analysis." Preprints.org 202606.1478, 2026. *(OCR of
  screen video auto‑labels keystroke audio; CNN + transfer learning — removes the labelling
  bottleneck.)*
- **Akinbi, A., Deniz, E., Ismael, A. M., Rashid, Z. N., Sengur, A.** "Password‑Sniffing Acoustic
  Keylogger Using Machine Learning." SSRN 4431909, 2023.

### 7.4 VoIP / remote (Skype, Zoom)

- **Compagno, A., Conti, M., Lain, D., Tsudik, G.** "Don't Skype & Type! Acoustic Eavesdropping in
  Voice‑Over‑IP." *ACM ASIACCS*, 2017.
- **Cecconello, S., Compagno, A., Conti, M., Lain, D., Tsudik, G.** "Skype & Type: Keyboard
  Eavesdropping in Voice‑Over‑IP." *ACM TOPS* 22(4), 2019.
- **Anand, S. A., Saxena, N.** "Keyboard Emanations in Remote Voice Calls: Password Leakage and
  Noise(less) Masking Defenses." *ACM CODASPY*, 2018.
- **Sabra, M., Maiti, A., Jadliwala, M.** "Zoom on the Keystrokes: Exploiting Video Calls for
  Keystroke Inference Attacks." arXiv:2010.12078, 2020.

### 7.5 Triangulation / TDoA / microphone arrays / single phone

- **Liu, J., Wang, Y., Kar, G., Chen, Y., Yang, J., Gruteser, M.** "Snooping Keystrokes with mm‑level
  Audio Ranging on a Single Phone." *ACM MobiCom*, 2015. *(94% keystroke recovery via TDoA.)*
- **Wang, J., et al.** "Accurate Combined Keystrokes Detection Using Acoustic Signals." *MSN*, 2016.
- **Rosmansyah, Y., et al.** "The Microphone Array Sensor Attack on Keyboard Acoustic Emanations."
  *ICITSI*, 2017.
- **Qin, et al.** "LOL: Localization‑free Online Keystroke Tracking Using Acoustic Signals." *Soft
  Computing* 23(21), 2019.
- **Cheng, P., et al.** "Dictionary Attacks Based on TDOA Using a Smartphone." *IEEE IAEAC*, 2022.
- **Narain, S., Sanatinia, A., Noubir, G.** "Single‑Stroke Language‑Agnostic Keylogging Using
  Stereo‑Microphones and Domain Specific Machine Learning." *ACM WiSec*, 2014.
- **Bai, J., Liu, J., Song, W.** "I Know Your Keyboard Input: A Robust Keystroke Eavesdropper Based on
  Acoustic Signals." *ACM Multimedia*, 2021.

### 7.6 Touchscreen / smartphone acoustic keystroke

- **Lu, L., et al.** "KeyListener: Inferring Keystrokes on QWERTY Keyboard of Touch Screen through
  Acoustic Signals." *IEEE INFOCOM*, 2019.
- **Yu, et al.** "An Indirect Eavesdropping Attack of Keystrokes on Touch Screen through Acoustic
  Sensing." *IEEE Transactions on Mobile Computing* 20(2), 2019.
- **Kim, Joe, Liu.** "TapSnoop: Leveraging Tap Sounds to Infer Tapstrokes on Touchscreen Devices."
  *IEEE Access* 8, 2020.
- **Gupta, H., et al.** "Deciphering Text from Touchscreen Key Taps." *DBSec*, 2016.
- **Shumailov, I., et al.** "Hearing Your Touch: A New Acoustic Side Channel on Smartphones."
  arXiv:1903.11137, 2019.
- **Zarandy, A., Shumailov, I., Anderson, R.** "Hey Alexa What Did I Just Type? Decoding Smartphone
  Sounds with a Voice Assistant." arXiv:2012.00687, 2020.

### 7.7 PIN pads / ATM / keypad acoustic

- **de Souza Faria, G., Kim, H. Y.** "Differential Audio Analysis: A New Side‑Channel Attack on PIN
  Pads." *International Journal of Information Security* 18(1), 2019.
- **de Souza Faria, G., Kim, H. Y.** "Identification of Pressed Keys by Acoustic Transfer Function."
  *IEEE SMC*, 2015.
- **Panda, S., et al.** "Behavioral Acoustic Emanations: Attack and Verification of PIN Entry Using
  Keypress Sounds." *Sensors* 20(11), 2020.
- **Cardaioli, M., et al.** "Your PIN Sounds Good! On the Feasibility of PIN Inference through Audio
  Leakage." arXiv:1905.08742, 2019.
- **Ranade, Smith, Switala.** "Acoustic Side Channel Attack on ATM Keypads." 2009.
- **Anand, S. A., Shrestha, B., Saxena, N.** "Bad Sounds Good Sounds: Attacking and Defending
  Tap‑Based Rhythmic Passwords Using Acoustic Signals." *ICCNS*, 2015.

### 7.8 Active acoustic & related acoustic side‑channels

- **Cheng, P., et al.** "SonarSnoop: Active Acoustic Side‑Channel Attacks." *International Journal of
  Information Security* 19(2), 2020.
- **Zhou, M., et al.** "PatternListener: Cracking Android Pattern Lock Using Acoustic Signals." *ACM
  CCS*, 2018.
- **Toreini, E., Randell, B., Hao, F.** "An Acoustic Side Channel Attack on Enigma." Newcastle Univ.
  technical report, 2015. *(ANN on MFCCs, 92.18% on Enigma keys.)*
- **Das, A., Borisov, N., Caesar, M.** "Do You Hear What I Hear? Fingerprinting Smart Devices Through
  Embedded Acoustic Components." *ACM CCS*, 2014.
- **Schlegel, R., et al.** "Soundcomber: A Stealthy and Context‑Aware Sound Trojan for Smartphones."
  *NDSS*, 2011.
- **Simon, L., Anderson, R.** "PIN Skimmer: Inferring PINs Through the Camera and Microphone." *ACM
  SPSM*, 2013.

### 7.9 Defences / countermeasures

- **Anand, S. A., Saxena, N.** "A Sound for a Sound: Mitigating Acoustic Side Channel Attacks on
  Password Keystrokes with Active Sounds." *Financial Cryptography (FC)*, 2016.
- "A Prototype for Generating Random Key Sounds to Prevent Keyboard Acoustic Side‑Channel Attacks."
  *(listed in the AI‑perspective survey — synthetic/noise‑based countermeasure.)*
- Mitigations discussed in Harrison et al. 2023 and the WOOT'25 paper: fake‑keystroke injection,
  masking/active sound, typing‑style change, randomised (non‑dictionary) passwords.

### 7.10 Closely related non‑acoustic keystroke side‑channels (for context / multi‑modal work)

- **Song, D. X., Wagner, D., Tian, X.** "Timing Analysis of Keystrokes and Timing Attacks on SSH."
  *USENIX Security*, 2001.
- **Foo Kune, D., Kim, Y.** "Timing Attacks on PIN Input Devices." *ACM CCS*, 2010.
- **Liu, X., et al.** "When Human Cognitive Modeling Meets PINs: User‑Independent Inter‑Keystroke
  Timing Attacks." *Computers & Security* 80, 2019.
- **Ali, K., Liu, A. X., Wang, W., Shahzad, M.** "Keystroke Recognition Using WiFi Signals." *ACM
  MobiCom*, 2015. *(Wi‑Fi CSI channel.)*
- **Yang, et al.** "Wireless Training‑Free Keystroke Inference Attack and Defense" (Wi‑Fi CSI; 95%
  word recovery after ~150 words, unsupervised).
- **Marquardt, P., et al.** "(sp)iPhone: Decoding Vibrations From Nearby Keyboards Using Mobile Phone
  Accelerometers." *ACM CCS*, 2011. *(Motion/vibration side‑channel.)*
- **Li, et al.** "Periscope: A Keystroke Inference Attack Using Human‑Coupled Electromagnetic
  Emanations." *ACM CCS*, 2021. *(EM channel.)*
- Keystroke inference via **video** (e.g. "Towards a General Video‑Based Keystroke Inference Attack,"
  *USENIX Security*, 2023) and **RF/backscatter** (e.g. "RadKey," arXiv:2606.10148, 2026) — relevant
  for multi‑modal fusion.

---

*Prepared as a planning document. Verify each citation's exact venue/year against the paper before
using it in a formal write‑up — a few entries from the surveys carry incomplete publication details
in the original.*
