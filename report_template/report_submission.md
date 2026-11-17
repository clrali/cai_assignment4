# Assignment 4 report: Hartwell Voice Assistant

**Partners:** Clara Li (`cli725`) and Oankar Patil (`opatil31`)  
**Models:** 
LLM: `meta-llama/Llama-3.2-3B-Instruct`. 
ASR: `faster-whisper-small.en` from the shared cache. 
TTS: `Piper en_US lessac medium voice` from the shared cache.  
**Run behind the figures:** Job ID: 7125302 on one NVIDIA H100 80 GB HBM3; 40 turns; trace: `run_output/trace.jsonl`.

## 1. The system
Each run replays recorded utterances through the full pipeline. ASR transcribes each utterance once the speaker is done. Python rules identify common booking intents, and LLM requests extract structured details or interpret other requests. The task state is then updated, and an LLM call streams the spoken answer, which Piper synthesizes. ASR and interpretation finish before answer generation starts; TTS can overlap answer generation once a sentence is ready.

The task state lasts for the session and includes timer deadlines, reminders, and room-booking constraints, alternatives and proposals. Python checks capacity, equipment and availability before proposing a room and requires confirmation before booking. The general planner sees the last six conversation messages plus a state snapshot. The booking extractor instead sees the current utterance, the feature vocabulary and the pending field, with a JSON schema constraining its output. Python merges the extracted changes with existing constraints and checks for unstated values. Booking responses are built from the verified state and streamed with constrained decoding to preserve their exact text; room-information questions can skip LLM planning. Lookup answers receive the knowledge base.

All LLM calls use temperature 0, seed 42, and a 256-token limit. ASR and Piper each get eight CPU threads, with one ASR worker and one ONNX inter-op thread. A single TTS worker synthesizes each sentence as soon as sentence-ending punctuation followed by whitespace appears, and sends any leftover text when the stream ends. Before timing starts, a warm-up pass transcribes the first recording, exercises JSON mode, both planning schemas and constrained spoken output, and synthesizes "Ready."

## 2. The latency waterfall

![Stage contributions to p50 and p95 TTFA, with overlapping LLM generation shown separately](../run_output/evaluation/waterfall.svg)

| Interval | At p50 TTFA (ms) | At p95 TTFA (ms) |
|---|---:|---:|
| **TTFA, all turns** | 327.78 | 473.66 |
| ASR: `asr_final` − `end_of_user_speech` | 45.75 | 60.94 |
| LLM start: `llm_first_token` − `asr_final` | 122.58 | 202.69 |
| Speech start: `tts_first_audio` − `llm_first_token` | 135.16 | 256.49 |
| Overlapping LLM generation: `llm_complete` − `llm_first_token` | 50.41 | 125.22 |

**Turns counted:** 40 of 40

**How I treat stages that overlap:** `llm_first_token` marks the first nonblank spoken-answer token, so LLM start includes any interpretation requests and Python state handling before the spoken answer. Sentence synthesis can overlap answer generation, but first audio followed LLM completion in all 40 measured turns. The LLM remaining stage overlaps the speech-start interval and is never added to the three TTFA stages.

**How to recompute these numbers:** From the repo root, run `python3 evaluate_run.py`. The evaluation is dependent on `run_output/trace.jsonl`. Outputs are found in `summary.json` in the `run_output/evaluation/` folder.

## 3. Word error rate on our own voices

| Speaker | Utterances | Reference words | Pooled WER | Mean WER per utterance |
|---|---:|---:|---:|---:|
| Clara Li | 20 | 213 | 0.94% (2/213) | 1.08% |
| Oankar Patil | 20 | 171 | 5.85% (10/171) | 6.74% |

The ASR handles Oankar’s voice slightly worse. The difference was sometimes coming from the grammar of the utterance, like a reference of “makerspace” being transcribed into “maker space,” or in utterances that included Oankar’s name, which would be misspelled in the transcription because it’s a less common name. An example utterance of the latter was a reference of “yes please book it under the name oankar” was transcribed into “yes please book it under the name omkar.” The associated recording also had less emphasis placed on the -an part of Oankar, which could also contribute to a transcription of omkar instead of a more phonetically accurate misspelling of onkar.

## 4. The bottleneck

**The dominant stage:** In the p50 waterfall, LLM start contributes the largest share at **182.59 ms**, or **55.71%** of the **327.78 ms** TTFA. Speech start dominates the p95 waterfall at **271.58 ms / 473.66 ms ≈ 57.34%**.

**Why:** LLM start contains completed interpretation requests, Python state updates and the wait for the first answer token. Speech start contains sentence buffering, TTS scheduling and synthesis up to the first audible chunk. First audio follows answer completion by a median 85.36 ms, showing that the wait extends beyond text generation. 

**The fix I would ship next:** Test a more compact JSON plan that excludes unused fields and null values, particularly in booking extraction. The agent waits for interpretation to finish before requesting the spoken answer, so generating fewer planning tokens could shorten the LLM start interval. Add timestamps around the interpretation requests and the start of the answer request to measure where the time is spent, then compare p50 LLM start, p50 TTFA and task correctness with the current version. The goal is to reduce the LLM start contribution to the p50 waterfall; the actual saving would need to be verified experimentally.

**One thing I tried that did not help:** I implemented sentence-level streaming so Piper could synthesize completed sentences while the LLM continued generating the answer. However, none of the 40 turns produced audible speech before LLM completion. The first audio arrived at a median of 85.36 ms afterward. This shows that streaming didn't achieve speech before the full answer was generated. 

## 5. Reproducing this

From the repository root on PACE-ICE, with access to the course's shared model cache and `CAI_A4` environment:

```bash
module load apptainer
export SCRATCH_DIR="$HOME/scratch"
export APPTAINER_CACHEDIR="$SCRATCH_DIR/cache"
export APPTAINER_TMPDIR="$SCRATCH_DIR/tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" run_output
apptainer build --fakeroot "$SCRATCH_DIR/agent.sif" recipe/agent.def
sbatch --export=ALL,OUTPUT_DIR=run_output/reproduction recipe/run_agent.sbatch
```

After the job finishes:

```bash
python3 evaluate_run.py --trace run_output/reproduction/trace.jsonl
python3 wer/score_wer.py --trace run_output/reproduction/trace.jsonl --references references.json --details --json
```

Use a fresh `OUTPUT_DIR` for each replay. The script starts the local LLM server, selects its port from the job ID, waits for readiness and passes the allocated CPU count to the agent. It sets offline model access and disables user-site packages inside Apptainer. Dependency versions are in `recipe/requirements.txt`; vLLM comes from the shared `CAI_A4` environment. NumPy, the LLM requests and the server use seed 42; ASR uses temperature 0. The recorded job completed in **3 minutes 53 seconds**, including startup; building the image is a separate step.

## Appendix: AI-use disclosure

**Tools used:** Codex, the AI coding assistant used in this workspace. Its exact model version was not recorded in the project artifacts.

**What for:** Codex helped debug and implement `agent.py`, including booking extraction, constraint updates, confirmation handling and grounded responses. It also helped explain latency accounting, check run artifacts, and draft and revise this report, including the numerical tables and bottleneck discussion. AI assistance therefore covered both code and report prose.

**What I wrote or decided myself:** We chose the separation between request interpretation and spoken answers, the models and tools to use, and the amount of knowledge-base context supplied to the system, as described in our earlier report. Codex assisted with later implementation changes and report wording; the resulting code and analysis are not solely human-written.

**How I checked it:** We replayed our recordings through the local pipeline. For job 7125302, all 40 trace records passed validation; recomputed latency summaries and WER matched `run_output/evaluation/summary.json` and `wer.json`. The report's table was checked against the waterfall contributions, and implementation descriptions were checked against `agent.py` and the job script. Trace validation checks timing and required fields, not correctness of every spoken answer; the optimization proposal is identified as unmeasured.
