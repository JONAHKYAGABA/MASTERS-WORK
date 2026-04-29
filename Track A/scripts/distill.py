"""
scripts/distill.py — self-distillation corpus generator (Stage C).

For each scenario in train.json, prompt the running LLM with the GROUND TRUTH
answer and ask it to construct a reasoning trace that arrives there. Keep
only traces whose final \\boxed{...} matches ground truth AND pass a
structural sanity check. Writes traces/train_traces.jsonl.

Resumable: a second run skips scenarios already accepted.

Run after the LLM server is up (default http://localhost:8001/v1):

    python scripts/distill.py \\
        --train_file data/Phase_1/train.json \\
        --output traces/train_traces.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
sys.path.insert(0, str(PROJECT_DIR))

from main import format_scenario_for_prompt, normalize_multi_answer  # noqa: E402
from utils import extract_answer_all  # noqa: E402

API_KEY = os.environ.get("AGENT_API_KEY", "dummy")

TEACHER_PROMPT = """You are producing training data for a smaller reasoning model.

You are given a 5G drive-test scenario, the candidate optimization actions,
and THE CORRECT ANSWER. Your job is to produce a clean, step-by-step
reasoning trace that arrives at exactly the given answer using only the
data provided.

Constraints:
1. Use ONLY data from the scenario block below. Do not invent timestamps,
   PCIs, cell IDs, or values that are not in the data.
2. Reference at least one specific Timestamp from user_plane_data and at
   least one specific PCI from network_configuration_data.
3. Length 200-1000 tokens. No padding, no apologies, no caveats.
4. Follow this diagnostic procedure: scan user-plane for the throughput
   collapse, classify the failure mode (coverage / interference /
   scheduler) by comparing RSRP, SINR, BLER, MCS, RB count, then map the
   mode to the candidate action targeting the right cell.
5. End the trace with exactly one \\boxed{...} on the last line. The
   boxed value MUST equal the correct answer below.

Correct answer: {gt}
Question type: {qtype}

Scenario data:
{scenario_block}

Candidate options:
{options}

Produce the reasoning trace now."""


# ----------------------------- helpers -----------------------------

def build_options_text(options: List[Dict[str, str]]) -> str:
    return "\n".join(f"{o['id']}: {o['label']}" for o in options)


def question_type(task: Dict[str, Any]) -> str:
    desc = (task or {}).get("description", "") or ""
    return "multiple-answer" if "two to four" in desc.lower() or "select two" in desc.lower() else "single-answer"


_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}")
_PCI_RE = re.compile(r"\b\d{2,4}\b")
_CELL_ID_RE = re.compile(r"\b\d{6,}(?:_\d+)?\b")


def structural_check(trace: str, scenario: Dict[str, Any]) -> Tuple[bool, str]:
    """Return (ok, reason)."""
    if not trace or len(trace) < 200:
        return False, "too_short"
    if len(trace) > 6000:
        return False, "too_long"
    if len(_BOXED_RE.findall(trace)) != 1:
        return False, "boxed_not_unique"
    data = scenario.get("data", {}) or {}
    up = data.get("user_plane_data", "") or ""
    cfg = data.get("network_configuration_data", "") or ""

    # Need at least one shared timestamp prefix from user_plane.
    up_timestamps = set(_TS_RE.findall(up))
    if up_timestamps and not any(ts in trace for ts in up_timestamps):
        return False, "no_timestamp"

    # Need at least one PCI mentioned in trace that's in the configuration.
    cfg_pcis = set(m for m in _PCI_RE.findall(cfg) if 0 < int(m) < 1024)
    trace_pcis = set(m for m in _PCI_RE.findall(trace) if 0 < int(m) < 1024)
    if cfg_pcis and not (trace_pcis & cfg_pcis):
        return False, "no_matching_pci"

    # No hallucinated cell IDs (7+ digit underscored IDs in trace must be in scenario).
    cfg_cells = set(_CELL_ID_RE.findall(cfg))
    trace_cells = set(_CELL_ID_RE.findall(trace))
    if trace_cells and not trace_cells.issubset(cfg_cells | {""}):
        return False, "hallucinated_cell"

    return True, "ok"


def load_done(out_path: Path) -> Dict[str, Dict[str, Any]]:
    done: Dict[str, Dict[str, Any]] = {}
    if not out_path.exists():
        return done
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("scenario_id")
            if sid:
                done[sid] = rec
    return done


# ----------------------------- main -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", default="data/Phase_1/train.json")
    ap.add_argument("--output", default="traces/train_traces.jsonl")
    ap.add_argument("--model_url", default=os.environ.get("MODEL_URL", "http://localhost:8001/v1"))
    ap.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-35B-A3B"))
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Cap the number of scenarios processed (None = all)")
    ap.add_argument("--attempts_per_scenario", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max_tokens", type=int, default=1500)
    args = ap.parse_args()

    train_path = Path(args.train_file)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with train_path.open("r", encoding="utf-8") as f:
        scenarios: List[Dict[str, Any]] = json.load(f)
    if args.max_samples:
        scenarios = scenarios[: args.max_samples]
    print(f"[distill] loaded {len(scenarios)} scenarios from {train_path}")

    done = load_done(out_path)
    print(f"[distill] {len(done)} already accepted in {out_path}; will skip")

    client = OpenAI(base_url=args.model_url, api_key=API_KEY)

    accepted = 0
    rejected = 0
    started = time.time()
    out_f = out_path.open("a", encoding="utf-8")
    summary_path = out_path.parent / "distill_summary.md"

    try:
        for idx, scen in enumerate(scenarios):
            sid = scen.get("scenario_id")
            if not sid or sid in done:
                continue
            gt = scen.get("answer", "") or ""
            task = scen.get("task", {}) or {}
            options = task.get("options", []) or []
            scenario_block = format_scenario_for_prompt(scen)
            prompt = TEACHER_PROMPT.format(
                gt=gt,
                qtype=question_type(task),
                scenario_block=scenario_block,
                options=build_options_text(options),
            )

            this_accepted: Optional[Dict[str, Any]] = None
            this_attempts: List[Dict[str, Any]] = []
            for attempt in range(args.attempts_per_scenario):
                try:
                    resp = client.chat.completions.create(
                        model=args.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                except Exception as e:
                    print(f"[distill] {sid} attempt {attempt+1} API error: {e}")
                    this_attempts.append({"attempt": attempt + 1, "error": str(e)})
                    continue
                trace = resp.choices[0].message.content or ""
                pred = normalize_multi_answer(extract_answer_all(trace))
                gt_norm = normalize_multi_answer(gt)
                ok_struct, why = structural_check(trace, scen)
                this_attempts.append({
                    "attempt": attempt + 1,
                    "pred": pred,
                    "ok_struct": ok_struct,
                    "why": why,
                    "len": len(trace),
                })
                if pred == gt_norm and ok_struct:
                    this_accepted = {
                        "scenario_id": sid,
                        "input": scenario_block,
                        "task": task.get("description", ""),
                        "options": options,
                        "trace": trace.strip(),
                        "answer": gt,
                        "qtype": question_type(task),
                        "attempt_used": attempt + 1,
                    }
                    break

            if this_accepted is not None:
                accepted += 1
                out_f.write(json.dumps(this_accepted, ensure_ascii=False) + "\n")
                out_f.flush()
            else:
                rejected += 1

            if (idx + 1) % 20 == 0 or (idx + 1) == len(scenarios):
                elapsed = time.time() - started
                done_so_far = accepted + rejected
                rate = done_so_far / max(elapsed, 1.0)
                eta = (len(scenarios) - len(done) - done_so_far) / max(rate, 1e-6)
                print(
                    f"[distill] {idx+1}/{len(scenarios)} "
                    f"accepted={accepted} rejected={rejected} "
                    f"{rate*60:.1f}/min eta_remaining={eta/60:.0f}min"
                )
    finally:
        out_f.close()
        elapsed = time.time() - started
        with summary_path.open("w", encoding="utf-8") as f:
            f.write(f"# Distillation summary\n\n")
            f.write(f"- Scenarios processed this run: {accepted + rejected}\n")
            f.write(f"- Already in corpus before run:  {len(done)}\n")
            f.write(f"- Accepted this run:             {accepted}\n")
            f.write(f"- Rejected this run:             {rejected}\n")
            f.write(f"- Total accepted in corpus:      {len(done) + accepted}\n")
            f.write(f"- Wall-clock:                    {elapsed/60:.1f} min\n")
        print(f"[distill] done. accepted={accepted} rejected={rejected} total={len(done)+accepted}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
