"""
scripts/agentic_agent.py — agentic LLM agent for Track A.

Pipeline per scenario:
  1. Build a tight prompt: truncated scenario data + options + tool descriptors
  2. Ask the LLM with tools=[judge_mainlobe_or_not, calculate_overlap_ratio,
     calculate_pathloss]. Temperature 0.
  3. If the LLM responds with tool_calls, execute them against server.py
     (port 7860 by default), feed results back as a second turn.
  4. Extract \\boxed{Cx} (single) or \\boxed{Cx|Cy|Cz} (multi) from the final
     response. If empty / malformed, fall back to the deterministic heuristic.
  5. Per-scenario hard timeout 60s.

Outputs:
  <out_dir>/completions.jsonl     # resumable per-scenario log
  <out_dir>/result.csv            # scenario_id, answers
  <out_dir>/result_v1_raw.csv     # same
  <out_dir>/result_v2_multi_recall.csv
  <out_dir>/result_v3_insurance.csv

Auto-scores against ground truth if scenarios are labeled.

Run:
    python scripts/agentic_agent.py \\
        --test_file data/Phase_1/test.json \\
        --out_dir   eval/results/agentic

Assumes llm_server is up at $LLM_URL (default http://localhost:8001) and
server.py is up at $TOOL_URL (default http://localhost:7860).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
sys.path.insert(0, str(PROJECT_DIR))

# Heuristic fallback (validated ~0.30 mean IoU on train)
from scripts.build_baseline_submission import pick_answer as heuristic_pick  # noqa: E402
from scripts.build_baseline_submission import is_multi as task_is_multi  # noqa: E402


# --------------------------------------------------------------------- prompts

SYSTEM_PROMPT_SINGLE = (
    "You are a 5G RAN troubleshooting expert. Read the scenario data and "
    "options. You MAY call any of the provided diagnostic tools to confirm "
    "your hypothesis (judge_mainlobe_or_not, calculate_overlap_ratio, "
    "calculate_pathloss, calculate_horizontal_angle, calculate_tilt_angle, "
    "optimize_antenna_gain, plus data accessors like get_serving_cell_rsrp, "
    "get_kpi_data, get_mr_data, get_signaling_plane_event_log, etc.). "
    "Call at most 2 tools — they cost wall-clock time. Then output ONLY "
    "the final answer on the last line as: \\boxed{Cx}\n"
    "Procedure:\n"
    "1. Scan user_plane_data for the throughput collapse (>50% drop). "
    "Note t_drop and serving PCI.\n"
    "2. Classify: COVERAGE (RSRP drops >6dB AND SINR drops >5dB), "
    "INTERFERENCE (SINR drops >5dB, RSRP stable), or "
    "SCHEDULER/PDCCH (RSRP+SINR healthy but RB count drops).\n"
    "3. Optionally call 1-2 tools.\n"
    "4. Map mode -> matching option on the right cell.\n"
    "Reasoning <200 tokens. End with \\boxed{Cx}."
)

SYSTEM_PROMPT_MULTI = (
    "You are a 5G RAN troubleshooting expert. Read the scenario data and "
    "options. You MAY call any of the provided diagnostic tools. Call at "
    "most 2. Then output ONLY the final answer on the last line as "
    "\\boxed{Cx|Cy|Cz} with 2 to 4 options in ASCENDING numeric order "
    "separated by pipes (no spaces).\n"
    "Scoring is intersection-over-union — missing a correct option costs "
    "as much as adding a wrong one. Pick the 2-3 most likely options.\n"
    "Procedure:\n"
    "1. Scan user_plane_data for the throughput collapse.\n"
    "2. Classify failure mode (COVERAGE, INTERFERENCE, SCHEDULER).\n"
    "3. Optionally call 1-2 tools.\n"
    "4. Pick all options matching the mode + correct cell.\n"
    "Reasoning <200 tokens. End with \\boxed{Cx|Cy|Cz}."
)


# Function name -> server.py URL path. Full set of tools exposed by server.py;
# matches Environment.endpoint_mapper in main.py. Meta endpoints (health,
# get_all_scenario, get_available_tools) are NOT exposed to the LLM as tools.
ENDPOINT_MAP = {
    "get_config_data":                "/config-data",
    "get_user_plane_data":            "/user-plane-data",
    "get_throughput_logs":            "/throughput-logs",
    "get_cell_info":                  "/cell-info",
    "get_gnodeb_location":            "/gnodeb-location",
    "get_user_location":              "/user-location",
    "get_serving_cell_pci":           "/serving-cell-pci",
    "get_serving_cell_rsrp":          "/serving-cell-rsrp",
    "get_serving_cell_sinr":          "/serving-cell-sinr",
    "get_rbs_allocated_to_user":      "/rbs-allocated-to-user",
    "get_neighboring_cells_pci":      "/neighboring-cells-pci",
    "get_neighboring_cell_rsrp":      "/neighboring-cell-rsrp",
    "get_signaling_plane_event_log":  "/signaling-plane-event-log",
    "get_all_cells_pci":              "/all-cells-pci",
    "get_kpi_data":                   "/get_kpi_data",
    "get_mr_data":                    "/get_mr_data",
    "judge_mainlobe_or_not":          "/judge_mainlobe",
    "calculate_horizontal_angle":     "/calculate_horizontal_angle",
    "calculate_tilt_angle":           "/calculate_tilt_angle",
    "calculate_pathloss":             "/calculate_pathloss",
    "calculate_overlap_ratio":        "/calculate_overlap_ratio",
    "optimize_antenna_gain":          "/optimize_antenna_gain",
}

_EXCLUDE_META = {"health", "get_all_scenario", "get_available_tools"}

# Tool descriptors lazy-loaded from server.py /tools on first use.
_TOOL_DEFS_CACHE: Optional[List[Dict[str, Any]]] = None


def _fetch_tool_defs(tool_url: str, timeout_s: float = 10.0) -> List[Dict[str, Any]]:
    """Discover the full tool catalog from server.py /tools. Cached for the run."""
    global _TOOL_DEFS_CACHE
    if _TOOL_DEFS_CACHE is not None:
        return _TOOL_DEFS_CACHE
    try:
        r = requests.get(f"{tool_url.rstrip('/')}/tools", timeout=timeout_s)
        r.raise_for_status()
        raw = r.json()
    except Exception as exc:
        print(f"  [tools] failed to fetch /tools: {exc} — using empty catalog", file=sys.stderr)
        _TOOL_DEFS_CACHE = []
        return _TOOL_DEFS_CACHE

    tools: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for t in raw:
            if not isinstance(t, dict):
                continue
            fn = t.get("function", t)
            name = fn.get("name") if isinstance(fn, dict) else None
            if not name or name in _EXCLUDE_META:
                continue
            # Normalize to OpenAI tools schema
            if t.get("type") == "function" and "function" in t:
                tools.append(t)
            else:
                tools.append({"type": "function", "function": fn})
    _TOOL_DEFS_CACHE = tools
    print(f"  [tools] loaded {len(tools)} tool descriptors from {tool_url}/tools",
          file=sys.stderr)
    return tools


# --------------------------------------------------------------------- helpers

_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_CX_RE = re.compile(r"\bC\d+\b")


def _truncate_scenario(scenario: Dict[str, Any], max_chars: int = 9000) -> str:
    """Render scenario data with priority order. Drop low-value tables if oversize."""
    d = scenario.get("data") or {}
    parts: List[str] = []

    def add(label: str, key: str) -> None:
        val = d.get(key)
        if val:
            parts.append(f"## {label}\n```\n{val.strip()}\n```")

    add("Network Configuration", "network_configuration_data")
    add("User-Plane Time Series", "user_plane_data")
    add("Signaling Plane Events", "signaling_plane_data")
    add("Cell-Level Traffic KPIs", "traffic_data")
    add("Measurement Reports", "mr_data")

    text = "\n\n".join(parts)
    if len(text) <= max_chars:
        return text
    # Drop mr_data first (least diagnostic), then traffic_data.
    for drop_section in ("Measurement Reports", "Cell-Level Traffic KPIs"):
        text = re.sub(
            rf"## {drop_section}\n```.*?```\n*", "", text, flags=re.DOTALL
        )
        if len(text) <= max_chars:
            return text
    # Last resort: hard truncate user_plane_data's middle
    return text[:max_chars]


def _extract_boxed(text: str, valid_options: List[str]) -> str:
    """Parse \\boxed{Cx} or \\boxed{Cx|Cy|Cz} from model output."""
    if not text:
        return ""
    matches = _BOXED_RE.findall(text)
    if not matches:
        # last-ditch: any Cx mention
        valid = set(valid_options)
        cx = [c for c in _CX_RE.findall(text) if c in valid]
        return cx[0] if cx else ""
    inner = re.sub(r"[{}\s]", "", matches[-1]).lstrip(":").rstrip("./")
    if not inner:
        return ""
    valid = set(valid_options)
    parts = [p for p in inner.split("|") if p in valid]
    if not parts:
        return ""
    parts = sorted(set(parts), key=lambda s: int(re.search(r"\d+", s).group()))
    return "|".join(parts)


def _format_question(scenario: Dict[str, Any]) -> str:
    options = (scenario.get("task") or {}).get("options", []) or []
    options_block = "\n".join(f"  {o['id']}: {o['label']}" for o in options if "id" in o)
    task_desc = (scenario.get("task") or {}).get("description") or ""
    data_block = _truncate_scenario(scenario)
    return (
        f"{data_block}\n\n"
        f"## Task\n{task_desc}\n\n"
        f"## Options\n{options_block}\n\n"
        f"Final answer:"
    )


def _execute_tool(
    name: str,
    arguments: Dict[str, Any],
    scenario_id: str,
    tool_url: str,
    timeout_s: float = 5.0,
) -> str:
    """Call server.py tool endpoint. Returns JSON string (or error JSON)."""
    endpoint = ENDPOINT_MAP.get(name)
    if endpoint is None:
        # Last-ditch: try the tool name itself as a path (e.g. /tool_name).
        endpoint = "/" + name.replace("_", "-")
    try:
        r = requests.get(
            f"{tool_url.rstrip('/')}{endpoint}",
            params=arguments,
            headers={"X-Scenario-Id": scenario_id} if scenario_id else {},
            timeout=timeout_s,
        )
        if r.status_code != 200:
            return json.dumps({"error": f"status {r.status_code}", "detail": r.text[:200]})
        # Truncate huge tool results so they don't blow the next-turn prompt
        text = r.text or ""
        if len(text) > 1500:
            text = text[:1500] + " ...[truncated]"
        return text
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _call_llm(
    messages: List[Dict[str, Any]],
    llm_url: str,
    model_name: str,
    timeout_s: float,
    max_tokens: int,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Single chat completion call. Returns assistant message dict or None.
    Pass tools=[] (or None) to disable tool-use for this turn."""
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    try:
        r = requests.post(
            f"{llm_url.rstrip('/')}/v1/chat/completions",
            json=payload,
            timeout=timeout_s,
            headers={"Authorization": f"Bearer {os.environ.get('AGENT_API_KEY', 'sk-dummy')}"},
        )
        if r.status_code != 200:
            print(f"  [llm-{r.status_code}] {r.text[:200]}", file=sys.stderr)
            return None
        return r.json().get("choices", [{}])[0].get("message") or {}
    except Exception as exc:
        print(f"  [llm-error] {exc}", file=sys.stderr)
        return None


def _agent_turn(
    scenario: Dict[str, Any],
    llm_url: str,
    tool_url: str,
    model_name: str,
    timeout_s: float,
    max_tokens: int,
    max_tool_calls: int = 2,
) -> Dict[str, Any]:
    """
    Agentic loop for one scenario. Up to `max_tool_calls` tool calls + 1
    final answer. All server.py tools are available; the LLM picks which.
    Returns dict with text, tool_calls_made, num_tool_calls.
    """
    sid = scenario.get("scenario_id", "")
    is_multi = task_is_multi(scenario)
    system = SYSTEM_PROMPT_MULTI if is_multi else SYSTEM_PROMPT_SINGLE
    question = _format_question(scenario)
    tool_defs = _fetch_tool_defs(tool_url)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    tool_calls_made: List[Dict[str, Any]] = []

    for turn in range(max_tool_calls):
        msg = _call_llm(messages, llm_url, model_name, timeout_s, max_tokens, tools=tool_defs)
        if msg is None:
            return {"text": "", "tool_calls_made": tool_calls_made, "num_tool_calls": len(tool_calls_made)}

        tcs = msg.get("tool_calls") or []
        text = msg.get("content") or ""

        # If model produced a boxed answer with no tool calls, we're done
        if not tcs and ("\\boxed{" in text or _BOXED_RE.search(text)):
            return {"text": text, "tool_calls_made": tool_calls_made, "num_tool_calls": len(tool_calls_made)}

        if not tcs:
            # No tool, no boxed — break to the explicit final-answer turn
            messages.append({"role": "assistant", "content": text})
            break

        # Execute the FIRST tool call (one per turn)
        tc = tcs[0]
        fn = (tc.get("function") or {}).get("name", "")
        args_str = (tc.get("function") or {}).get("arguments", "{}")
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
        except json.JSONDecodeError:
            args = {}
        result_str = _execute_tool(fn, args, sid, tool_url)
        tool_calls_made.append({
            "name": fn, "arguments": args, "result_head": result_str[:200],
        })
        messages.append({"role": "assistant", "content": text, "tool_calls": [tc]})
        messages.append({
            "role": "tool",
            "tool_call_id": tc.get("id", f"call_{turn}"),
            "content": result_str,
        })

    # Final-answer turn (no tools, tighter instruction)
    messages.append({
        "role": "user",
        "content": (
            "Based on the scenario data"
            + (" and tool results" if tool_calls_made else "")
            + ", output ONLY the final answer on the LAST line in this exact format: "
            + ("\\boxed{Cx|Cy|Cz} with 2-4 options in ascending order"
               if is_multi else "\\boxed{Cx}")
            + ". Keep any reasoning under 80 tokens."
        ),
    })
    final = _call_llm(messages, llm_url, model_name, timeout_s, max_tokens, tools=None)
    if final is None:
        return {"text": "", "tool_calls_made": tool_calls_made, "num_tool_calls": len(tool_calls_made)}
    return {"text": final.get("content") or "",
            "tool_calls_made": tool_calls_made,
            "num_tool_calls": len(tool_calls_made)}


# --------------------------------------------------------------------- runner

def _load_completions(path: Path) -> Dict[str, str]:
    done: Dict[str, str] = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                sid = rec.get("scenario_id")
                if sid:
                    done[sid] = rec.get("answer", "")
            except json.JSONDecodeError:
                continue
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--llm_url", default=os.environ.get("LLM_URL", "http://localhost:8001"))
    ap.add_argument("--tool_url", default=os.environ.get("TOOL_URL", "http://localhost:7860"))
    ap.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-35B-A3B"))
    ap.add_argument("--max_tokens", type=int, default=384)
    ap.add_argument("--llm_timeout_s", type=float, default=45.0)
    ap.add_argument("--scenario_timeout_s", type=float, default=120.0)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--max_tool_calls", type=int, default=2,
                    help="Max number of tool-call turns per scenario before forcing final answer.")
    args = ap.parse_args()

    test_path = (PROJECT_DIR / args.test_file).resolve()
    out_dir = (PROJECT_DIR / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    completions_path = out_dir / "completions.jsonl"
    csv_path = out_dir / "result.csv"

    if not test_path.exists():
        print(f"FATAL: {test_path} not found", file=sys.stderr)
        return 1

    # Health checks
    llm_ok = False
    try:
        h = requests.get(f"{args.llm_url}/health", timeout=5).json()
        llm_ok = h.get("status") == "ok"
        print(f"[llm] {args.llm_url} -> {h}")
    except Exception as exc:
        print(f"[llm] not reachable: {exc}", file=sys.stderr)
    tool_ok = False
    try:
        r = requests.get(f"{args.tool_url}/health", timeout=5)
        tool_ok = r.status_code == 200
        print(f"[tool] {args.tool_url} -> {r.status_code}")
    except Exception as exc:
        print(f"[tool] not reachable: {exc}", file=sys.stderr)
    if not llm_ok:
        print("[run] LLM unreachable — all answers will be heuristic fallback", file=sys.stderr)
    if not tool_ok:
        print("[run] tool server unreachable — agent will run without tool execution", file=sys.stderr)

    with test_path.open("r", encoding="utf-8") as f:
        scenarios = json.load(f)
    if args.max_samples is not None:
        scenarios = scenarios[: args.max_samples]

    done = _load_completions(completions_path)
    rows: List[Dict[str, str]] = [
        {"scenario_id": sid, "answers": ans} for sid, ans in done.items()
    ]
    print(f"[run] {len(scenarios)} scenarios, {len(done)} already done, "
          f"{len(scenarios) - len(done)} remaining")

    n_llm = n_fb = n_done = n_with_tool = 0
    f_jsonl = completions_path.open("a", encoding="utf-8")
    t_start = time.time()
    try:
        for i, scenario in enumerate(scenarios):
            sid = scenario.get("scenario_id", "")
            if sid in done:
                continue
            options = (scenario.get("task") or {}).get("options", []) or []
            valid_ids = [o["id"] for o in options if "id" in o]

            t0 = time.time()
            llm_text = ""
            tool_calls_made: List[Dict[str, Any]] = []
            num_tool_calls = 0
            answer = ""

            if llm_ok:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(
                        _agent_turn, scenario, args.llm_url, args.tool_url,
                        args.model_name, args.llm_timeout_s, args.max_tokens,
                        args.max_tool_calls,
                    )
                    try:
                        res = fut.result(timeout=args.scenario_timeout_s)
                        llm_text = res.get("text", "") or ""
                        tool_calls_made = res.get("tool_calls_made", [])
                        num_tool_calls = res.get("num_tool_calls", 0)
                        if num_tool_calls > 0:
                            n_with_tool += 1
                        answer = _extract_boxed(llm_text, valid_ids)
                    except concurrent.futures.TimeoutError:
                        print(f"  [timeout] scenario {sid[:8]}", file=sys.stderr)
                        fut.cancel()

            source = "llm"
            if not answer:
                answer = heuristic_pick(scenario)
                source = "heuristic"
                n_fb += 1
            else:
                n_llm += 1

            elapsed = time.time() - t0
            rec = {
                "scenario_id": sid,
                "answer": answer,
                "source": source,
                "num_tool_calls": num_tool_calls,
                "tool_calls": tool_calls_made,
                "elapsed_s": round(elapsed, 2),
                "llm_text_head": (llm_text or "")[:200],
            }
            f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_jsonl.flush()
            rows.append({"scenario_id": sid, "answers": answer})
            n_done += 1

            if (i + 1) % 10 == 0 or (i + 1) == len(scenarios):
                _write_csv(rows, csv_path)
            running = time.time() - t_start
            avg = running / max(n_done, 1)
            eta = avg * (len(scenarios) - len(done) - n_done)
            print(
                f"[{i+1:4d}/{len(scenarios)}] {sid[:8]} ans={answer:<18s} "
                f"src={source:9s} tool={num_tool_calls} {elapsed:5.1f}s  "
                f"(llm={n_llm} fb={n_fb} tool_used={n_with_tool}, eta={eta/60:.1f}min)"
            )
    finally:
        f_jsonl.close()
        _write_csv(rows, csv_path)

    # Three Zindi-ready variants (identical for now; later we can diverge multi-recall)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "result_v1_raw.csv", index=False)
    df.to_csv(out_dir / "result_v2_multi_recall.csv", index=False)
    df.to_csv(out_dir / "result_v3_insurance.csv", index=False)

    # Auto-score if labeled
    labeled = [s for s in scenarios if s.get("answer") and s.get("answer") != "To be determined"]
    if labeled:
        ans_by_id = {r["scenario_id"]: r["answers"] for r in rows}
        cx_re = re.compile(r"^C\d+(\|C\d+)*$")
        n_scored = 0
        score_total = 0.0
        n_multi_correct = n_multi_total = n_single_correct = n_single_total = 0
        n_empty = n_malformed = 0
        for s in labeled:
            sid = s.get("scenario_id")
            pred = ans_by_id.get(sid, "")
            gt = s.get("answer", "")
            if not pred: n_empty += 1
            elif not cx_re.match(pred): n_malformed += 1
            if "|" in gt:
                n_multi_total += 1
                p, g = set(pred.split("|")) if pred else set(), set(gt.split("|"))
                iou = len(p & g) / max(len(p | g), 1)
                score_total += iou
                if iou == 1.0: n_multi_correct += 1
            else:
                n_single_total += 1
                ok = pred == gt
                score_total += 1.0 if ok else 0.0
                if ok: n_single_correct += 1
            n_scored += 1
        print()
        print(f"=== AGENTIC HOLDOUT SCORE ===")
        print(f"  scored {n_scored}")
        print(f"  mean   : {score_total/max(n_scored,1):.4f}")
        print(f"  single : {n_single_correct}/{n_single_total} exact "
              f"({100*n_single_correct/max(n_single_total,1):.1f}%)")
        print(f"  multi  : {n_multi_correct}/{n_multi_total} full IoU=1.0 "
              f"({100*n_multi_correct/max(n_multi_total,1):.1f}%)")
        print(f"  empty/malformed: {n_empty}/{n_malformed}")
        print(f"  tool used      : {n_with_tool}")
        print(f"  heuristic fb   : {n_fb}")

    print()
    print(f"=== DONE ===")
    print(f"  total       : {len(rows)}")
    print(f"  llm answers : {n_llm}")
    print(f"  heuristic fb: {n_fb}")
    print(f"  with tool   : {n_with_tool}")
    print(f"  csv         : {csv_path}")
    return 0


def _write_csv(rows: List[Dict[str, str]], path: Path) -> None:
    import pandas as pd
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    sys.exit(main())
