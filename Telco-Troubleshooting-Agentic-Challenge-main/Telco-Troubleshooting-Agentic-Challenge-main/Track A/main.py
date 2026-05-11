#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Telco Troubleshooting Agent (Track A) — Stage B refactor.

Changes vs baseline:
  - Inlines the scenario data into the first user message instead of forcing
    the model to fetch each slice via tool calls.
  - Restricts the tool set to three high-leverage compute tools.
  - Loads the diagnostic system prompt from prompts/system_prompt.md.
  - max_iterations defaults to 2; per-scenario hard timeout 300 s.
  - Majority-vote bug fixed (was [0]; now Counter.most_common).
  - Caches /tools and /scenario/all so they hit the server once per run.
  - --quiet default suppresses tool/response printing for production runs.
  - Persists per-scenario completions.jsonl with accuracy + latency for
    downstream scoring. Resumable: a second run skips scenarios already
    present in completions.jsonl.
  - Multi-answer answers are normalized to ascending order.

Locked files (NOT modified): server.py, _types.py, utils.py, requirements.txt,
data/Phase_1/*.json. main.py is owned by us.
"""

import argparse
import concurrent.futures
import json
import logging
import os
import random
import re
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd
import requests
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, APIError

from _types import ToolCall
from logger import init_logger
from utils import (
    print_model_response,
    print_tool_call,
    print_tool_result,
    extract_answer,
    extract_answer_all,
    compute_score,
)

os.environ.setdefault("AGENT_API_KEY", "sk-XXXXXXXXXXXXX")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
API_KEY = os.environ.get("AGENT_API_KEY", "dummy")


def set_seeds(seed: int = 42) -> None:
    """Required for Zindi code-review reproducibility (rules: 'Always set the seed')."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


set_seeds(int(os.environ.get("SEED", "42")))

ALLOWED_TOOLS = {
    "judge_mainlobe_or_not",
    "calculate_overlap_ratio",
    "calculate_pathloss",
}

PER_SCENARIO_TIMEOUT_S = 300
INSUFFICIENT_DATA_LABEL_PREFIX = "Insufficient data"


# ------------------------------------------------------------------------------
# Scenario formatting
# ------------------------------------------------------------------------------

def format_scenario_for_prompt(scenario: Dict[str, Any]) -> str:
    """Render scenario data inline as a single Markdown block.

    Empty optional sections are omitted so we don't waste tokens on placeholders.
    """
    d = scenario.get("data", {}) or {}
    parts: List[str] = []

    cfg = d.get("network_configuration_data")
    if cfg:
        parts.append("## Network Configuration\n```\n" + cfg.strip() + "\n```")

    up = d.get("user_plane_data")
    if up:
        parts.append("## User-Plane Time Series\n```\n" + up.strip() + "\n```")

    sig = d.get("signaling_plane_data")
    if sig:
        parts.append("## Signaling Plane Events\n```\n" + sig.strip() + "\n```")

    traffic = d.get("traffic_data")
    if traffic:
        parts.append("## Cell-Level Traffic KPIs\n```\n" + traffic.strip() + "\n```")

    mr = d.get("mr_data")
    if mr:
        parts.append("## Measurement Reports Sample\n```\n" + mr.strip() + "\n```")

    return "\n\n".join(parts)


def normalize_multi_answer(answer: str) -> str:
    """Sort multi-answer in ascending Cx order, pipe-separated, no spaces."""
    if not answer:
        return ""
    if "|" not in answer:
        return answer.strip()
    items = [a.strip() for a in answer.split("|") if a.strip()]
    def _key(s: str) -> int:
        m = re.search(r"\d+", s)
        return int(m.group()) if m else 0
    items = sorted(set(items), key=_key)
    return "|".join(items)


def find_insufficient_data_option(options: List[Dict[str, str]]) -> Optional[str]:
    """Return the Cx id of the 'Insufficient data' option if present."""
    for opt in options or []:
        label = (opt.get("label") or "").strip().lower()
        if label.startswith(INSUFFICIENT_DATA_LABEL_PREFIX.lower()):
            return opt.get("id")
    return None


def load_system_prompt() -> str:
    here = Path(__file__).parent
    sp_path = here / "prompts" / "system_prompt.md"
    if sp_path.exists():
        return sp_path.read_text(encoding="utf-8").strip()
    return (
        "You are a 5G RAN troubleshooting expert. Diagnose the throughput drop "
        "and answer with \\boxed{Cx} (single) or \\boxed{Cx|Cy|...} (multi, "
        "ascending). Use the inline scenario data; call tools sparingly."
    )


# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------

class Environment:
    """
    Discovers tool descriptors via /tools and executes tool calls against the
    locked server. Per-scenario context is conveyed via the X-Scenario-Id header.
    """

    endpoint_mapper = {
        "get_all_scenario": "/scenario/all",
        "get_config_data": "/config-data",
        "get_user_plane_data": "/user-plane-data",
        "get_throughput_logs": "/throughput-logs",
        "get_cell_info": "/cell-info",
        "get_gnodeb_location": "/gnodeb-location",
        "get_user_location": "/user-location",
        "get_serving_cell_pci": "/serving-cell-pci",
        "get_serving_cell_rsrp": "/serving-cell-rsrp",
        "get_serving_cell_sinr": "/serving-cell-sinr",
        "get_rbs_allocated_to_user": "/rbs-allocated-to-user",
        "get_neighboring_cells_pci": "/neighboring-cells-pci",
        "get_neighboring_cell_rsrp": "/neighboring-cell-rsrp",
        "get_signaling_plane_event_log": "/signaling-plane-event-log",
        "get_all_cells_pci": "/all-cells-pci",
        "get_available_tools": "/tools",
        "health": "/health",
        "judge_mainlobe_or_not": "/judge_mainlobe",
        "calculate_horizontal_angle": "/calculate_horizontal_angle",
        "calculate_tilt_angle": "/calculate_tilt_angle",
        "calculate_pathloss": "/calculate_pathloss",
        "calculate_overlap_ratio": "/calculate_overlap_ratio",
        "get_kpi_data": "/get_kpi_data",
        "get_mr_data": "/get_mr_data",
        "optimize_antenna_gain": "/optimize_antenna_gain",
    }

    def __init__(
        self,
        server_url: str,
        verbose: bool = False,
        timeout: float = 15.0,
        logger: logging.Logger = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.verbose = verbose
        self.timeout = timeout
        self.logger = logger if logger is not None else init_logger()
        # Caches: /tools and /scenario/all are static across the run.
        self._tools_cache: Optional[List[Dict[str, Any]]] = None
        self._scenarios_cache: Optional[List[Dict[str, Any]]] = None

    def _headers(self, scenario_id: Optional[str] = None) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if scenario_id:
            headers["X-Scenario-Id"] = scenario_id
        return headers

    def _call_api(
        self,
        function_name: str,
        scenario_id: Optional[str] = None,
        **params: Any,
    ) -> Dict[str, Any]:
        endpoint = self.endpoint_mapper.get(function_name)
        if endpoint is None:
            return {"error": f"Unknown tool '{function_name}'"}
        url = f"{self.server_url}{endpoint}"
        headers = self._headers(scenario_id=scenario_id)
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            if self.verbose:
                self.logger.info(f"[Tools API] GET {endpoint} params={params}")
            return resp.json()
        except requests.exceptions.HTTPError:
            try:
                detail = resp.json().get("detail", str(resp.text))
            except Exception:
                detail = str(resp.text)
            if self.verbose:
                self.logger.info(f"[Tools API] GET {endpoint} -> HTTPError: {detail}")
            return {"error": detail}
        except Exception as e:
            if self.verbose:
                self.logger.info(f"[Tools API] GET {endpoint} -> ERROR: {e}")
            return {"error": str(e)}

    def get_tools(self) -> List[Dict[str, Any]]:
        if self._tools_cache is not None:
            return self._tools_cache
        tools = self._call_api("get_available_tools")
        if isinstance(tools, dict) and "error" in tools:
            self._tools_cache = []
        elif isinstance(tools, list):
            self._tools_cache = tools
        else:
            self._tools_cache = []
        return self._tools_cache

    def get_scenarios(self) -> List[Dict[str, Any]]:
        if self._scenarios_cache is not None:
            return self._scenarios_cache
        scenarios = self._call_api("get_all_scenario")
        if isinstance(scenarios, dict) and "error" in scenarios:
            self._scenarios_cache = []
        elif isinstance(scenarios, list):
            self._scenarios_cache = scenarios
        else:
            self._scenarios_cache = []
        return self._scenarios_cache

    def execute(self, tool_call: ToolCall, scenario_id: Optional[str] = None) -> str:
        try:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments or "{}")
            result = self._call_api(function_name=function_name, scenario_id=scenario_id, **arguments)
            return json.dumps(result, ensure_ascii=False)
        except json.JSONDecodeError:
            err = f"Tool parameter parsing failed: {tool_call.function.arguments}"
            if self.verbose:
                self.logger.error(err, exc_info=True)
            return json.dumps({"error": err}, ensure_ascii=False)
        except Exception as e:
            err = f"Tool invocation execution failed: {str(e)}"
            if self.verbose:
                self.logger.error(err, exc_info=True)
            return json.dumps({"error": err}, ensure_ascii=False)


# ------------------------------------------------------------------------------
# Agent runner
# ------------------------------------------------------------------------------

class AgentsRunner:
    def __init__(
        self,
        environment: Environment,
        model_url: str,
        model_name: str,
        model_provider: Optional[str] = None,
        max_tokens: int = 1024,
        max_retries: int = 3,
        max_iterations: int = 2,
        verbose: bool = False,
        logger: logging.Logger = None,
        system_prompt: Optional[str] = None,
    ):
        self.environment = environment
        self.model_url = model_url
        self.model_name = model_name
        self.model_provider = model_provider
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.logger = logger if logger is not None else init_logger()
        self.system_prompt = system_prompt or load_system_prompt()
        self.client = OpenAI(
            base_url=model_url,
            api_key=API_KEY,
            http_client=httpx.Client(verify=False),
        )

    # ------------------------------------------------------------------ model

    def _call_model(self, messages: List[Dict[str, Any]], functions: List[Dict[str, Any]], **kwargs):
        base_wait_time = 1.0
        call_kwargs = {
            "model": f"{self.model_provider}/{self.model_name}" if self.model_provider else self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            **kwargs,
        }
        if functions:
            call_kwargs["tools"] = functions
            call_kwargs["tool_choice"] = "auto"

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**call_kwargs)
                return response.choices[0].message
            except (RateLimitError, APIConnectionError, APITimeoutError, APIError) as exc:
                if self.verbose:
                    self.logger.error(traceback.format_exc())
                if hasattr(exc, "status_code") and 400 <= exc.status_code < 500 and exc.status_code != 429:
                    if self.verbose:
                        self.logger.info("Non-retriable: %s", exc)
                    return None
                if attempt == self.max_retries:
                    if self.verbose:
                        self.logger.info("Final failure after %d attempts: %s", self.max_retries, exc)
                    return None
                wait = base_wait_time * (2 ** (attempt - 1))
                if self.verbose:
                    self.logger.info("Retry %d/%d after %.1fs: %s", attempt, self.max_retries, wait, exc)
                time.sleep(wait)
            except Exception as exc:
                if self.verbose:
                    self.logger.info("Unhandled: %s", exc)
                return None
        return None

    # --------------------------------------------------------------- run loop

    def _run_inner(self, scenario: Dict[str, Any], free_mode: bool) -> Dict[str, Any]:
        """The core agent loop. Wrapped by run() with a timeout."""
        scenario_id = scenario.get("scenario_id")
        task = scenario.get("task", {}) or {}
        options = task.get("options", []) or []
        root_causes = "".join([f"{item['id']}:{item['label']}\n" for item in options])

        all_tool_defs = self.environment.get_tools()
        if not all_tool_defs:
            return {"scenario_id": scenario_id, "status": "unresolved", "reason": "No tools available"}

        # Restrict to the three derived-signal tools.
        tool_defs = [
            t for t in all_tool_defs
            if (t.get("function", {}).get("name") in ALLOWED_TOOLS)
        ]
        if self.verbose:
            self.logger.info(
                f"[Tools] {len(all_tool_defs)} available, "
                f"{len(tool_defs)} allowed: "
                f"{[t['function']['name'] for t in tool_defs]}"
            )

        # Inline the scenario data into the first user message.
        scenario_block = format_scenario_for_prompt(scenario)
        question = (
            f"{scenario_block}\n\n"
            f"## Task\n{task.get('description', '')}\n\n"
            f"## Options\n{root_causes}"
        )
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

        num_tool_calls = 0
        list_tool_calls: List[Dict[str, Any]] = []
        status: Optional[str] = None
        reason: Optional[str] = None
        last_msg = None
        i = 0

        for i in range(self.max_iterations):
            if self.verbose:
                self.logger.info(f"\n[Scenario: {scenario_id}] Round {i + 1}")

            msg = self._call_model(messages, functions=tool_defs)
            if msg is None:
                continue
            last_msg = msg
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})
            if self.verbose:
                print_model_response(msg, logger=self.logger, minimize=False)

            if msg.tool_calls:
                num_tool_calls += len(msg.tool_calls)
                for j, tool_call in enumerate(msg.tool_calls):
                    if self.verbose:
                        print_tool_call(tool_call, logger=self.logger)
                    tool_result = self.environment.execute(tool_call, scenario_id=scenario_id)
                    messages.append({"role": "tool", "content": tool_result, "tool_call_id": tool_call.id})
                    if self.verbose:
                        print_tool_result(tool_result, logger=self.logger)
                    list_tool_calls.append({
                        "function_name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                        "turn": i + 1,
                        "has_failed": "error" in tool_result,
                        "order": j + 1,
                        "results": tool_result,
                    })
            elif msg.content:
                status = "solved"
                break
            else:
                status = "unresolved"
                reason = "Empty response and no tool calls."
                break

        if status is None:
            status = "unresolved"
            reason = "Reached max_iterations without a final answer."

        # Final-answer constraint pass (free_mode): if the last message has no
        # \boxed{...}, force one more turn with the explicit option list.
        # Also: if the task is multi-answer but the model emitted a single
        # answer, force expansion (recall bias — better IoU).
        if free_mode:
            current_answer = (getattr(last_msg, "content", "") or "") if last_msg else ""
            current_traces = (getattr(last_msg, "reasoning_content", "") or "") if last_msg else ""
            agent_answer = extract_answer(current_answer) or extract_answer(current_traces)
            is_multi_task = "Select the most appropriate" not in (task.get("description") or "")
            single_for_multi = (
                is_multi_task
                and agent_answer
                and "|" not in agent_answer
                and os.environ.get("ENFORCE_MULTI", "1") == "1"
            )
            if agent_answer == "" or single_for_multi:
                if self.verbose:
                    self.logger.info(f"[Scenario: {scenario_id}] Forcing final-answer turn")
                if "Select the most appropriate" in (task.get("description") or ""):
                    messages.append({
                        "role": "user",
                        "content": (
                            "This is a SINGLE-answer question. Select the most appropriate "
                            "optimization solution and enclose its number in \\boxed{} "
                            f"in the final answer. For example, \\boxed{{C3}}\nOptions:\n{root_causes}"
                        ),
                    })
                else:
                    # Multi-answer recall bias: scoring is IoU, so predicting
                    # 2-3 likely options beats predicting 1 confident one. Tell
                    # the model explicitly. If single_for_multi, mention the
                    # current answer and ask for additions, not a replacement.
                    extra = ""
                    if single_for_multi and agent_answer:
                        extra = (
                            f"\nYour previous answer ({agent_answer}) was a single option, "
                            "but this task requires 2 to 4. Keep that option AND add the "
                            "next most plausible options from your analysis."
                        )
                    messages.append({
                        "role": "user",
                        "content": (
                            "This is a MULTIPLE-answer question. Select 2 to 4 optimization "
                            "solutions and enclose their numbers in \\boxed{} in ascending "
                            "order separated by | in the final answer. Scoring is "
                            "intersection-over-union, so missing a correct option costs you "
                            "as much as adding a wrong one — pick the 2-3 most likely "
                            "options, not just the single most confident."
                            f"{extra}\n"
                            f"For example, \\boxed{{C3|C5}}\nOptions:\n{root_causes}"
                        ),
                    })
                msg2 = self._call_model(messages, functions=[])
                if msg2 is not None:
                    last_msg = msg2
                    status = "solved"

        return {
            "scenario_id": scenario_id,
            "num_iterations": i + 1,
            "tool_calls": list_tool_calls,
            "num_tool_calls": num_tool_calls,
            "status": status,
            "traces": (getattr(last_msg, "reasoning_content", "") or "") if last_msg else "",
            "answer": (getattr(last_msg, "content", "") or "") if last_msg else "",
            "messages": messages,
            "reason": reason,
        }

    def _timeout_fallback(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Return a structured fallback when a scenario blows the per-scenario budget."""
        options = (scenario.get("task", {}) or {}).get("options", []) or []
        cx = find_insufficient_data_option(options) or (options[0]["id"] if options else "C1")
        if self.verbose:
            self.logger.info(
                f"[Scenario: {scenario.get('scenario_id')}] timeout fallback -> {cx}"
            )
        return {
            "scenario_id": scenario.get("scenario_id"),
            "num_iterations": 0,
            "tool_calls": [],
            "num_tool_calls": 0,
            "status": "timeout",
            "traces": "",
            "answer": f"\\boxed{{{cx}}}",
            "messages": [],
            "reason": f"per-scenario timeout {PER_SCENARIO_TIMEOUT_S}s exceeded",
        }

    def run(self, scenario: Dict[str, Any], free_mode: bool = False) -> Dict[str, Any]:
        """Run a single scenario with a hard wall-clock cap."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(self._run_inner, scenario, free_mode)
            try:
                return future.result(timeout=PER_SCENARIO_TIMEOUT_S)
            except concurrent.futures.TimeoutError:
                future.cancel()
                return self._timeout_fallback(scenario)
            except Exception as e:
                if self.verbose:
                    self.logger.error(f"[Scenario {scenario.get('scenario_id')}] inner error: {e}", exc_info=True)
                return self._timeout_fallback(scenario)

    # ------------------------------------------------------------- benchmark

    def benchmark(
        self,
        num_attempts: int,
        save_dir: str,
        save_freq: int = 10,
        max_samples: Optional[int] = None,
        free_mode: bool = False,
    ) -> Dict[str, Any]:
        os.makedirs(save_dir, exist_ok=True)
        completions_path = os.path.join(save_dir, "completions.jsonl")
        result_csv_path = os.path.join(save_dir, "result.csv")

        # Resume support: skip scenario_ids already in completions.jsonl.
        already_done: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(completions_path):
            with open(completions_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if rec.get("scenario_id"):
                            already_done[rec["scenario_id"]] = rec
                    except json.JSONDecodeError:
                        continue
            if already_done:
                self.logger.info(f"[Resume] {len(already_done)} scenarios already in {completions_path}")

        scenarios = self.environment.get_scenarios()
        if max_samples is not None:
            scenarios = scenarios[: min(max_samples, len(scenarios))]

        completions: List[Dict[str, Any]] = list(already_done.values())
        save_result: List[Dict[str, Any]] = [
            {"scenario_id": rec["scenario_id"], "answers": rec.get("final_answer", "")}
            for rec in already_done.values()
        ]

        f_jsonl = open(completions_path, "a", encoding="utf-8")
        try:
            for idx, scenario in enumerate(scenarios):
                scenario_id = scenario.get("scenario_id")
                if scenario_id in already_done:
                    continue

                start_time = time.time()
                n_success = 0.0
                agent_answers: List[str] = []
                sample_response: Dict[str, Any] = {}

                for attempt in range(num_attempts):
                    if self.verbose:
                        self.logger.info(f"[Scenario {scenario_id}] attempt {attempt + 1}/{num_attempts}")
                    response = self.run(scenario=scenario, free_mode=free_mode)
                    sample_response = response

                    raw_extracted = extract_answer_all(response.get("answer", "")) or extract_answer_all(response.get("traces", ""))
                    agent_answer = normalize_multi_answer(raw_extracted)
                    ground_truth = scenario.get("answer", "")
                    if agent_answer:
                        try:
                            n_success += float(compute_score(agent_answer, ground_truth))
                        except Exception:
                            pass
                    agent_answers.append(agent_answer)

                    pink, reset = "\033[95m", "\033[0m"
                    self.logger.info(
                        f"{pink}[Scenario: {scenario_id}] answer={agent_answer} gt={ground_truth}{reset}"
                    )

                # Majority vote across attempts (fixed bug: original used [0]).
                if agent_answers:
                    counted = Counter(a for a in agent_answers if a)
                    final_answer = counted.most_common(1)[0][0] if counted else (agent_answers[0] or "")
                else:
                    final_answer = ""

                acc = n_success / float(max(num_attempts, 1))
                latency = round((time.time() - start_time) / float(max(num_attempts, 1)), 2)

                rec = {
                    "scenario_id": scenario_id,
                    "free_mode": free_mode,
                    "final_answer": final_answer,
                    "all_attempts": agent_answers,
                    "ground_truth": scenario.get("answer", ""),
                    "accuracy": acc,
                    "latency": latency,
                    "status": sample_response.get("status"),
                    "num_iterations": sample_response.get("num_iterations", 0),
                    "num_tool_calls": sample_response.get("num_tool_calls", 0),
                    "tool_calls": sample_response.get("tool_calls", []),
                    "answer_raw": sample_response.get("answer", ""),
                    "traces": sample_response.get("traces", ""),
                    "reason": sample_response.get("reason"),
                }
                completions.append(rec)
                save_result.append({"scenario_id": scenario_id, "answers": final_answer})

                f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_jsonl.flush()

                if ((idx + 1) % save_freq == 0) or ((idx + 1) == len(scenarios)):
                    pd.DataFrame(save_result).to_csv(result_csv_path, index=False)
        finally:
            f_jsonl.close()

        pd.DataFrame(save_result).to_csv(result_csv_path, index=False)

        # Aggregate metrics.
        latencies = [c.get("latency", 0.0) for c in completions if c.get("latency") is not None]
        accs = [c.get("accuracy", 0.0) for c in completions if c.get("accuracy") is not None]
        timeouts = sum(1 for c in completions if c.get("status") == "timeout")
        unresolved = sum(1 for c in completions if c.get("status") == "unresolved")
        summary = {
            "n_scenarios": len(completions),
            "mean_accuracy": (sum(accs) / len(accs)) if accs else 0.0,
            "mean_latency_s": (sum(latencies) / len(latencies)) if latencies else 0.0,
            "max_latency_s": max(latencies) if latencies else 0.0,
            "timeouts": timeouts,
            "unresolved": unresolved,
            "mean_tool_calls": (sum(c.get("num_tool_calls", 0) for c in completions) / len(completions)) if completions else 0.0,
        }
        with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Telco Track A agent (Stage B refactor)")
    parser.add_argument("--server_url", type=str, default="http://localhost:7860")
    parser.add_argument("--model_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--model_provider", type=str, default=None)
    parser.add_argument("--num_attempts", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Max generation tokens per call. 1024 is enough for "
                             "<800 tok reasoning + boxed answer; higher values blow timeout on Turing.")
    parser.add_argument("--max_iterations", type=int, default=2)
    parser.add_argument("--save_dir", type=str, default="./eval/results/run")
    parser.add_argument("--log_file", type=str, default="./run.log")
    parser.add_argument("--free_mode", action="store_true", default=True,
                        help="Force a final-answer turn if no \\boxed{} after main loop")
    parser.add_argument("--no_free_mode", dest="free_mode", action="store_false")
    verb = parser.add_mutually_exclusive_group()
    verb.add_argument("--verbose", action="store_true", help="Verbose logging (off by default for speed)")
    verb.add_argument("--quiet",   action="store_true", help="Quiet (default)")
    args = parser.parse_args()

    verbose = bool(args.verbose) and not bool(args.quiet)
    logger = init_logger(log_file=args.log_file)

    env = Environment(server_url=args.server_url, verbose=verbose, logger=logger)
    runner = AgentsRunner(
        environment=env,
        model_url=args.model_url,
        model_name=args.model_name,
        model_provider=args.model_provider,
        max_tokens=args.max_tokens,
        max_iterations=args.max_iterations,
        verbose=verbose,
        logger=logger,
    )
    summary = runner.benchmark(
        max_samples=args.max_samples,
        num_attempts=args.num_attempts,
        save_dir=args.save_dir,
        save_freq=args.save_freq,
        free_mode=args.free_mode,
    )
    logger.info(
        "RUN_SUMMARY n=%d acc=%.3f latency_s_mean=%.1f timeouts=%d tool_calls_mean=%.2f",
        summary["n_scenarios"],
        summary["mean_accuracy"],
        summary["mean_latency_s"],
        summary["timeouts"],
        summary["mean_tool_calls"],
    )


if __name__ == "__main__":
    main()
