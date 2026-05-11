
from typing import Optional, Dict, Any, List
from _types import Scenario
from logger import logging
import json
import re
from datetime import datetime, timezone
from dateutil import parser
import os

import pandas as pd



def load_scenarios(config, default_scenario="random", split="test"):

    with open(f"./{config}/{split}.json", "r", encoding="utf-8") as f:
        raw_scenarios = json.load(f)

    if not raw_scenarios:
        raise RuntimeError(f"No scenarios loaded. Check ./{config}/test.json or DOWNLOAD_FROM_REMOTE.")

    validated_scenarios: List[Scenario] = [Scenario.model_validate(s) for s in raw_scenarios]

    if default_scenario == "random":
        import random
        default_scenario_id = random.choice(validated_scenarios).scenario_id
    else:
        default_scenario_id = validated_scenarios[0].scenario_id

    return validated_scenarios, default_scenario_id



def df_first_or_none(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if df is None or df.empty:
        return None
    return df.iloc[0].to_dict()


def df_all_or_none(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if df is None or df.empty:
        return None
    return df.to_csv(index=False, sep='|')


def normalize_time(ts: str) -> datetime:
    dt = parser.parse(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def get_fields_at_time(df: pd.DataFrame, user_input_time: str, fields: List[str]) -> pd.DataFrame:
    target_time = normalize_time(user_input_time)
    tmp = df.copy()
    tmp["NormalizedTime"] = tmp["Timestamp"].apply(normalize_time)
    return tmp.loc[tmp["NormalizedTime"] == target_time, fields]


def get_fields_before_time(df: pd.DataFrame, user_input_time: str, fields: List[str]) -> pd.DataFrame:
    target_time = normalize_time(user_input_time)
    tmp = df.copy()
    tmp["NormalizedTime"] = tmp["Timestamp"].apply(normalize_time)
    return tmp.loc[tmp["NormalizedTime"] <= target_time, fields]


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def print_tool_result(tool_result, logger: logging.Logger):
    color = "\033[95m"
    reset = "\033[0m"
    logger.info(
        f"{color}[Tool result] {tool_result} {reset}"
    )


def print_tool_call(tool_call, logger: logging.Logger, step=None):
    function_name = tool_call.function.name
    try:
        arguments = json.loads(tool_call.function.arguments)
        args = ", ".join([f"{k}={v}" for (k, v) in arguments.items()])
    except:
        args = str(tool_call.function.arguments)

    color = "\033[95m"
    reset = "\033[0m"
    step_msg = f"Step {step}: " if step else ""
    logger.info(
        f"{color}[{step_msg}Executing tool call {tool_call.id}] function {function_name}({args}){reset}"
    )


def print_model_response(response, logger: logging.Logger, minimize=False):
    color = "\x1b[31;1m"
    reset = "\033[0m"
    if minimize:
        logger.info(
            f"{color}[Agent response] {getattr(response, 'content', None)}{reset}\n"
        )
    else:
        if getattr(response, 'content', None):
            logger.info(
                f"{color}[Agent response] {response.content}{reset}\n"
            )
        if getattr(response, 'reasoning_content', None):
            logger.info(
                f"[Agent reasoning] {response.reasoning_content}\n"
            )

        if getattr(response, 'tool_calls', None):
            for tool_call in response.tool_calls:
                function_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    args = ", ".join([f"{k}={v}" for (k, v) in arguments.items()])
                except:
                    args = str(tool_call.function.arguments)

                logger.info(
                    f"[Request tool call {tool_call.id}] function {function_name}({args})"
                )


def extract_answer(response: str):
    try:
        matches = re.findall(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', response)
        if matches:
            pred = matches[-1].strip()
            pred = re.sub(r"\n\s*", "", pred).lstrip(":").rstrip("./")
            pred = re.sub(r"[{}]", "", pred)
            return pred
        return ""
    except:
        return ""

def extract_answer_all(response: str):
    try:
        matches = re.findall(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', response)
        if matches:
            pred = matches[-1].strip()
            pred = re.sub(r"[{}]", "", pred)
            return pred
        return ""
    except:
        return ""


def compute_score(gt: str, answer: str, decision="hard"):
    if '|' in gt:
        gts = gt.split("|")
        return any([compute_score(g, answer, decision) for g in gts])
    try:
        if decision == "soft":
            match = re.search(r'\d+', gt)
            gt_int = int(match.group()) if match else None
            match = re.search(r'\d+', answer)
            answer_int = int(match.group()) if match else None
            return gt_int == answer_int
        else:
            return gt.lower() == answer.lower()
    except:
        return False