# -*- coding: utf-8 -*-
"""
CTBench Batch Inference Script — Load questions from local JSON, invoke Agent, and output results to CSV
"""

import argparse
import csv
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ============================================================
# Configuration
# ============================================================

# Default input JSON file path
DEFAULT_INPUT_JSON = "questions.json"

# OPENCLAW_DIR = r"D:\PycharmProjects\openclaw"
OPENCLAW_DIR = r""
# OPENCLAW_SESSION_DIR = r"C:\Users\y00633222\.openclaw\agents\main\sessions"
OPENCLAW_SESSION_DIR = r""
FORCE_KILL_TIMEOUT = 600  # Force kill timeout per question (10 minutes)
DEFAULT_CONCURRENCY = 2

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results")
RESULT_CSV = os.path.join(RESULT_DIR, "result.csv")
DETAIL_LOG = os.path.join(RESULT_DIR, "eval_detail.jsonl")
PROGRESS_FILE = os.path.join(RESULT_DIR, "progress.json")

# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("inference")


# ============================================================
# Local JSON Data Loading
# ============================================================

def load_questions_from_file(filepath: str) -> dict:
    """Load scenarios from the specified JSON file."""
    if not os.path.exists(filepath):
        log.error(f"Input file not found: {filepath}")
        sys.exit(1)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = {}
        for item in data:
            task = item.get("task", {})
            question_id = task.get("id")
            question_text = task.get("question", "")

            if question_id is not None and question_text:
                questions[question_id] = question_text
            else:
                log.warning("Skipping invalid item in JSON. Missing task.id or task.question.")

        return questions
    except Exception as e:
        log.error(f"Failed to parse JSON file {filepath}: {e}")
        sys.exit(1)


# ============================================================
# Invoke openclaw Agent
# ============================================================

def _ensure_wrapper_script(wrapper_path: str):
    if os.path.exists(wrapper_path):
        return
    script_content = """\
const fs = require("fs");
const cp = require("child_process");
const msgFile = process.argv[2];
const sessionId = process.argv[3] || "";
const msg = fs.readFileSync(msgFile, "utf-8");
const args = [
    "scripts/run-node.mjs", "agent",
    "-m", msg,
    "--json"
];
if (sessionId) {
    args.push("--session-id", sessionId);
}
const r = cp.spawnSync("node", args, {
    encoding: "utf-8",
    stdio: ["ignore", "pipe", "pipe"]
});
if (r.stdout) process.stdout.write(r.stdout);
if (r.stderr) process.stderr.write(r.stderr);
process.exitCode = r.status || 0;
"""
    os.makedirs(os.path.dirname(wrapper_path), exist_ok=True)
    with open(wrapper_path, "w", encoding="utf-8") as f:
        f.write(script_content)


def _kill_proc_tree(pid: int):
    try:
        if sys.platform == "win32":
            subprocess.run(f"taskkill /F /T /PID {pid}", shell=True, capture_output=True, timeout=10)
        else:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
    except Exception:
        pass


def invoke_openclaw(message: str, question_id: int) -> dict:
    full_message = f"[CTBench Question ID: {question_id}]\n{message}"

    session_id = f"ctbench-q{question_id}-{uuid.uuid4().hex[:8]}"
    msg_file = os.path.join(RESULT_DIR, f"_msg_{question_id}.txt")

    with open(msg_file, "w", encoding="utf-8") as mf:
        mf.write(full_message)

    wrapper_script = os.path.join(RESULT_DIR, "_invoke_wrapper.js")
    _ensure_wrapper_script(wrapper_script)
    cmd = f'node "{wrapper_script}" "{msg_file}" {session_id}'
    kill_timeout = FORCE_KILL_TIMEOUT
    start_time = time.time()
    timed_out = False
    proc = None

    try:
        proc = subprocess.Popen(
            cmd, cwd=OPENCLAW_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            shell=True, encoding="utf-8", errors="replace",
        )
        try:
            stdout, stderr = proc.communicate(timeout=kill_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            log.warning(f"  Question [{question_id}] exceeded {kill_timeout}s, force killing process...")
            _kill_proc_tree(proc.pid)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()

        elapsed_s = time.time() - start_time
        elapsed_ms = int(elapsed_s * 1000)

        if timed_out:
            return {"success": False, "reply": "", "session_id": session_id, "duration_ms": elapsed_ms,
                    "duration_s": round(elapsed_s, 1), "timed_out": True, "error": "Force killed due to timeout"}

        stdout = stdout or ""
        lines = stdout.split("\n")
        json_start = next((i for i, line in enumerate(lines) if line.strip().startswith("{")), None)

        if json_start is None:
            return {"success": False, "reply": "", "session_id": session_id, "duration_ms": elapsed_ms,
                    "duration_s": round(elapsed_s, 1), "timed_out": False, "error": "Failed to parse JSON output"}

        json_text = "\n".join(lines[json_start:])
        data = json.loads(json_text)

        if data.get("status") != "ok":
            return {"success": False, "reply": "", "session_id": session_id, "duration_ms": elapsed_ms,
                    "duration_s": round(elapsed_s, 1), "timed_out": False,
                    "error": f"Non-ok status: {data.get('status')}"}

        payloads = data.get("result", {}).get("payloads", [])
        all_text = "".join([p["text"] + "\n" for p in payloads if p.get("text")])

        return {"success": True, "reply": all_text.strip(), "session_id": session_id, "duration_ms": elapsed_ms,
                "duration_s": round(elapsed_s, 1), "timed_out": False, "error": None}

    except Exception as e:
        elapsed_s = time.time() - start_time
        if proc and proc.poll() is None:
            _kill_proc_tree(proc.pid)
        return {"success": False, "reply": "", "session_id": session_id, "duration_ms": int(elapsed_s * 1000),
                "duration_s": round(elapsed_s, 1), "timed_out": False, "error": str(e)}


# ============================================================
# Answer Extraction
# ============================================================

def _extract_text_from_content(content) -> str:
    if isinstance(content, list):
        texts = []
        for item in content:
            if not isinstance(item, dict): continue
            item_type = item.get("type")
            if item_type == "text":
                t = item.get("text", "").strip()
                if t: texts.append(t)
            elif item_type == "thinking":
                sig = item.get("thinkingSignature", "")
                if not sig: continue
                try:
                    sig_obj = json.loads(sig) if isinstance(sig, str) else sig
                    for part in sig_obj.get("content", []):
                        if isinstance(part, dict):
                            t = part.get("text", "").strip()
                            if t: texts.append(t)
                except:
                    continue
        return "\n".join(texts)
    elif isinstance(content, str):
        return content.strip()
    return ""


def _read_last_assistant_text(session_id: str) -> str:
    session_file = os.path.join(OPENCLAW_SESSION_DIR, f"{session_id}.jsonl")
    if not os.path.exists(session_file):
        return ""
    last_text = ""
    with open(session_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
                msg = rec.get("message")
                if msg and msg.get("role") == "assistant":
                    text = _extract_text_from_content(msg.get("content", []))
                    if text: last_text = text
            except:
                continue
    return last_text


def extract_answer(session_id: str) -> str:
    """Extract structured answer. If regex doesn't match, return the complete last reply."""
    last_text = _read_last_assistant_text(session_id)
    if not last_text:
        return ""

    answer_patterns = [
        r'[\w-]+\([A-Za-z0-9/]+\)\s*->\s*[\w-]+\([A-Za-z0-9/]+\)',
        r'[\w-]+\s*->\s*[\w-]+(?:\s*->\s*[\w-]+)+',
        r'[\w-]+;[^;\n]+;[^;\n]+',
    ]

    for pattern in answer_patterns:
        lines = last_text.strip().splitlines()
        answer_lines = [line.strip() for line in lines if re.search(pattern, line.strip())]
        if answer_lines:
            return "\n".join(answer_lines)

    return last_text


# ============================================================
# Progress Management & Output Setup
# ============================================================

def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": []}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def init_csv():
    """Initialize CSV file and write header"""
    if not os.path.exists(RESULT_CSV):
        # Use utf-8-sig to prevent character encoding issues in Excel
        with open(RESULT_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "prediction"])


def append_to_csv(question_id: int, output_text: str):
    """Safely append a row of data to CSV"""
    with open(RESULT_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([question_id, output_text])


# ============================================================
# Main Execution Flow
# ============================================================

def _process_one_question(question_id: int, question_text: str) -> dict:
    result = invoke_openclaw(question_text, question_id)
    extracted = extract_answer(result.get("session_id", "")) if result["success"] else f"Error: {result.get('error')}"

    return {
        "question_id": question_id,
        "question_text": question_text,
        "result": result,
        "extracted": extracted,
        "duration_s": result.get("duration_s", result.get("duration_ms", 0) / 1000),
        "timed_out": result.get("timed_out", False),
    }


def evaluate(input_file: str, resume: bool, concurrency: int, specific_ids: list = None):
    os.makedirs(RESULT_DIR, exist_ok=True)
    init_csv()

    progress = load_progress() if resume else {"completed": []}
    completed_set = set(progress["completed"])

    if resume and completed_set:
        log.info(f"Resume mode: {len(completed_set)} questions completed, skipping them.")

    log.info(f"Loading questions from local JSON file: {input_file}")
    questions = load_questions_from_file(input_file)
    log.info(f"Successfully loaded {len(questions)} questions.")

    # Filter by specific IDs if provided
    if specific_ids:
        todo = [qid for qid in questions.keys() if qid in specific_ids and qid not in completed_set]
    else:
        todo = [qid for qid in questions.keys() if qid not in completed_set]

    log.info(f"Pending inference: {len(todo)} questions.")

    if not todo:
        log.info("All questions completed, nothing to execute.")
        return

    total = len(todo)
    lock = threading.Lock()
    detail_f = open(DETAIL_LOG, "a", encoding="utf-8")
    finished_count = 0

    def on_question_done(qid: int, qr: dict):
        nonlocal finished_count
        result = qr["result"]
        extracted = qr["extracted"]
        duration_s = qr["duration_s"]

        with lock:
            finished_count += 1
            if result["success"]:
                log.info(f"  [#{qid}] ✓ Inference completed (took {duration_s:.1f}s) [{finished_count}/{total}]")
            else:
                log.warning(
                    f"  [#{qid}] ✗ Execution failed (took {duration_s:.1f}s): {result['error']} [{finished_count}/{total}]")

            # Write result to CSV
            append_to_csv(qid, extracted)

            # Record detailed log
            detail_entry = {
                "question_id": qid,
                "session_id": result.get("session_id", ""),
                "success": result["success"],
                "duration_s": duration_s,
                "extracted_answer": extracted,
                "agent_reply": result.get("reply", ""),
                "timestamp": datetime.now().isoformat(),
            }
            detail_f.write(json.dumps(detail_entry, ensure_ascii=False) + "\n")
            detail_f.flush()

            # Update progress
            progress["completed"].append(qid)
            save_progress(progress)

    # Execute inference
    if concurrency <= 1:
        for idx, qid in enumerate(todo):
            log.info(f"[{idx + 1}/{total}] Executing question #{qid} ...")
            qr = _process_one_question(qid, questions[qid])
            on_question_done(qid, qr)
    else:
        log.info(f"Concurrent mode: Up to {concurrency} questions running simultaneously.")
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_qid = {executor.submit(_process_one_question, qid, questions[qid]): qid for qid in todo}
            for future in as_completed(future_to_qid):
                qid = future_to_qid[future]
                try:
                    qr = future.result()
                    on_question_done(qid, qr)
                except Exception as e:
                    log.error(f"  [#{qid}] Thread exception: {e}")

    detail_f.close()
    log.info(f"All executions finished, results saved to: {RESULT_CSV}")


def main():
    parser = argparse.ArgumentParser(description="CTBench openclaw Agent Batch Inference Script")
    parser.add_argument("-i", "--input", type=str, default=DEFAULT_INPUT_JSON,
                        help=f"Path to the input JSON file (default: {DEFAULT_INPUT_JSON})")
    parser.add_argument("--questions", type=str, default=None,
                        help="Specify list of question IDs to run, comma-separated (e.g., 1,2,5)")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help=f"Concurrency limit (default {DEFAULT_CONCURRENCY})")
    parser.add_argument("--resume", action="store_true", help="Resume from the last interruption point")
    args = parser.parse_args()

    specific_ids = [int(x.strip()) for x in args.questions.split(",")] if args.questions else None

    evaluate(args.input, args.resume, args.concurrency, specific_ids)


if __name__ == "__main__":
    main()