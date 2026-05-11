# Additional guidelines to run Openclaw Agent

This repository contains a Python-based batch inference pipeline for evaluating the openclaw AI agent against the CTBench dataset.

The script automates the process of loading questions from a local JSON file, invoking the Node.js-based agent, extracting the structured answers, and compiling the results into a CSV format suitable for downstream evaluation.

## **Features**

* **Local JSON Processing:** Reads benchmark tasks directly from a locally stored JSON file format without needing an external API.  
* **Concurrent Execution:** Supports multithreading to evaluate multiple questions simultaneously, significantly speeding up the benchmark process.  
* **Resumable Operations:** Tracks progress automatically using the id field. If the script is interrupted, you can resume exactly where it left off without re-running completed scenarios.  
* **Robust Error Handling & Timeouts:** Implements strict timeouts (default 10 minutes per question) and force-kill mechanisms to prevent hanging processes.  
* **Answer Extraction:** Uses regex patterns to extract specific structured reasoning paths (e.g., A(1) \-\> B(2)) or falls back to the agent's complete final response.

## **Prerequisites**

Before running the script, ensure you have the following installed and configured:

* **Python 3.8+**  
* **Node.js** (Required to execute the openclaw agent)  
* **Openclaw Agent:** The openclaw repository must be cloned and accessible on your local machine.

*Note: Since the script reads from a local file, third-party networking libraries like requests are no longer required.*

## **Input Data Format**

The script expects a JSON file containing an array of task objects. Each object must contain a nested task object with both an id (integer) and a question (string).

**Example questions.json:**

\[  
  {  
    "scenario\_id": "535afb0d-fa81-419b-9bcc-b456d032df5d",  
    "task": {  
      "question": "The link planning data of Gamma-Aegis-01 within the Big Data Zone has been accidentally deleted...",  
      "id": 1  
    }  
  }  
\]

*(The script parses task.id as the primary identifier for saving files, tracking progress, and building the output CSV.)*

## **Configuration**

Before the first run, you **must** update the absolute directory paths at the top of the Python script:

\# \============================================================  
\# Configuration  
\# \============================================================

\# Default input JSON file path  
DEFAULT\_INPUT\_JSON \= "questions.json"

\# Set this to the absolute path of your openclaw project directory  
OPENCLAW\_DIR \= r"C:\\path\\to\\your\\openclaw" 

\# Set this to the absolute path where openclaw stores its session logs  
OPENCLAW\_SESSION\_DIR \= r"C:\\Users\\YourUser\\.openclaw\\agents\\main\\sessions" 

FORCE\_KILL\_TIMEOUT \= 600  \# Timeout per question in seconds  
DEFAULT\_CONCURRENCY \= 2   \# Number of parallel agent executions

## **Usage**

Run the script from the command line using various arguments to control the batch job. By default, it will attempt to process every scenario found in the specified JSON file.

### **Command Line Arguments**

| Argument | Type | Default | Description |
| :---- | :---- | :---- | :---- |
| \-i, \--input | str | questions.json | The path to the input JSON file containing the scenarios. |
| \--questions | str | None | A comma-separated list of specific ids to run. If not provided, all tasks in the file are executed. |
| \--concurrency | int | 2 | Number of concurrent agent invocations. |
| \--resume | flag | False | If passed, skips previously completed IDs based on progress.json. |

### **Examples**

```bash
# Install Python dependencies
pip install -r agent/requirements.txt

# Run all questions from the input JSON
python agent/evaluate_openclaw.py -i data/Phase_1/test.json

# Run specific questions only
python agent/evaluate_openclaw.py -i data/Phase_1/test.json --questions 1,2,5

# Run with concurrency (max 2 for competition compliance)
python agent/evaluate_openclaw.py -i data/Phase_1/test.json --concurrency 2

# Resume from an interrupted run
python agent/evaluate_openclaw.py -i data/Phase_1/test.json --resume
```

## **Output Files**

The script automatically generates an eval\_results directory in the same folder as the script. It contains the following files:

* **result.csv**: The primary output file containing the extracted answers.  
  * **Headers:** id, prediction (where id maps to the task.id from the JSON).  
* **eval\_detail.jsonl**: A JSON Lines log file containing granular details for every processed task, including timestamps, full agent payloads, success status, and execution duration. Highly useful for debugging failed runs.  
* **progress.json**: An internal state file used to track completed ids for the \--resume feature.  
* **\_msg\_\<id\>.txt & \_invoke\_wrapper.js**: Temporary files generated during execution to pass prompts to the Node.js process.

## **Troubleshooting**

* **Script gets stuck indefinitely:** Check the FORCE\_KILL\_TIMEOUT variable. Ensure you have the necessary permissions to execute task kill commands (Windows taskkill or Unix killpg).  
* **Empty prediction in CSV:** This usually means the agent failed to generate a response, or the session log file could not be found. Check OPENCLAW\_SESSION\_DIR and the eval\_detail.jsonl file for specific error messages.  
* **Invalid JSON Structure Warning:** If the script prints Skipping invalid item in JSON, double-check that your JSON format strictly follows the nested structure where task.id and task.question exist.