# Track A: Telco Troubleshooting and Optimization Agentic Challenge

## 1. Competition Overview

This is the first global evaluation competition focused on the design of AI Agents for wireless network troubleshooting and optimization. Participants are required to build intelligent agents that complete troubleshooting and optimization tasks by calling the simulation interfaces provided by the **Agent Tool Server**.

**Base Model:** The entire competition  uses **Qwen3.5-35B-A3B** as the base model. Participants may fine-tune the model (LoRA, full fine-tuning, etc.), but are not allowed to replace it with a different architecture or a model of a different parameter scale.

The challenge will leverage a dedicated, multi-vendor Agent Sandbox capable of modeling realistic, real-world telecom scenarios, together with a central orchestration framework that
- Exposes structured questions to participating agents
- Connects agents to domain-specific Tools APIs
- Coordinates interactions between agents and the sandbox environment

**Agent Design:** We encourage participants to develop their own Agent structure in this competition by modifying main.py.

---

## 2. Competition Design

### Schedule Overview

| Phase        | Name              | Duration      | Purpose                             | Problem Scale                                  | Participant Actions                                          |
| ------------ | ----------------- | ------------- | ----------------------------------- | ---------------------------------------------- | ------------------------------------------------------------ |
| **Phase 1**  | Open Practice     | 3 April–4 May | Debug Agent, familiarize with API   | 2000 problems for train, 500 problems for test | submit result.csv                                            |
| **Phase 2** | Elimination Round | 4 May-18 May  | Select top participants for Phase 3 | 500 problems                                   | submit result.csv                                            |
| **Phase 3**  | Final Round       | 18 May-29 May | Final ranking                       | 500 problems                                   | Submit a zip file, including the design of the Agent (main.py) and the fine-tuned model weights (Qwen3.5-35B) |



### Phase 1

- **Number of Problems:** training set has 2000 questions with questions and answers, test set has 500 questions with questions and without answers
- **Base Model:** Qwen3.5-35B-A3B (participants may fine-tune), locally deployed by the participants
- **Scoring:** according to the result.csv; scored by accuracy according to the test set
- **Submission:** unlimited

### Phase 2

- **Number of Problems:** 500 (with questions, without answers)
- **Base Model:** Qwen3.5-35B-A3B (participants may fine-tune), locally deployed by the participants
- **Scoring:** according to the result.csv; scored by accuracy
- **Submission:** Participants can submit three times
- **Selection Mechanism:**  Top 30 Participants to the final round
- **Hint:** Phase 2 is only a test set to eval participants' agents. This is not the final ranking, which relies only on the Phase 3.

### Phase 3

- **Participants:** Top 30 selected from Phase 2
- **Number of Problems:** 500 (private dataset without questions or answers)
- **Base Model:** Qwen3.5-35B-A3B (participants may fine-tune), uniformly deployed by the organizer to each participant's GPU instance (participants may submit fine-tuned weights)
- **Scoring:** Run the`main.py`; scored by accuracy and execution time
- **Submission:** Participants can submit only once.
- **Hint:**  Evaluated participants' agents by the organizer

---

## 3. Server Architecture


### Architecture Topology

```
                        ┌─────────────────┐
                        │  Environment    │
                        │(private dataset)│
                        └────────┬────────┘
                                 │ 
                        ┌────────▼────────┐
                        │   server.py     │  
                        │                 │ 
                        │(tools&simulator)│  
                        │Can't be modified│
                        │                 │
                        └────────┬────────┘
                                 │ 
              ┌──────────────────┼──────────────────┐
              │                  │                  │
     ┌────────▼──────┐  ┌────────▼──────┐  ┌────────▼──────┐
     │               │  │               │  │ Qwen3.5-35B   │
     │    main.py    │  │     skills    │  │ fine-tuned    │
     │(Agent design) │  │   (Optional)  │  │   weights     │
     │can be modified│  │               │  │  (Optional)   │
     └───────────────┘  └───────────────┘  └───────────────┘
```
---

## 4. Participant Guide

Participants may deploy the Agent Tool Server locally:

```
# Deploy the server
python server.py

# Run the Agent (example)
python main.py
```

The main.py will output the result.csv in this format:

| ID                                   | Answer |
| ------------------------------------ | ------ |
| 102fb27a-f7a4-480b-ac9b-ef51d7feb912 | C3     |
| 1cb15db5-674a-4586-9ff1-a739e00c98e8 | C5\|C7 |

The questions could be single-answer questions or multiple-answer questions. The answers of multiple-answer questions are divided by "|", e.g., 'C3|C7' or 'C5|C9|C11|C20', in ascending order.

---

## 5. Deployment File Inventory

| File                     | Purpose                            |
|--------------------------| ---------------------------------- |
| `server.py`              | Definition of tools and simulators |
| `utils.py`               | Related functions                  |
| `requirements.txt`       | Python dependencies                |
| `data/Phase_1/test.json` | Question description and choices   |
| `main.py`                | Interface and Agent runner         |
| `results/result.csv`     | output files                       |

---

## 6. Evaluation Criteria

For each question, the score is given by  score = accuracy $\times$ discount, where the accuracy is defined by IOU (intersection over union), that is, $$\text{accuracy} = \text{intersection}(\text{answers}, \text{ground truth})/\text{union}(\text{answers}, \text{ground truth}),$$
and to limit the answering time, the discount is calculated as:

| Answering time            | Discount |
| ------------------------- | -------- |
| `< 5 minutes`             | 100%     |
| `5 minutes  - 10 minutes` | 80%      |
| `10 minutes - 15 minutes` | 60%      |
| `> 15 minutes`            | 0%       |