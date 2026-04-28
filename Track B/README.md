# Track B: Telco Troubleshooting and Optimization Agentic Challenge

## Competition Overview

This task focuses on IP network operations and maintenance. Participants are required to build intelligent agents that complete IP network fault diagnosis and troubleshooting tasks by calling the CLI simulation interfaces provided by the **Agent Tool Server**.

**Base Model:** The entire competition (Phase 1 / 2 / 3) uses **Qwen3.5-35B-A3B** as the base model. Participants may fine-tune the model (LoRA, full fine-tuning, etc.), but are not allowed to replace it with a different architecture or a model of a different parameter scale.

**Server Core Capabilities:**
- Simulates CLI interactions for network devices (Huawei / Cisco / H3C)
- Supports regex whitelist validation for 45+ command categories
- Faithfully reproduces vendor-level syntax errors (incomplete / unrecognized / ambiguous / wrong parameter)
- Multiple command output file types covering routing tables, BGP, VXLAN, SRv6, and more

---

## Competition Design

### Schedule Overview

| Phase | Name | Duration      | Purpose | Problem Scale & Requirements                     | Participant Actions                                                   |
|-------|------|---------------|---------|--------------------------------------------------|-----------------------------------------------------------------------|
| **Phase 1** | Open Practice | 3 April–4 May | Debug Agent, familiarize with API | 50 problems / Multi vendor / advanced protocols  | Run locally, submit result.csv                                        |
| **Phase 2** | Elimination Round | 4 May-18 May  | Select top participants for Phase 3 | 100 problems / multi-vendor / advanced protocols | Run locally, 1 submission only, upload execution trace and result.csv |
| **Phase 3** | Final | 18 May-29 May | Final ranking | 70 problems / multi-vendor / advanced protocols    | Server-side Docker auto-execution                                     |

### Detailed Problem Distribution

| Phase | Problem Composition                                                 | Task Types | Protocols Covered (examples, not exhaustive) | Device Vendors       |
|-------|---------------------------------------------------------------------|------------|-------------------------------|----------------------|
| **Phase 1** | 32-node financial network & 22-node cloud computing network (50 problems) | Topology reconstruction, path query, fault localization | +LLDP, OSPF, VXLAN            | **Multi-vendor mix** |
| **Phase 2** | 40-node campus network (100)                                        | Topology reconstruction, path query, fault localization | +VLAN, VRRP, MP-BGP           | **Multi-vendor mix** |
| **Phase 3** | 64-node financial network (70)                                      | Topology reconstruction, path query, fault localization | +VXLAN, EVPN, SRv6, ISIS, BGP | **Multi-vendor mix** |

### Phase 1 (Open Practice)

- **Number of Problems:** 50
- **Execution Environment:** Participants run Agent locally
- **Base Model:** Qwen3.5-35B-A3B (participants may fine-tune)
- **API Call Limit:** Max **1,000 API calls per participant per day** (only Phase 1 enforces this daily quota; Phase 2 and Phase 3 do not have this restriction)
- **Scoring:** Participants submit `result.csv`; scored by **accuracy only** (the sole evaluation metric for Phase 1)
- **Tool Call Rules:**
  - Agent must call tools **sequentially** when solving a single problem; no concurrent calls within a problem
  - Max concurrency per participant is **2** (i.e., at most 2 problems running in parallel)
- **Onboarding:** Server source code and local deployment guide provided to help participants deploy locally

### Phase 2 (Elimination Round)

- **Number of Problems:** 100 (released in batches of **20 problems every 3 days**)
- **Base Model:** Qwen3.5-35B-A3B (participants may fine-tune)
- **Submission Limit:** Each participant is allowed **only three submissions**; execution trace must be uploaded to the server. Unlike Phase 1, there is no daily API call quota, but participants must ensure at least a single run completes successfully.
- **Scoring:** **Accuracy** as the primary metric; for participants with the same accuracy, the **number of API calls used to solve correct problems** serves as the secondary metric (fewer calls = higher rank)
- **Selection Mechanism:** Participants reserve a submission time slot; top 30 advance to the final
- **Focus:** Agent generalization across multi-vendor environments and stability in one-shot execution

### Phase 3 (Final)

- **Number of Problems:** 70
- **Participants:** Top 30 selected from Phase 2
- **Execution Environment:** Organizer-provided GPU resources + isolated Docker environment
- **Base Model:** Qwen3.5-35B-A3B (participants may fine-tune)
- **Time Limit:** Must be completed within 24 hours
- **Network Data:** Uses **different network data** from Phase 1 / 2
- **Focus:** Complex protocol reasoning (SRv6/EVPN) and deep fault isolation in large-scale networks (64 nodes)
- **Resource Allocation:**
  - **GPU Resources:** Huawei Cloud GPU instances for deploying the base model, independently allocated per participant
  - **Agent Tool Server:** Deployed on HuggingFace free CPU resources (free within 24h)
- **Base Model:** Qwen3.5-35B-A3B, deployed by the organizer to each participant's GPU instance
- **Submission Requirements:** Participants must upload their **fine-tuned model weights** and **Agent code** to the organizer; the organizer will deploy and execute them in the isolated environment

---

## Authentication & Security

### Agent Tool Server API

To help participants get started quickly, we provide access to an online Agent Tool API with two locations. 
```
- Hong Kong & Others: 124.71.227.61
- China: 120.46.145.77
```


### Token Authentication

All API requests must include a Bearer Token:

```
Authorization: ${Token}
```

Token type is **READ permission**, only allowing calls to the `/api/agent/execute` endpoint.

### Security Measures

| Measure | Implementation |
|---------|---------------|
| Access Authentication | Bearer Token validation |
| Rate Limiting | Nginx rate limit: 50 req/s per IP |
| Daily Quota | 1,000 calls per participant per day (Phase 1 only) |
| Concurrency Limit | Max 2 concurrent per participant |
| Network Isolation | Backends have no public IP; only the router is exposed |
| Fault Isolation | Backend failures are automatically circuit-broken; no impact on other nodes |

---

## API Reference

### Request

```
POST /api/agent/execute
Content-Type: application/json
Authorization: Bearer ${Token}

{
    "device_name": "BoardLeaf1",
    "command": "display ip routing-table"
}
```

### Response

**Success (200):**
```json
{
    "status": "success",
    "device_name": "BoardLeaf1",
    "vendor": "huawei",
    "command_executed": "display ip routing-table",
    "result": "<BoardLeaf1> display ip routing-table\n..."
}
```

**Command Syntax Error (422):**
```json
{
    "status": "execution_failed",
    "device_name": "BoardLeaf1",
    "vendor": "huawei",
    "command_executed": "display ip rout",
    "result": "<BoardLeaf1> display ip rout\n                              ^\nError: Incomplete command found at '^' position."
}
```

**Device Not Found (404):**
```json
{
    "error": "Device 'UnknownDevice' not found"
}
```

---

## Available Device Commands

This section provides a comprehensive reference of all network device commands available for interacting with the Agent Tool Server. Commands are organized by vendor and device type, covering routers, switches, and firewalls from **Huawei (VRP)**, **Cisco (IOS/IOS-XE/NX-OS)**, and **H3C (Comware)**.

### Supported Vendors & Device Types

| Vendor | Routers | Switches | Firewalls |
|--------|---------|----------|-----------|
| Huawei (VRP) | Supported | Supported | Supported |
| Cisco (IOS/IOS-XE/NX-OS) | Supported | Supported | Supported |
| H3C (Comware) | Supported | Supported | N/A |

### Placeholder Notation

| Placeholder | Meaning |
|-------------|---------|
| `[x]` | Port name (e.g., `GigabitEthernet0/0/1`) |
| `[id]` | Interface ID |
| `[xxx]` | VRF or VPN instance name |
| `[type] [num]` | Interface type and number (e.g., `GigabitEthernet 1/0/1`) |

> Replace placeholders with actual values from your network topology when calling commands via the API.

---

### 1. Configuration & Logging

| Function | Huawei (All) | Cisco Router | Cisco Switch | Cisco FW | H3C Router | H3C Switch |
|----------|-------------|--------------|--------------|----------|------------|------------|
| View current configuration | `display current-configuration` | `show running-config` | `show running-config` | `show running-config` | `display current-configuration` | `display current-configuration` |
| View log buffer | `display logbuffer` | `show logging` | `show logging` | `show logging` | `display logbuffer` | `display logbuffer` |

### 2. Alarms & Monitoring

| Function | Huawei (All) | Cisco Router | Cisco Switch | Cisco FW | H3C Router | H3C Switch |
|----------|-------------|--------------|--------------|----------|------------|------------|
| Active alarms | `display alarm active` | `show facility-alarm status` | N/A | N/A | `display alarm active` | `display alarm` |
| Memory usage | `display memory-usage` | `show processes memory` | `show processes memory` | `show processes memory` | `display memory` | `display memory` |

### 3. Interface & Link Status

| Function | Huawei (All) | Cisco Router | Cisco Switch | Cisco FW | H3C Router | H3C Switch |
|----------|-------------|--------------|--------------|----------|------------|------------|
| LLDP neighbor summary | `display lldp neighbor brief` | `show lldp neighbors` | `show lldp neighbors` | N/A | `display lldp neighbor-information` | `display lldp neighbor-information` |
| Interface brief status | `display interface brief` | `show ip interface brief` | `show interface brief` | `show interface summary` | `display interface brief` | `display interface brief` |
| Interface description | `display interface description` | `show interface description` | `show interface description` | N/A | N/A | N/A |
| Link aggregation (Eth-Trunk) | `display eth-trunk` | `show etherchannel summary` | N/A | N/A | `display link-aggregation summary` | N/A |
| IP interface brief | `display ip interface brief` | `show ip interface brief` | `show ip interface brief` | `show interface ip brief` | `display ip interface brief` | `display ip interface brief` |
| IPv6 interface | `display ipv6 interface [x]` | `show ipv6 interface [type] [num]` | N/A | N/A | `display ipv6 interface [type] [num]` | N/A |

### 4. Layer 2 (VLAN, MAC, STP)

| Function | Huawei (All) | Cisco Router | Cisco Switch | Cisco FW | H3C Router | H3C Switch |
|----------|-------------|--------------|--------------|----------|------------|------------|
| VLAN status | `display vlan` | `show vlans` | `show vlan` | `show vlan` | `display vlan` | `display vlan` |
| MAC address table | `display mac-address` | N/A | `show mac address-table` | `show mac-address-table` | `display mac-address` | `display mac-address` |
| STP summary | `display stp brief` | `show spanning-tree summary` | `show spanning-tree summary` | N/A | N/A | `display stp brief` |
| STP interface state | `display stp interface [x]` | `show spanning-tree interface [id]` | N/A | N/A | `display stp interface brief` | N/A |

### 5. Address Resolution & Neighbor Discovery

| Function | Huawei (All) | Cisco Router | Cisco Switch | Cisco FW | H3C Router | H3C Switch |
|----------|-------------|--------------|--------------|----------|------------|------------|
| ARP table | `display arp all` (Router)<br/>`display arp` (Switch/FW) | `show ip arp` | `show ip arp` | `show arp` | `display arp all` | `display arp all` |
| IPv6 neighbors | `display ipv6 neighbors` | `show ipv6 neighbors` | `show ipv neighbor` | `show ipv6 neighbor` | `display ipv6 neighbors` | `display ipv6 neighbor all` |

### 6. Routing

| Function | Huawei (All) | Cisco Router | Cisco Switch | Cisco FW | H3C Router | H3C Switch |
|----------|-------------|--------------|--------------|----------|------------|------------|
| IPv4 routing table | `display ip routing-table` | `show ip route` | `show ip route` | `show route` | `display ip routing-table` | `display ip routing-table` |
| VPN instance routing table | `display ip routing-table vpn-instance [x]` | `show ip route vrf [xxx]` | N/A | N/A | `display ip routing-table vpn-instance [xxx]` | N/A |
| IPv6 routing table | `display ipv6 routing-table` | `show ipv6 route` | `show ipv6 route` | `show ipv6 route` | `display ipv6 routing-table` | `display ipv6 routing-table` |

### 7. OSPF

| Function | Huawei (All) | Cisco Router | Cisco Switch | Cisco FW | H3C Router | H3C Switch |
|----------|-------------|--------------|--------------|----------|------------|------------|
| OSPF neighbor state | `display ospf peer` | `show ip ospf neighbor` | `show ip ospf neighbors` | `show ospf neighbor` | `display ospf peer` | `display ospf peer` |
| OSPF internal routes | `display ospf routing` | `show ip route ospf` | `show ip ospf route` | `show route ospf` | `display ospf routing` | `display ospf routing` |
| OSPF LSDB | `display ospf lsdb` | `show ip ospf database` | `show ip ospf database` | `show ospf database` | `display ospf lsdb` | `display ospf lsdb` |

### 8. BGP

| Function | Huawei (All) | Cisco Router | Cisco Switch | Cisco FW | H3C Router | H3C Switch |
|----------|-------------|--------------|--------------|----------|------------|------------|
| BGP EVPN routes | `display bgp evpn all routing-table` | `show bgp l2vpn evpn` | N/A | N/A | `display bgp l2vpn evpn` | `display bgp l2vpn evpn` |
| BGP VPNv4 routes | `display bgp vpnv4 all routing-table` | `show bgp vpnv4 unicast all` | N/A | N/A | `display bgp vpnv4 all routing-table` | `display bgp routing-table vpnv4` |

### 9. VXLAN

| Function | Huawei (All) | Cisco Router | Cisco Switch | Cisco FW | H3C Router | H3C Switch |
|----------|-------------|--------------|--------------|----------|------------|------------|
| VXLAN tunnel status | `display vxlan tunnel` | `show nve vni`<br/>`show nve peers` | N/A | `show nve` | `display vxlan tunnel` | `display vxlan tunnel` |
| VXLAN troubleshooting | `display vxlan troubleshooting` | `show nve [detailed debugs]` | N/A | N/A | `display vxlan troubleshooting` | N/A |

### 10. High Availability & Forwarding

| Function | Huawei (All) | Cisco Router | Cisco Switch | Cisco FW | H3C Router | H3C Switch |
|----------|-------------|--------------|--------------|----------|------------|------------|
| VRRP verbose state | `display vrrp verbose` | `show vrrp detail` | N/A | N/A | `display vrrp verbose` | N/A |
| BFD session state | `display bfd session all` | `show bfd neighbors` | `show bfd neighbors` | `show bfd neighbors` | `display bfd session` | `display bfd session` |

### 11. DHCP

| Function | Huawei (All) | Cisco Router | Cisco Switch | Cisco FW | H3C Router | H3C Switch |
|----------|-------------|--------------|--------------|----------|------------|------------|
| DHCP address pool | `display ip pool` | `show ip dhcp pool` | N/A | `show ip local pool [x]` | `display ip pool` | `display ip pool` |

### 12. SRv6 (Segment Routing over IPv6)

| Function | Huawei (All) | Cisco Router | Cisco Switch | Cisco FW | H3C Router | H3C Switch |
|----------|-------------|--------------|--------------|----------|------------|------------|
| SRv6 Policy status | `display srv6-te policy status` | `show segment-routing srv6 policy` | N/A | N/A | `display segment-routing ipv6 te policy` | N/A |
| SRv6 Policy details | `display srv6-te policy` | N/A | N/A | N/A | `display segment-routing ipv6 te policy` | N/A |
| SRv6 End forwarding info | `display segment-routing ipv6 local-sid end forwarding` | N/A | N/A | N/A | `display segment-routing ipv6 local-sid` | N/A |
| SRv6 End.X forwarding info | `display segment-routing ipv6 local-sid end-x forwarding` | N/A | N/A | N/A | `display segment-routing ipv6 local-sid` | N/A |

---

### Quick Command Reference by Vendor

Below are consolidated command lists grouped by vendor for quick reference when building your Agent's tool-calling logic.

<details>
<summary><strong>Huawei (VRP) - All Device Types</strong></summary>

```
# Configuration & Logging
display current-configuration
display logbuffer

# Monitoring
display alarm active
display memory-usage

# Interface & Link
display lldp neighbor brief
display interface brief
display interface description
display eth-trunk
display ip interface brief
display ipv6 interface [x]

# Layer 2
display vlan
display mac-address
display stp brief
display stp interface [x]

# Address Resolution
display arp all          # Router; use "display arp" on Switch/Firewall
display ipv6 neighbors

# Routing
display ip routing-table
display ip routing-table vpn-instance [x]
display ipv6 routing-table

# OSPF
display ospf peer
display ospf routing
display ospf lsdb

# BGP
display bgp evpn all routing-table
display bgp vpnv4 all routing-table

# VXLAN
display vxlan tunnel
display vxlan troubleshooting

# HA & Forwarding
display vrrp verbose
display bfd session all

# DHCP
display ip pool

# SRv6
display srv6-te policy status
display srv6-te policy
display segment-routing ipv6 local-sid end forwarding
display segment-routing ipv6 local-sid end-x forwarding
```

</details>

<details>
<summary><strong>Cisco (IOS/IOS-XE/NX-OS) - Router</strong></summary>

```
# Configuration & Logging
show running-config
show logging

# Monitoring
show facility-alarm status
show processes memory

# Interface & Link
show lldp neighbors
show ip interface brief
show interface description
show etherchannel summary
show ipv6 interface [type] [num]

# Layer 2
show vlans
show spanning-tree summary
show spanning-tree interface [id]
show ip arp
show ipv6 neighbors

# Routing
show ip route
show ip route vrf [xxx]
show ipv6 route

# OSPF
show ip ospf neighbor
show ip route ospf
show ip ospf database

# BGP
show bgp l2vpn evpn
show bgp vpnv4 unicast all

# VXLAN
show nve vni
show nve peers
show nve [detailed debugs]

# HA & Forwarding
show vrrp detail
show bfd neighbors

# DHCP
show ip dhcp pool

# SRv6
show segment-routing srv6 policy
```

</details>

<details>
<summary><strong>Cisco (IOS/IOS-XE/NX-OS) - Switch</strong></summary>

```
# Configuration & Logging
show running-config
show logging

# Monitoring
show processes memory

# Interface & Link
show lldp neighbors
show interface brief
show ip interface brief

# Layer 2
show vlan
show mac address-table
show spanning-tree summary
show ip arp
show ipv neighbor

# Routing
show ip route
show ipv6 route

# OSPF
show ip ospf neighbors
show ip ospf route
show ip ospf database

# HA & Forwarding
show bfd neighbors

# DHCP
show ip dhcp pool
```

</details>

<details>
<summary><strong>Cisco (IOS/IOS-XE/NX-OS) - Firewall</strong></summary>

```
# Configuration & Logging
show running-config
show logging

# Monitoring
show processes memory

# Interface & Link
show interface summary
show interface ip brief

# Layer 2
show vlan
show mac-address-table
show arp
show ipv6 neighbor

# Routing
show route
show ipv6 route

# OSPF
show ospf neighbor
show route ospf
show ospf database

# VXLAN
show nve

# DHCP
show ip local pool [x]
```

</details>

<details>
<summary><strong>H3C (Comware) - Router</strong></summary>

```
# Configuration & Logging
display current-configuration
display logbuffer

# Monitoring
display alarm active
display memory

# Interface & Link
display lldp neighbor-information
display interface brief
display ip interface brief
display ipv6 interface [type] [num]

# Layer 2
display vlan
display mac-address
display stp brief
display arp all
display ipv6 neighbors

# Routing
display ip routing-table
display ip routing-table vpn-instance [xxx]
display ipv6 routing-table

# OSPF
display ospf peer
display ospf routing
display ospf lsdb

# BGP
display bgp l2vpn evpn
display bgp vpnv4 all routing-table

# VXLAN
display vxlan tunnel
display vxlan troubleshooting

# HA & Forwarding
display vrrp verbose
display bfd session

# DHCP
display ip pool

# SRv6
display segment-routing ipv6 te policy
display segment-routing ipv6 local-sid
```

</details>

<details>
<summary><strong>H3C (Comware) - Switch</strong></summary>

```
# Configuration & Logging
display current-configuration
display logbuffer

# Monitoring
display alarm
display memory

# Interface & Link
display lldp neighbor-information
display interface brief
display ip interface brief

# Layer 2
display vlan
display mac-address
display stp brief
display arp all
display ipv6 neighbor all

# Routing
display ip routing-table
display ipv6 routing-table

# OSPF
display ospf peer
display ospf routing
display ospf lsdb

# BGP
display bgp l2vpn evpn
display bgp routing-table vpnv4

# VXLAN
display vxlan tunnel

# HA & Forwarding
display bfd session

# DHCP
display ip pool
```

</details>

---

### Important Notes

- Commands marked as **N/A** are not supported on that vendor/device combination. Attempting to use them will result in a command syntax error (HTTP 422).
- **Placeholder values** in brackets (e.g., `[x]`, `[xxx]`) must be replaced with actual interface names, VRF/VPN instance names, or interface types/numbers appropriate for your network topology.
- On **Cisco NX-OS switches**, certain features (e.g., NVE/VXLAN) require activation via `configure terminal` followed by the appropriate `feature` command before the related `show` commands become available.
- The Agent Tool Server faithfully reproduces vendor-level syntax errors including **incomplete**, **unrecognized**, **ambiguous**, and **wrong parameter** errors (HTTP 422). Use this to validate your Agent's error handling.
---

## Participant Guide

### Environment Requirements

- Participants run Agent **locally** (Phase 1 / Phase 2)
- Agent calls the remote Agent Tool Server via HTTP
- No need to deploy the server locally (but a fallback option is provided)

### Model Rules

1. **Base Model:** The entire competition (Phase 1 / 2 / 3) uses **Qwen3.5-35B-A3B** as the base model
2. **Fine-tuning Allowed:** Participants may fine-tune Qwen3.5-35B-A3B (LoRA, full fine-tuning, etc.)
3. **No Replacement:** Replacing it with a different architecture or parameter scale is not allowed; final inference must be based on Qwen3.5-35B-A3B
4. **Phase 3 Submission:** For Phase 3, the organizer deploys the base model on GPU instances; participants only need to submit their fine-tuned weights

### API Call Rules

1. **Sequential Calls:** When solving a single problem, the Agent must call tools sequentially; no concurrent calls within a problem
2. **Concurrency Limit:** Each participant may run at most 2 tasks simultaneously (2 problems in parallel)
3. **Daily Quota:** Max 1,000 API calls per day (Phase 1 only; Phase 2 and Phase 3 are not subject to this limit)
4. **Authentication:** Every request must include the Authorization header

### Local Server Deployment

If server access issues occur, participants may deploy the Agent Tool Server locally:

1. First, unzip `devices_outputs.zip` inside the same directory
2. Then, run `python server.py` to deploy the local server.
3. An example agent workflow is provide in `agent/` folder.

After local deployment, change the Agent's target URL to `http://localhost:7860/api/agent/execute`; no Token required.

---

## Evaluation Metrics & Scoring

Each phase uses different evaluation criteria, progressively increasing in rigor:

### Scoring by Phase

| Phase | Primary Metric | Secondary Metric | Description |
|-------|---------------|-----------------|-------------|
| **Phase 1** | Accuracy | — | Scored by accuracy only |
| **Phase 2** | Accuracy | API Call Count | Accuracy first; ties broken by fewer API calls on correct problems |
| **Phase 3** | Accuracy + Speed | — | Correctness and time-based scoring (see table below) |

### Phase 3: Detailed Scoring Standards

- **Pass@1 (Consistency Metric):** Rewards agents with high success rates across 4 independent trials to ensure solution robustness.
- **TTS (Time To Solve):** Measures efficiency in pinpointing and fixing faults via logical deduction.
- **Execution Guardrail:** A strict 15-minute cutoff is implemented to prevent infinite loops and resource exhaustion.

Points are awarded only if the task is completed correctly within the allotted time. We will select the fastest solution in 4 generations. The faster the resolution, the higher the score:

| Answering time            | Discount |
| ------------------------- | -------- |
| `< 5 minutes`             | 100%     |
| `5 minutes  - 10 minutes` | 80%      |
| `10 minutes - 15 minutes` | 60%      |
| `> 15 minutes`            | 0%       |

---

## Quick Start: Running the Agent with OpenClaw

The `agent/` directory provides a ready-to-use agent solution based on **OpenClaw** (an open-source agentic framework). By combining the pre-configured skills, OpenClaw configuration files, and the batch evaluation script, participants can quickly launch an agent to solve CTBench problems locally.

### Prerequisites

1. **Clone and install OpenClaw** from its open-source repository, and ensure it runs locally (refer to the OpenClaw official documentation for installation steps).
2. **Python 3.8+** and **Node.js** installed.
3. **Agent Tool Server** running locally at `http://127.0.0.1:7860` (or the remote server with a valid Token).

### Directory Structure

```
agent/
├── openclaw_config/          # OpenClaw configuration files
│   ├── IDENTITY.md           # Agent identity definition (name, persona)
│   ├── SOUL.md               # Code of conduct & core principles
│   ├── USER.md               # Competition scenario & requirements
│   ├── AGENTS.md             # Agent coordination settings
│   └── TOOLS.md              # Available tools & NOC API conventions
├── skills/                   # Skill definitions for the agent
│   ├── infra_maintenance/    # Infrastructure maintenance (config/log/alarm/memory/LLDP)
│   ├── l2_link/              # Layer 2 link O&M (interface/VLAN/MAC/STP)
│   ├── l3_route/             # Layer 3 routing (IP/ARP/OSPF/BGP)
│   └── adv_tunnel/           # Advanced tunnels (VXLAN/VRRP/BFD/DHCP/SRv6)
├── evaluate_openclaw.py      # Batch evaluation script
├── evaluate_openclaw_guideline.md  # Detailed usage guide for the evaluation script
└── requirements.txt          # Python dependencies
```

### Configuration

1. **Copy the `openclaw_config/` files** into your local OpenClaw project's configuration directory (or symlink them) so that OpenClaw loads the agent identity, tools, and skills at startup.

2. **Copy the `skills/` directory** into your local OpenClaw project's skills directory so that the four network O&M skills (`infra_maintenance`, `l2_link`, `l3_route`, `adv_tunnel`) are available to the agent.

3. **Edit `evaluate_openclaw.py`** and set the following paths at the top of the file:

   ```python
   # Set to the absolute path of your local OpenClaw project directory
   OPENCLAW_DIR = r"C:\path\to\your\openclaw"

   # Set to the absolute path where OpenClaw stores session logs
   OPENCLAW_SESSION_DIR = r"C:\Users\YourUser\.openclaw\agents\main\sessions"
   ```

### Running the Agent

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

### How It Works

1. **`evaluate_openclaw.py`** loads questions from the input JSON file, then invokes the locally running OpenClaw agent for each question.
2. The OpenClaw agent, guided by `openclaw_config/` (identity, tools, and behavioral rules), uses the four **skills** to collect device data via the API (`http://127.0.0.1:7860/api/agent/execute`).
3. The agent analyzes the collected data and produces a final answer.
4. The script extracts the answer from the OpenClaw session log and writes results to `agent/eval_results/result.csv`.

### Output

Results are saved under `agent/eval_results/`:

| File | Description |
|------|-------------|
| `result.csv` | Final answers (`id`, `prediction`) — the file to submit |
| `eval_detail.jsonl` | Detailed execution logs per question |
| `progress.json` | Progress tracking for the `--resume` feature |

For more details on the evaluation script, see [`agent/evaluate_openclaw_guideline.md`](agent/evaluate_openclaw_guideline.md).