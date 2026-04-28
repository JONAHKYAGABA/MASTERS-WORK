---
name: infra_maintenance
description: Infrastructure maintenance and connectivity detection (config/log/alarm/memory/LLDP), executing show/display commands via the local NOC API.
metadata:
  openclaw:
    emoji: "🛠️"
---

# infra_maintenance — Infrastructure Maintenance & Connectivity Detection

Used for assessing the operational status of network devices (Huawei/Cisco/H3C): configuration, logs, alarms, memory usage, and LLDP neighbor summary.

## Parameters

- `device_name` (required): Target device name (e.g., `Core-Router-01`)
- `vendor` (required): `Huawei` | `Cisco` | `H3C`
- `question_number` (required): Number representing the current question/problem ID being solved
- `action` (required):
  - `config`: View current configuration
  - `log`: View log buffer
  - `alarm`: View active alarms
  - `memory`: View memory usage
  - `lldp`: View LLDP neighbor summary

## Execution Method (Local NOC API)

- Endpoint: `http://127.0.0.1:5000/api/agent/execute`
- Method: `POST`
- Body: `{ "device_name": "...", "command": "...", "question_number": 1 }`

### Command Mapping

```text
Huawei:
  config: display current-configuration
  log: display logbuffer
  alarm: display alarm active
  memory: display memory-usage
  lldp: display lldp neighbor brief

Cisco:
  config: show running-config
  log: show logging
  alarm: show facility-alarm status
  memory: show processes memory
  lldp: show lldp neighbors

H3C:
  config: display current-configuration
  log: display logbuffer
  alarm: display alarm active
  memory: display memory
  lldp: display lldp neighbor-list
```

### Python requests Example (Recommended)

> Note: In Windows/enterprise environments, system proxies may interfere with local `127.0.0.1` calls; `s.trust_env = False` is used here to disable environment proxies.

```python
import os
import requests

# Remove proxy environment variables to prevent interference with local API calls
for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
    os.environ.pop(key, None)

s = requests.Session()
s.trust_env = False  # Do not read system/environment proxies

url = "http://127.0.0.1:5000/api/agent/execute"
body = {
    "device_name": "Core-Router-01",
    "command": "display logbuffer",
    "question_number": 1,
}

r = s.post(url, json=body, timeout=30)
r.raise_for_status()
print(r.text)
```

## Notes

- Currently designed for read-only queries (show/display) only.
