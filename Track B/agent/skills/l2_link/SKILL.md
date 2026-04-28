---
name: l2_link
description: Interface and Layer 2 link O&M (interface/aggregation/VLAN/MAC/STP), executing show/display commands via the local NOC API.
metadata:
  openclaw:
    emoji: "🔌"
---

# l2_link — Interface & Layer 2 Link O&M

Used for troubleshooting device interface physical status, description information, link aggregation (Eth-Trunk/EtherChannel), VLAN, MAC address table, and STP status.

## Parameters

- `device_name` (required): Device name
- `vendor` (required): `Huawei` | `Cisco` | `H3C`
- `question_number` (required): Number representing the current question/problem ID being solved
- `action` (required):
  - `int_brief`: Interface brief status
  - `int_desc`: Interface description
  - `link_agg`: Link aggregation information
  - `vlan`: VLAN status
  - `mac`: MAC address table
  - `stp_brief`: STP summary
  - `stp_interface`: Interface STP status (Cisco requires `interface_id`)
- `interface_id` (optional): Interface ID (e.g., `Gi1/0/1` / `Eth-Trunk1`, etc.)

## Execution Method (Local NOC API)

- Endpoint: `http://127.0.0.1:5000/api/agent/execute`
- Method: `POST`
- Body: `{ "device_name": "...", "command": "...", "question_number": 1 }`

### Command Mapping

```text
Huawei:
  int_brief: display interface brief
  int_desc: display interface description
  link_agg: display eth-trunk
  vlan: display vlan
  mac: display mac-address
  stp_brief: display stp brief
  stp_interface: display stp interface brief

Cisco:
  int_brief: show ip int brief
  int_desc: show interface description
  link_agg: show etherchannel summary
  vlan: show vlan brief
  mac: show mac address-table
  stp_brief: show spanning-tree brief
  stp_interface: show spanning-tree interface <interface_id>

H3C:
  int_brief: display interface brief
  int_desc: display interface description
  link_agg: display link-aggregation summary
  vlan: display vlan
  mac: display mac-address
  stp_brief: display stp brief
  stp_interface: display stp interface brief
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
    "device_name": "SW-01",
    "command": "display interface brief",
    "question_number": 1,
}

r = s.post(url, json=body, timeout=30)
r.raise_for_status()
print(r.text)
```

## Notes

- Currently designed for read-only queries (show/display) only.
