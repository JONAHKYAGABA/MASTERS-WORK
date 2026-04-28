---
name: adv_tunnel
description: Data center / carrier network advanced O&M (VXLAN/VRRP/BFD/DHCP/SRv6), executing show/display commands via the local NOC API.
metadata:
  openclaw:
    emoji: "🧵"
---

# adv_tunnel — Advanced Technologies & Tunnel Protocols

Deep O&M for data center and next-generation carrier network scenarios: VXLAN tunnels, VRRP, BFD sessions, DHCP address pools, SRv6 Policy/SID forwarding information, etc.

## Required Parameters

- `device_name` (required): Device name
- `vendor` (required): Vendor: `Huawei` | `Cisco` | `H3C`
- `question_number` (required): Number representing the current question/problem ID being solved
- `action` (required):
  - `vxlan_tunnel`: VXLAN tunnel status
  - `vxlan_ts`: VXLAN troubleshooting
  - `vrrp`: VRRP detailed status
  - `bfd`: BFD session status
  - `dhcp`: DHCP address pool status
  - `srv6_policy_status`: SRv6 Policy status
  - `srv6_policy_detail`: SRv6 Policy details
  - `srv6_end`: SRv6 End forwarding information
  - `srv6_end_x`: SRv6 End.X forwarding information

## Execution Method (Local NOC API)

This skill executes read-only commands by calling the local HTTP endpoint:

- Endpoint: `http://127.0.0.1:5000/api/agent/execute`
- Method: `POST`
- Body: `{ "device_name": "...", "command": "...", "question_number": 1 }`

### Command Mapping Table

```text
Huawei:
  vxlan_tunnel: display vxlan tunnel
  vxlan_ts: display vxlan troubleshooting
  vrrp: display vrrp verbose
  bfd: display bfd session all
  dhcp: display ip pool
  srv6_policy_status: display srv6-te policy status
  srv6_policy_detail: display srv6-te policy
  srv6_end: display segment-routing ipv6 local-sid end forwarding
  srv6_end_x: display segment-routing ipv6 local-sid end-x forwarding

Cisco:
  vxlan_tunnel: show nve vni
  vxlan_ts: show nve
  vrrp: show vrrp detail
  bfd: show bfd neighbors
  dhcp: show ip dhcp pool
  srv6_policy_status: show segment-routing srv6 policy
  srv6_policy_detail: show segment-routing srv6 policy
  srv6_end: show segment-routing srv6 sid
  srv6_end_x: show segment-routing srv6 sid

H3C:
  vxlan_tunnel: display vxlan tunnel
  vxlan_ts: display vxlan troubleshooting
  vrrp: display vrrp verbose
  bfd: display bfd session all
  dhcp: display ip pool
  srv6_policy_status: display segment-routing ipv6 te policy
  srv6_policy_detail: display segment-routing ipv6 te policy
  srv6_end: display segment-routing ipv6 local-sid
  srv6_end_x: display segment-routing ipv6 local-sid
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
    "command": "display vrrp verbose",
    "question_number": 1,
}

r = s.post(url, json=body, timeout=30)
r.raise_for_status()
print(r.text)
```

## Notes

- By default, this is only used for **show/display** read-only queries; if configuration change commands are needed, a separate approval/whitelist mechanism is recommended.
