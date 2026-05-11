---
name: l3_route
description: Layer 3 forwarding and routing protocol O&M (IP/IPv6/ARP/ND/routing table/OSPF/BGP), executing show/display commands via the local NOC API.
metadata:
  openclaw:
    emoji: "🧭"
---

# l3_route — Layer 3 Forwarding & Routing Protocols

Covers IPv4/IPv6 interface information, ARP/ND neighbor tables, routing tables, OSPF, BGP EVPN/VPNv4, and other dynamic routing protocol O&M.

## Parameters

- `device_name` (required): Device name
- `vendor` (required): `Huawei` | `Cisco` | `H3C`
- `question_number` (required): Number representing the current question/problem ID being solved
- `action` (required):
  - `ip_int`: IPv4 interface brief information
  - `ipv6_int`: IPv6 interface information (optional `interface_id`)
  - `arp`: ARP table
  - `ipv6_nd`: IPv6 neighbors
  - `route_v4`: IPv4 routing table
  - `vpn_instance`: VPN instance table
  - `vpn_route`: VPN instance routing table (requires `vpn_name`)
  - `route_v6`: IPv6 routing table
  - `ospf_peer`: OSPF neighbors
  - `ospf_route`: OSPF routes
  - `ospf_lsdb`: OSPF LSDB
  - `bgp_evpn`: BGP EVPN routes
  - `bgp_vpnv4`: BGP VPNv4 routes
- `interface_id` (optional): Interface ID
- `vpn_name` (optional): VPN instance name

## Execution Method (Local NOC API)

- Endpoint: `http://127.0.0.1:5000/api/agent/execute`
- Method: `POST`
- Body: `{ "device_name": "...", "command": "...", "question_number": 1 }`

### Command Mapping

```text
Huawei:
  ip_int: display ip interface brief
  ipv6_int: display ipv6 interface [<interface_id>]
  arp: display arp
  ipv6_nd: display ipv6 neighbors
  route_v4: display ip routing-table
  vpn_instance: display ip vpn-instance
  vpn_route: display ip routing-table vpn-instance <vpn_name>
  route_v6: display ipv6 routing-table
  ospf_peer: display ospf peer
  ospf_route: display ospf routing
  ospf_lsdb: display ospf lsdb
  bgp_evpn: display bgp evpn all routing-table
  bgp_vpnv4: display bgp vpnv4 all routing-table

Cisco:
  ip_int: show ip interface brief
  ipv6_int: show ipv6 interface [<interface_id>]
  arp: show ip arp
  ipv6_nd: show ipv6 neighbors
  route_v4: show ip route
  vpn_instance: show ip vrf
  vpn_route: show ip route vrf <vpn_name>
  route_v6: show ipv6 route
  ospf_peer: show ip ospf neighbor
  ospf_route: show ip route ospf
  ospf_lsdb: show ip ospf database
  bgp_evpn: show bgp l2vpn evpn
  bgp_vpnv4: show bgp vpnv4 unicast all

H3C:
  ip_int: display ip interface brief
  ipv6_int: display ipv6 interface [<interface_id>]
  arp: display arp all
  ipv6_nd: display ipv6 neighbor
  route_v4: display ip routing-table
  vpn_instance: display ip vpn-instance
  vpn_route: display ip routing-table vpn-instance <vpn_name>
  route_v6: display ipv6 routing-table
  ospf_peer: display ospf peer
  ospf_route: display ospf routing
  ospf_lsdb: display ospf lsdb
  bgp_evpn: display bgp l2vpn evpn
  bgp_vpnv4: display bgp vpnv4 all routing-table
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
    "command": "display ospf peer",
    "question_number": 1,
}

r = s.post(url, json=body, timeout=30)
r.raise_for_status()
print(r.text)
```

## Notes

- Currently designed for read-only queries (show/display) only.
