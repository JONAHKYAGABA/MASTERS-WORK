# **TOOLS.md \- Available Tools List**

## **Sole Source of Tools**

All tools originate from the skills/ directory and are invoked via the local NOC API. **The use of any tools outside of skills is prohibited.**

## **NOC API Calling Conventions**

* Endpoint: http://127.0.0.1:5000/api/agent/execute  
* Method: POST  
* Body: { "device\_name": "...", "command": "...", "question\_number": N }  
* Proxies must be disabled: s.trust\_env \= False

## **Skill Quick Reference**

### **🛠️ infra\_maintenance — Infrastructure Maintenance**

| action | Huawei | Cisco | H3C |
| :---- | :---- | :---- | :---- |
| config | display current-configuration | show running-config | display current-configuration |
| log | display logbuffer | show logging | display logbuffer |
| alarm | display alarm active | show facility-alarm status | display alarm active |
| memory | display memory-usage | show processes memory | display memory |
| lldp | display lldp neighbor brief | show lldp neighbors | display lldp neighbor-list |

### **🔌 l2\_link — Layer 2 Link**

| action | Huawei | Cisco | H3C |
| :---- | :---- | :---- | :---- |
| int\_brief | display interface brief | show ip int brief | display interface brief |
| int\_desc | display interface description | show interface description | display interface description |
| link\_agg | display eth-trunk | show etherchannel summary | display link-aggregation summary |
| vlan | display vlan | show vlan brief | display vlan |
| mac | display mac-address | show mac address-table | display mac-address |
| stp\_brief | display stp brief | show spanning-tree brief | display stp brief |
| stp\_interface | display stp interface brief | show spanning-tree interface \<id\> | display stp interface brief |

### **🧭 l3\_route — Layer 3 Routing**

| action | Huawei | Cisco | H3C |
| :---- | :---- | :---- | :---- |
| ip\_int | display ip interface brief | show ip interface brief | display ip interface brief |
| ipv6\_int | display ipv6 interface \[\<id\>\] | show ipv6 interface \[\<id\>\] | display ipv6 interface \[\<id\>\] |
| arp | display arp | show ip arp | display arp all |
| ipv6\_nd | display ipv6 neighbors | show ipv6 neighbors | display ipv6 neighbor |
| route\_v4 | display ip routing-table | show ip route | display ip routing-table |
| vpn\_instance | display ip vpn-instance | show ip vrf | display ip vpn-instance |
| vpn\_route | display ip routing-table vpn-instance \<name\> | show ip route vrf \<name\> | display ip routing-table vpn-instance \<name\> |
| route\_v6 | display ipv6 routing-table | show ipv6 route | display ipv6 routing-table |
| ospf\_peer | display ospf peer | show ip ospf neighbor | display ospf peer |
| ospf\_route | display ospf routing | show ip route ospf | display ospf routing |
| ospf\_lsdb | display ospf lsdb | show ip ospf database | display ospf lsdb |
| bgp\_evpn | display bgp evpn all routing-table | show bgp l2vpn evpn | display bgp l2vpn evpn |
| bgp\_vpnv4 | display bgp vpnv4 all routing-table | show bgp vpnv4 unicast all | display bgp vpnv4 all routing-table |

### **🧵 adv\_tunnel — Advanced Tunnels**

| action | Huawei | Cisco | H3C |
| :---- | :---- | :---- | :---- |
| vxlan\_tunnel | display vxlan tunnel | show nve vni | display vxlan tunnel |
| vxlan\_ts | display vxlan troubleshooting | show nve | display vxlan troubleshooting |
| vrrp | display vrrp verbose | show vrrp detail | display vrrp verbose |
| bfd | display bfd session all | show bfd neighbors | display bfd session all |
| dhcp | display ip pool | show ip dhcp pool | display ip pool |
| srv6\_policy\_status | display srv6-te policy status | show segment-routing srv6 policy | display segment-routing ipv6 te policy |
| srv6\_policy\_detail | display srv6-te policy | show segment-routing srv6 policy | display segment-routing ipv6 te policy |
| srv6\_end | display segment-routing ipv6 local-sid end forwarding | show segment-routing srv6 sid | display segment-routing ipv6 local-sid |
| srv6\_end\_x | display segment-routing ipv6 local-sid end-x forwarding | show segment-routing srv6 sid | display segment-routing ipv6 local-sid |

## **Invocation Example**

import os  
import requests

\# Remove proxy environment variables to prevent interference with local API calls  
for key in \["http\_proxy", "https\_proxy", "HTTP\_PROXY", "HTTPS\_PROXY"\]:  
    os.environ.pop(key, None)

s \= requests.Session()  
s.trust\_env \= False

r \= s.post("\[http://127.0.0.1:5000/api/agent/execute\](http://127.0.0.1:5000/api/agent/execute)",  
           json={"device\_name": "Core-Router-01", "command": "display ospf peer", "question\_number": 1},  
           timeout=30)  
r.raise\_for\_status()  
print(r.text)  
