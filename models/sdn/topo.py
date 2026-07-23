#!/usr/bin/env python3
"""
SDN IDS Topology – 20 hosts, 5 OpenFlow switches (spine-leaf)
Requires: Mininet 2.3.0, OVS, remote Ryu controller on 127.0.0.1:6633

Layout:
                        s1  (core)
              ┌─────────┼─────────┬─────────┐
             s2         s3        s4        s5
          h1–h5      h6–h10   h11–h15  h16–h20
        (servers,   (bgnd      (normal   (h18-h20
         normal)    traffic)    users)   attackers)
"""
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink


HOST_CONFIG = [
    # (hostname, ip,            mac,                switch_idx, role)
    # ── s2: server + normal users ──────────────────────────────────
    ('h1',  '10.0.0.1',  '00:00:00:00:00:01',  2, 'server'),
    ('h2',  '10.0.0.2',  '00:00:00:00:00:02',  2, 'normal'),
    ('h3',  '10.0.0.3',  '00:00:00:00:00:03',  2, 'normal'),
    ('h4',  '10.0.0.4',  '00:00:00:00:00:04',  2, 'normal'),
    ('h5',  '10.0.0.5',  '00:00:00:00:00:05',  2, 'normal'),
    # ── s3: background traffic ──────────────────────────────────────
    ('h6',  '10.0.0.6',  '00:00:00:00:00:06',  3, 'background'),
    ('h7',  '10.0.0.7',  '00:00:00:00:00:07',  3, 'background'),
    ('h8',  '10.0.0.8',  '00:00:00:00:00:08',  3, 'background'),
    ('h9',  '10.0.0.9',  '00:00:00:00:00:09',  3, 'background'),
    ('h10', '10.0.0.10', '00:00:00:00:00:0a',  3, 'background'),
    # ── s4: normal users ────────────────────────────────────────────
    ('h11', '10.0.0.11', '00:00:00:00:00:0b',  4, 'normal'),
    ('h12', '10.0.0.12', '00:00:00:00:00:0c',  4, 'normal'),
    ('h13', '10.0.0.13', '00:00:00:00:00:0d',  4, 'normal'),
    ('h14', '10.0.0.14', '00:00:00:00:00:0e',  4, 'normal'),
    ('h15', '10.0.0.15', '00:00:00:00:00:0f',  4, 'normal'),
    # ── s5: mixed + attackers ───────────────────────────────────────
    ('h16', '10.0.0.16', '00:00:00:00:00:10',  5, 'normal'),
    ('h17', '10.0.0.17', '00:00:00:00:00:11',  5, 'normal'),
    ('h18', '10.0.0.18', '00:00:00:00:00:12',  5, 'ATTACKER'),
    ('h19', '10.0.0.19', '00:00:00:00:00:13',  5, 'ATTACKER'),
    ('h20', '10.0.0.20', '00:00:00:00:00:14',  5, 'ATTACKER'),
]


def build():
    net = Mininet(
        controller=RemoteController,
        switch=OVSKernelSwitch,
        link=TCLink,
        autoSetMacs=False,
        autoStaticArp=False
    )

    # ── Controller ───────────────────────────────────────────────────
    info('*** Adding remote controller (127.0.0.1:6633)\n')
    net.addController(
        'c0',
        controller=RemoteController,
        ip='127.0.0.1',
        port=6633
    )

    # ── Switches ─────────────────────────────────────────────────────
    info('*** Adding switches\n')
    switches = {}
    for i in range(1, 6):
        sw = net.addSwitch(f's{i}', protocols='OpenFlow13')
        switches[i] = sw

    # Core-to-edge links (1 Gbps uplinks)
    for edge_idx in range(2, 6):
        net.addLink(switches[1], switches[edge_idx], bw=1000)

    # ── Hosts ────────────────────────────────────────────────────────
    info('*** Adding hosts\n')
    info(f'  {"Host":<6} {"IP":<14} {"Role":<12} {"Switch"}\n')
    info('  ' + '-'*45 + '\n')

    for hostname, ip, mac, sw_idx, role in HOST_CONFIG:
        h = net.addHost(hostname, ip=ip + '/24', mac=mac)
        net.addLink(h, switches[sw_idx], bw=100)
        info(f'  {hostname:<6} {ip:<14} {role:<12} s{sw_idx}\n')

    # ── Start ────────────────────────────────────────────────────────
    info('\n*** Starting network\n')
    net.start()

    # Verify OF1.3 on all switches
    for i in range(1, 6):
        switches[i].cmd('ovs-vsctl set bridge s{0} protocols=OpenFlow13'.format(i))

    # Start HTTP server on h1 (victim web server)
    h1 = net.get('h1')
    h1.cmd('python3 -m http.server 80 &>/tmp/h1_httpd.log &')
    info('\n*** HTTP server started on h1:80\n')

    info('\n*** Roles summary:\n')
    info('  h1          → Web server (primary victim)\n')
    info('  h2-h5       → Normal users (s2)\n')
    info('  h6-h10      → Background traffic (s3)\n')
    info('  h11-h15     → Normal users (s4)\n')
    info('  h16-h17     → Normal users (s5)\n')
    info('  h18-h20     → ATTACKERS   (s5)\n')
    info('\n*** Starting CLI (type "exit" to stop)\n')

    CLI(net)

    info('*** Stopping network\n')
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    build()
