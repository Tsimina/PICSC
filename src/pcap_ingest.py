from scapy.all import IP, TCP, UDP
from scapy.utils import PcapReader
from typing import Iterator, Dict, Any, Tuple

FlowKey = Tuple[str, ...]
ParsedPacket = Dict[str, Any]
NormalizedPacket = Tuple[FlowKey, ParsedPacket]

def pcap_reader_generator(filepath: str) -> Iterator[ParsedPacket]:
    """
    Generator streaming: citește pachetele din PCAP/PCAPNG fără să încarce tot fișierul în RAM.
    """
    try:
        reader = PcapReader(filepath)
    except FileNotFoundError:
        print(f"Eroare: Fișierul PCAP nu a fost găsit la {filepath}")
        return
    except Exception as e:
        print(f"Eroare la deschiderea PCAP/PCAPNG: {e}")
        return

    try:
        for pkt in reader:
            if IP not in pkt:
                continue

            packet_info: ParsedPacket = {
                "timestamp": float(pkt.time),
                "ip_src": pkt[IP].src,
                "ip_dst": pkt[IP].dst,
                "proto": int(pkt[IP].proto),
                "pkt_len": int(len(pkt)),
            }

            if pkt.haslayer(TCP):
                packet_info["sport"] = int(pkt[TCP].sport)
                packet_info["dport"] = int(pkt[TCP].dport)
            elif pkt.haslayer(UDP):
                packet_info["sport"] = int(pkt[UDP].sport)
                packet_info["dport"] = int(pkt[UDP].dport)
            else:
                continue

            yield packet_info
    finally:
        reader.close()



def normalize_flow_key(packet: ParsedPacket) -> FlowKey:
    """
    Generează o cheie canonică bidirecțională de flux (5-tuple):
    (ip_low, port_low, ip_high, port_high, proto)

    Cheia este stabilă indiferent de direcția inițială a pachetului.
    """
    src = (packet["ip_src"], int(packet["sport"]))
    dst = (packet["ip_dst"], int(packet["dport"]))

    # Canonical: comparație lexicografică (IP, port)
    if src <= dst:
        ip1, p1 = src
        ip2, p2 = dst
    else:
        ip1, p1 = dst
        ip2, p2 = src

    return (str(ip1), str(p1), str(ip2), str(p2), str(packet["proto"]))
