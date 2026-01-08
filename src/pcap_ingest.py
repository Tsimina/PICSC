from scapy.all import IP, TCP, UDP
from scapy.utils import PcapReader
from typing import Iterator, Dict, Any, Tuple

# Data types
FlowKey = Tuple[str, ...]
ParsedPacket = Dict[str, Any]
NormalizedPacket = Tuple[FlowKey, ParsedPacket]

# Generator function to read and parse packets from a PCAP/PCAPNG file
def pcap_reader_generator(filepath: str) -> Iterator[ParsedPacket]:
    try:
        reader = PcapReader(filepath)
    except FileNotFoundError:
        print(f"Error: No pcap file was found in {filepath}")
        return
    except Exception as e:
        print(f"Error when downloading: {e}")
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


# Function to normalize flow key
def normalize_flow_key(packet: ParsedPacket) -> FlowKey:
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
