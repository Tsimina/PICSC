import scapy.all as scapy
import pandas as pd
from collections import defaultdict
import numpy as np

def extract_flow_features(pcap_file):
    """
    Extrage trăsături de flux dintr-un fișier PCAP și le salvează într-un CSV.
    """
    print(f"[*] Procesare fișier PCAP: {pcap_file}")
    
    # Folosim un defaultdict pentru a stoca informațiile despre fluxuri
    # Cheia va fi 5-tuple-ul, iar valoarea va fi un dicționar cu trăsăturile
    flows = defaultdict(lambda: {
        'packets': 0,
        'bytes': 0,
        'start_time': None,
        'end_time': None,
        'packet_sizes': [],
        'inter_arrival_times': []
    })

    # Citim pachetele din fișierul PCAP
    packets = scapy.rdpcap(pcap_file)

    for packet in packets:
        # Verificăm dacă pachetul are stratul IP și TCP/UDP
        if packet.haslayer(scapy.IP) and (packet.haslayer(scapy.TCP) or packet.haslayer(scapy.UDP)):
            
            # Extragem informațiile de bază
            src_ip = packet[scapy.IP].src
            dst_ip = packet[scapy.IP].dst
            proto = packet[scapy.IP].proto
            
            if proto == 6: # TCP
                src_port = packet[scapy.TCP].sport
                dst_port = packet[scapy.TCP].dport
            elif proto == 17: # UDP
                src_port = packet[scapy.UDP].sport
                dst_port = packet[scapy.UDP].dport
            else:
                continue

            # Creăm o cheie canonică pentru flux (sursă/destinație sunt interschimbabile)
            # Asta asigură că pachetele de la A->B și B->A ajung în același flux
            flow_key_part1 = tuple(sorted(((src_ip, src_port), (dst_ip, dst_port))))
            flow_key = flow_key_part1 + (proto,)
            
            # Timestamp-ul pachetului
            timestamp = float(packet.time)
            packet_size = len(packet)

            # Actualizăm trăsăturile pentru acest flux
            flow = flows[flow_key]
            
            if flow['start_time'] is None:
                flow['start_time'] = timestamp
            
            # Calculăm timpul dintre pachete (inter-arrival time)
            if flow['end_time'] is not None:
                inter_arrival_time = timestamp - flow['end_time']
                flow['inter_arrival_times'].append(inter_arrival_time)

            flow['end_time'] = timestamp
            flow['packets'] += 1
            flow['bytes'] += packet_size
            flow['packet_sizes'].append(packet_size)

    print(f"[*] Am identificat {len(flows)} fluxuri unice.")

    # Procesăm fluxurile pentru a calcula trăsăturile finale
    feature_list = []
    for key, data in flows.items():
        # Extragem cheia pentru a o adăuga în CSV
        (ip_tuple1, ip_tuple2, proto) = key
        (ip1, port1) = ip_tuple1
        (ip2, port2) = ip_tuple2
        
        # Calculăm durata fluxului
        duration = data['end_time'] - data['start_time'] if data['packets'] > 1 else 0

        # Calculăm statisticile pentru dimensiunea pachetelor
        avg_pkt_size = np.mean(data['packet_sizes']) if data['packet_sizes'] else 0
        std_pkt_size = np.std(data['packet_sizes']) if len(data['packet_sizes']) > 1 else 0
        min_pkt_size = np.min(data['packet_sizes']) if data['packet_sizes'] else 0
        max_pkt_size = np.max(data['packet_sizes']) if data['packet_sizes'] else 0

        # Calculăm statisticile pentru timpii dintre pachete (IAT)
        avg_iat = np.mean(data['inter_arrival_times']) if data['inter_arrival_times'] else 0
        std_iat = np.std(data['inter_arrival_times']) if len(data['inter_arrival_times']) > 1 else 0
        min_iat = np.min(data['inter_arrival_times']) if data['inter_arrival_times'] else 0
        max_iat = np.max(data['inter_arrival_times']) if data['inter_arrival_times'] else 0
        
        feature_list.append({
            'src_ip': ip1,
            'src_port': port1,
            'dst_ip': ip2,
            'dst_port': port2,
            'protocol': proto,
            'flow_duration_sec': duration,
            'total_packets': data['packets'],
            'total_bytes': data['bytes'],
            'avg_pkt_size': avg_pkt_size,
            'std_pkt_size': std_pkt_size,
            'min_pkt_size': min_pkt_size,
            'max_pkt_size': max_pkt_size,
            'avg_inter_arrival_time': avg_iat,
            'std_inter_arrival_time': std_iat,
            'min_inter_arrival_time': min_iat,
            'max_inter_arrival_time': max_iat
        })

    # Creăm un DataFrame pandas și îl salvăm ca CSV
    df = pd.DataFrame(feature_list)
    output_csv = "youtube_flow_features.csv"
    df.to_csv(output_csv, index=False)
    print(f"[*] Trăsăturile au fost salvate cu succes în fișierul: {output_csv}")

# --- Execuția scriptului ---
if __name__ == "__main__":
    pcap_file_path = "youtube2.pcap" 
    extract_flow_features(pcap_file_path)
    