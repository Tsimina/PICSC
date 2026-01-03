from scapy.all import rdpcap, IP, TCP, UDP

# Se citesc pachetele din fisierul .pcap
pachete = rdpcap('youtube2.pcap')
print(f"Au fost citite {len(pachete)} pachete.")

# Se parcurge lista de pachete si se afiseaza detalii
for i, pachet in enumerate(pachete):
    if IP in pachet:
        ip_sursa = pachet[IP].src
        ip_destinatie = pachet[IP].dst
        
        if TCP in pachet:
            protocol = "TCP"
            port_sursa = pachet[TCP].sport
            port_destinatie = pachet[TCP].dport
            print(f"Pachet {i+1}: Sursa: {ip_sursa}:{port_sursa} -> Destinatie: {ip_destinatie}:{port_destinatie} | Protocol: {protocol}")
        
        elif UDP in pachet:
            protocol = "UDP"
            port_sursa = pachet[UDP].sport
            port_destinatie = pachet[UDP].dport
            print(f"Pachet {i+1}: Sursa: {ip_sursa}:{port_sursa} -> Destinatie: {ip_destinatie}:{port_destinatie} | Protocol: {protocol}")
            