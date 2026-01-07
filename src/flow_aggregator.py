# src/flow_aggregator.py

from typing import Iterator, Dict, Any, List, Tuple, Optional
from pcap_ingest import FlowKey, ParsedPacket, NormalizedPacket

# Timeout-ul standard ISCX pentru inactivitate (10 minute)
FLOW_TIMEOUT_SECONDS = 600.0

# Tipul de date pentru a reprezenta un flux complet
CompleteFlow = Dict[str, Any]

def aggregate_flows(
    normalized_packets: Iterator[NormalizedPacket],
    timeout: float = FLOW_TIMEOUT_SECONDS,
    include_packets: bool = True,
    max_active_duration: Optional[float] = None,
) -> Iterator[CompleteFlow]:
    """
    Generator care reconstruiește fluxurile complete dintr-un stream de pachete.

    Comportament:
    - `timeout` este timpul de inactivitate (în secunde) care determină închiderea
      fluxurilor (comportament ISCX, implicit 600s).
    - `max_active_duration` (opțional) segmentează un flux activ în ferestre de
      durată fixă (de ex. 15s). Aceasta este diferită de timeout-ul de inactivitate
      și determină flush-ul unui segment activ când durata segmentului depășește
      valoarea specificată; ulterior se pornește un nou segment pentru același
      `flow_key`.

    `include_packets=True` păstrează lista completă de pachete în output (utilă pt feature-uri).
    """

    active_flows: Dict[FlowKey, List[ParsedPacket]] = {}
    flow_init_direction: Dict[FlowKey, Tuple[str, int]] = {}

    for flow_key, packet in normalized_packets:
        current_time = float(packet["timestamp"])

        # Închide fluxurile inactive (prin timeout)
        keys_to_close: List[FlowKey] = []
        for key, pkts in active_flows.items():
            if pkts and (current_time - float(pkts[-1]["timestamp"])) > timeout:
                keys_to_close.append(key)

        for key in keys_to_close:
            pkts = active_flows.pop(key)
            init_dir = flow_init_direction.pop(key)

            start_ts = float(pkts[0]["timestamp"])
            end_ts = float(pkts[-1]["timestamp"])
            duration = float(end_ts - start_ts)
            packet_count = int(len(pkts))
            total_bytes = int(sum(int(p.get("pkt_len", 0)) for p in pkts))

            flow_summary: CompleteFlow = {
                "flow_key": key,
                "start_time": start_ts,
                "end_time": end_ts,
                "duration": duration,
                "packet_count": packet_count,
                "total_bytes": total_bytes,
                "init_dir": init_dir,
            }

            if include_packets:
                flow_summary["packets"] = pkts

            yield flow_summary

        # Segmentare pe durată activă (dacă este setată)
        if flow_key in active_flows and max_active_duration is not None:
            pkts = active_flows[flow_key]
            if pkts:
                start_ts = float(pkts[0]["timestamp"])
                # Dacă segmentul curent a depășit durata activă, flush segment și începe unul nou
                if (current_time - start_ts) > float(max_active_duration):
                    # Flush current segment
                    popped_pkts = active_flows.pop(flow_key)
                    init_dir = flow_init_direction.pop(flow_key)

                    seg_start = float(popped_pkts[0]["timestamp"]) if popped_pkts else current_time
                    seg_end = float(popped_pkts[-1]["timestamp"]) if popped_pkts else current_time
                    seg_duration = float(seg_end - seg_start)
                    seg_packet_count = int(len(popped_pkts))
                    seg_total_bytes = int(sum(int(p.get("pkt_len", 0)) for p in popped_pkts))

                    flow_summary: CompleteFlow = {
                        "flow_key": flow_key,
                        "start_time": seg_start,
                        "end_time": seg_end,
                        "duration": seg_duration,
                        "packet_count": seg_packet_count,
                        "total_bytes": seg_total_bytes,
                        "init_dir": init_dir,
                    }

                    if include_packets:
                        flow_summary["packets"] = popped_pkts

                    yield flow_summary

        # Dacă nu există un flux activ pentru cheia curentă, inițializează unul nou
        if flow_key not in active_flows:
            active_flows[flow_key] = []
            flow_init_direction[flow_key] = (packet["ip_src"], int(packet["sport"]))

        active_flows[flow_key].append(packet)

    # Flush final
    for flow_key in list(active_flows.keys()):
        pkts = active_flows.pop(flow_key)
        init_dir = flow_init_direction.pop(flow_key)

        start_ts = float(pkts[0]["timestamp"])
        end_ts = float(pkts[-1]["timestamp"])
        duration = float(end_ts - start_ts)
        packet_count = int(len(pkts))
        total_bytes = int(sum(int(p.get("pkt_len", 0)) for p in pkts))

        flow_summary: CompleteFlow = {
            "flow_key": flow_key,
            "start_time": start_ts,
            "end_time": end_ts,
            "duration": duration,
            "packet_count": packet_count,
            "total_bytes": total_bytes,
            "init_dir": init_dir,
        }

        if include_packets:
            flow_summary["packets"] = pkts

        yield flow_summary
