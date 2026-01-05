# src/feature_extractor.py

import numpy as np
from scipy.stats import kurtosis, skew
import hashlib
from typing import Dict, Any, Iterator, Optional
from flow_aggregator import CompleteFlow

ExtractedFeatures = Dict[str, Any]

def _safe_stat(arr: np.ndarray, fn, default: float = 0.0) -> float:
    try:
        if arr.size == 0:
            return float(default)
        return float(fn(arr))
    except Exception:
        return float(default)

def calculate_flow_features(flow: CompleteFlow, label: int) -> ExtractedFeatures:
    """
    Calculează feature-uri la nivel de flux pentru clasificare binară:
      Label = 1 (Spotify), 0 (Restul)
    """
    packets = flow.get("packets", [])
    if len(packets) == 0:
        return {}

    init_ip, init_port = flow["init_dir"]

    timestamps = np.array([p["timestamp"] for p in packets], dtype=float)
    pkt_lengths = np.array([p.get("pkt_len", 0) for p in packets], dtype=float)

    # Duration & IAT
    if timestamps.size > 1:
        duration = float(timestamps[-1] - timestamps[0])
        iat = np.diff(timestamps)
    else:
        duration = 0.0
        iat = np.array([], dtype=float)

    # Direcționalitate (A->B = direcția inițială)
    bytes_ab = 0
    bytes_ba = 0
    pkts_ab = 0
    pkts_ba = 0

    for p in packets:
        is_ab = (p["ip_src"] == init_ip) and (int(p["sport"]) == int(init_port))
        if is_ab:
            bytes_ab += int(p.get("pkt_len", 0))
            pkts_ab += 1
        else:
            bytes_ba += int(p.get("pkt_len", 0))
            pkts_ba += 1

    total_packets = int(len(packets))
 
    total_bytes = int(bytes_ab + bytes_ba)

    # Rate-uri (evită div 0)
    pkt_rate = float(total_packets / duration) if duration > 0 else 0.0
    byte_rate = float(total_bytes / duration) if duration > 0 else 0.0

    # Porturi/protocol din cheie (ip1,port1,ip2,port2,proto)
    fk = flow["flow_key"]
    port1 = int(fk[1])
    port2 = int(fk[3])
    proto = int(fk[4])
    min_port = int(min(port1, port2))
    max_port = int(max(port1, port2))

    # Hash Flow_ID numeric
    flow_id_string = "_".join(flow["flow_key"])
    flow_id_hash = int(hashlib.sha1(flow_id_string.encode("utf-8")).hexdigest(), 16) % (10**16)

    features: ExtractedFeatures = {
        "Flow_ID": int(flow_id_hash),

        # Identificatori utili (numerici)
        "Proto": proto,
        "Min_Port": min_port,
        "Max_Port": max_port,

        # Volum / durată
        "Duration_s": float(duration),
        "Total_Packets": total_packets,
        "Total_Bytes": total_bytes,

        # Direcționalitate
        "Packets_AB": int(pkts_ab),
        "Packets_BA": int(pkts_ba),
        "Bytes_AB": int(bytes_ab),
        "Bytes_BA": int(bytes_ba),
        "Direction_Ratio": float(pkts_ab / total_packets) if total_packets > 0 else 0.0,

        # Rate-uri
        "Packet_Rate_pps": float(pkt_rate),
        "Byte_Rate_Bps": float(byte_rate),

        # Statistici pachete
        "Pkt_Len_Min": _safe_stat(pkt_lengths, np.min),
        "Pkt_Len_Max": _safe_stat(pkt_lengths, np.max),
        "Pkt_Len_Mean": _safe_stat(pkt_lengths, np.mean),
        "Pkt_Len_Std": _safe_stat(pkt_lengths, np.std),
        "Pkt_Len_Skew": float(skew(pkt_lengths)) if pkt_lengths.size >= 3 else 0.0,
        "Pkt_Len_Kurtosis": float(kurtosis(pkt_lengths, fisher=True)) if pkt_lengths.size >= 4 else 0.0,

        # Statistici IAT
        "IAT_Min": _safe_stat(iat, np.min),
        "IAT_Max": _safe_stat(iat, np.max),
        "IAT_Mean": _safe_stat(iat, np.mean),
        "IAT_Std": _safe_stat(iat, np.std),
        "IAT_Skew": float(skew(iat)) if iat.size >= 3 else 0.0,
        "IAT_Kurtosis": float(kurtosis(iat, fisher=True)) if iat.size >= 4 else 0.0,

        # Etichetă
        "Label": int(label),
    }

    return features


def extract_all_features(flows_generator: Iterator[CompleteFlow], label: int) -> Iterator[ExtractedFeatures]:
    """
    Generator care aplică extractarea pentru fiecare flux complet.
    """
    for flow in flows_generator:
        feats = calculate_flow_features(flow, label=label)
        if feats:
            yield feats
