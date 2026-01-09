# src/feature_extractor.py

import numpy as np
from scipy.stats import kurtosis, skew
import hashlib
from typing import Dict, Any, Iterator
from flow_aggregator import CompleteFlow

# Data type for extracted features
ExtractedFeatures = Dict[str, Any]

# Function to calculate statistical features for a given flow
def calculate_statistical_features(flow: CompleteFlow, label: int) -> ExtractedFeatures:
    packets = flow.get('packets', [])
    if not packets:
        return {}

    init_ip, init_port = flow['init_dir']

    timestamps = np.array([p['timestamp'] for p in packets], dtype=float)
    pkt_lengths = np.array([int(p.get('pkt_len', 0)) for p in packets], dtype=float)

    if timestamps.size > 1:
        duration = float(timestamps[-1] - timestamps[0])
        iat_sequence = np.diff(timestamps)
    else:
        duration = 0.0
        iat_sequence = np.array([], dtype=float)

    # --- Directional split (A->B = forward; B->A = backward) ---
    fwd_ts = []
    bwd_ts = []
    bytes_ab = 0
    bytes_ba = 0
    packets_ab = 0

    for p in packets:
        sport = p.get('sport', p.get('port'))
        pkt_len = int(p.get('pkt_len', 0))
        ts = float(p['timestamp'])

        if p.get('ip_src') == init_ip and sport == init_port:
            bytes_ab += pkt_len
            packets_ab += 1
            fwd_ts.append(ts)
        else:
            bytes_ba += pkt_len
            bwd_ts.append(ts)

    total_packets = int(len(packets))
    total_bytes = int(bytes_ab + bytes_ba)
    direction_ratio = float(packets_ab / total_packets) if total_packets > 0 else 0.0

    fwd_ts = np.array(fwd_ts, dtype=float)
    bwd_ts = np.array(bwd_ts, dtype=float)

    # Ensure sorted (they usually are, but flows can be merged/unsorted)
    if fwd_ts.size > 1:
        fwd_ts.sort()
        fiat_seq = np.diff(fwd_ts)
    else:
        fiat_seq = np.array([], dtype=float)

    if bwd_ts.size > 1:
        bwd_ts.sort()
        biat_seq = np.diff(bwd_ts)
    else:
        biat_seq = np.array([], dtype=float)

    # --- Packet length stats (extras) ---
    mean_pkt_len = float(np.mean(pkt_lengths)) if pkt_lengths.size > 0 else 0.0
    skewness_pkt_len = float(skew(pkt_lengths)) if pkt_lengths.size >= 3 else 0.0

    # --- Flow IAT stats (ARFF expects min/max/mean/std) ---
    min_flowiat = float(np.min(iat_sequence)) if iat_sequence.size > 0 else 0.0
    max_flowiat = float(np.max(iat_sequence)) if iat_sequence.size > 0 else 0.0
    mean_flowiat = float(np.mean(iat_sequence)) if iat_sequence.size > 0 else 0.0
    std_flowiat = float(np.std(iat_sequence)) if iat_sequence.size > 0 else 0.0

    # Keep original fields for backward compatibility
    std_iat = std_flowiat
    kurtosis_iat = float(kurtosis(iat_sequence, fisher=True)) if iat_sequence.size >= 4 else 0.0

    # --- Forward/backward IAT stats (ARFF expects total/min/max/mean) ---
    total_fiat = float(np.sum(fiat_seq)) if fiat_seq.size > 0 else 0.0
    min_fiat = float(np.min(fiat_seq)) if fiat_seq.size > 0 else 0.0
    max_fiat = float(np.max(fiat_seq)) if fiat_seq.size > 0 else 0.0
    mean_fiat = float(np.mean(fiat_seq)) if fiat_seq.size > 0 else 0.0

    total_biat = float(np.sum(biat_seq)) if biat_seq.size > 0 else 0.0
    min_biat = float(np.min(biat_seq)) if biat_seq.size > 0 else 0.0
    max_biat = float(np.max(biat_seq)) if biat_seq.size > 0 else 0.0
    mean_biat = float(np.mean(biat_seq)) if biat_seq.size > 0 else 0.0

    # --- Rates (ARFF expects pkts/s and bytes/s) ---
    flowPktsPerSecond = float(total_packets / duration) if duration > 0 else 0.0
    flowBytesPerSecond = float(total_bytes / duration) if duration > 0 else 0.0

    # --- Active / Idle stats (ARFF expects min/mean/max/std) ---
    # Common CICFlowMeter-style threshold (seconds)
    ACTIVE_IDLE_THRESH = 2.0

    active_durations = []
    idle_durations = []

    if iat_sequence.size > 0:
        current_active = 0.0
        for gap in iat_sequence:
            g = float(gap)
            if g <= ACTIVE_IDLE_THRESH:
                current_active += g
            else:
                # end of an active segment
                active_durations.append(current_active)
                idle_durations.append(g)
                current_active = 0.0
        active_durations.append(current_active)

    active_arr = np.array(active_durations, dtype=float) if active_durations else np.array([], dtype=float)
    idle_arr = np.array(idle_durations, dtype=float) if idle_durations else np.array([], dtype=float)

    min_active = float(np.min(active_arr)) if active_arr.size > 0 else 0.0
    mean_active = float(np.mean(active_arr)) if active_arr.size > 0 else 0.0
    max_active = float(np.max(active_arr)) if active_arr.size > 0 else 0.0
    std_active = float(np.std(active_arr)) if active_arr.size > 0 else 0.0

    min_idle = float(np.min(idle_arr)) if idle_arr.size > 0 else 0.0
    mean_idle = float(np.mean(idle_arr)) if idle_arr.size > 0 else 0.0
    max_idle = float(np.max(idle_arr)) if idle_arr.size > 0 else 0.0
    std_idle = float(np.std(idle_arr)) if idle_arr.size > 0 else 0.0

    # --- Stable Flow ID ---
    flow_id_string = "_".join(flow['flow_key'])
    flow_id_hash = int(hashlib.sha1(flow_id_string.encode('utf-8')).hexdigest(), 16) % (10**16)

    features: ExtractedFeatures = {
        # =========================
        # REQUIRED / IDENTIFIERS
        # =========================
        'Flow_ID': int(flow_id_hash),

        # =========================
        # ORIGINAL FEATURES (yours)
        # Comment out any line you don't want in the output CSV.
        # =========================
        # 'Duration_s': float(duration),
        # 'Total_Packets': int(total_packets),
        # 'Bytes_AB': int(bytes_ab),
        # 'Bytes_BA': int(bytes_ba),
        # 'Mean_Pkt_Len': float(mean_pkt_len),
        # 'STD_IAT': float(std_iat),
        # 'Kurtosis_IAT': float(kurtosis_iat),
        # 'Skewness_Pkt_Len': float(skewness_pkt_len),
        # 'Direction_Ratio': float(direction_ratio),

        # Label (keep at least one of these)
        # 'Label': int(label),
        # 'class1': str(label),  # optional: swap to string labels if you map them elsewhere

        # =========================
        # ARFF-ALIGNED FEATURES
        # Comment out any you don't need.
        # =========================
        'duration': float(duration),

        # ---- Forward IAT (fiat) ----
        'total_fiat': float(total_fiat),
        'min_fiat': float(min_fiat),
        'mean_fiat': float(mean_fiat),
        'max_fiat': float(max_fiat),

        # ---- Backward IAT (biat) ----
        'total_biat': float(total_biat),
        'min_biat': float(min_biat),
        'mean_biat': float(mean_biat),
        'max_biat': float(max_biat),

        # ---- Flow rates ----
        'flowPktsPerSecond': float(flowPktsPerSecond),
        'flowBytesPerSecond': float(flowBytesPerSecond),

        # ---- Flow IAT (flowiat) ----
        'min_flowiat': float(min_flowiat),
        'mean_flowiat': float(mean_flowiat),
        'max_flowiat': float(max_flowiat),
        'std_flowiat': float(std_flowiat),

        # ---- Active / Idle ----
        'min_active': float(min_active),
        'mean_active': float(mean_active),
        'max_active': float(max_active),
        'std_active': float(std_active),

        'min_idle': float(min_idle),
        'mean_idle': float(mean_idle),
        'max_idle': float(max_idle),
        'std_idle': float(std_idle),

        'Label': int(label),
    }

    return features

# Generator function to extract features from all flows
def extract_all_features(flows_generator: Iterator[CompleteFlow], label: int) -> Iterator[ExtractedFeatures]:
    for flow in flows_generator:
        feats = calculate_statistical_features(flow, label=label)
        if feats:
            yield feats
