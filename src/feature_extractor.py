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

    bytes_ab = 0
    bytes_ba = 0
    packets_ab = 0
    for p in packets:
        sport = p.get('sport', p.get('port'))
        if p.get('ip_src') == init_ip and sport == init_port:
            bytes_ab += int(p.get('pkt_len', 0))
            packets_ab += 1
        else:
            bytes_ba += int(p.get('pkt_len', 0))

    total_packets = int(len(packets))
    direction_ratio = float(packets_ab / total_packets) if total_packets > 0 else 0.0

    mean_pkt_len = float(np.mean(pkt_lengths)) if pkt_lengths.size > 0 else 0.0
    skewness_pkt_len = float(skew(pkt_lengths)) if pkt_lengths.size >= 3 else 0.0

    std_iat = float(np.std(iat_sequence)) if iat_sequence.size > 0 else 0.0
    kurtosis_iat = float(kurtosis(iat_sequence, fisher=True)) if iat_sequence.size >= 4 else 0.0

    flow_id_string = "_".join(flow['flow_key'])
    flow_id_hash = int(hashlib.sha1(flow_id_string.encode('utf-8')).hexdigest(), 16) % (10**16)

    features: ExtractedFeatures = {
        'Flow_ID': int(flow_id_hash),
        'Duration_s': float(duration),
        'Total_Packets': int(total_packets),
        'Bytes_AB': int(bytes_ab),
        'Bytes_BA': int(bytes_ba),
        'Mean_Pkt_Len': mean_pkt_len,
        'STD_IAT': std_iat,
        'Kurtosis_IAT': kurtosis_iat,
        'Skewness_Pkt_Len': skewness_pkt_len,
        'Direction_Ratio': float(direction_ratio),
        'Label': int(label),
    }

    return features

# Generator function to extract features from all flows
def extract_all_features(flows_generator: Iterator[CompleteFlow], label: int) -> Iterator[ExtractedFeatures]:
    for flow in flows_generator:
        feats = calculate_statistical_features(flow, label=label)
        if feats:
            yield feats
