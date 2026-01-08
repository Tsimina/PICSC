#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import time
import random
import pandas as pd

from pcap_ingest import pcap_reader_generator, normalize_flow_key
from flow_aggregator import aggregate_flows
from feature_extractor import extract_all_features


def normalized_packets_from_file(pcap_path: Path):
    for pkt in pcap_reader_generator(str(pcap_path)):
        yield normalize_flow_key(pkt), pkt


def infer_label_from_name(name: str) -> int:
    # Spotify = 1, Rest = 0
    return 1 if "spotify" in name.lower() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap_dir", type=str, required=True, help="Pcaps input directory")
    ap.add_argument("--out", type=str, required=True, help="Dataset output CSV file path")
    ap.add_argument("--timeout", type=float, default=600.0, help="Idle flow timeout in seconds (default: 600s)")
    ap.add_argument("--active_duration", "--flow_duration", dest="active_duration", type=float, default= None,
                    help="Segment active flows into fixed-duration windows in seconds (e.g., 15).")

    # Balancing
    ap.add_argument("--nonspotify_ratio", type=float, default=1.0,
                    help="If set to 0, keeps all non-Spotify samples.")
    ap.add_argument("--max_nonspotify_per_pcap", type=int, default=0,
                    help="Limit number of non-Spotify flows extracted per PCAP file (default: 0 = no limit).")
    args = ap.parse_args()

    pcap_dir = Path(args.pcap_dir)

    # Recursive .pcap/.pcapng files
    files = sorted(list(pcap_dir.rglob("*.pcap")) + list(pcap_dir.rglob("*.pcapng")))

    if not files:
        raise SystemExit(f"Nu am găsit fișiere .pcap/.pcapng în: {pcap_dir}")

    spotify_rows = []
    nonspotify_rows = []

    for f in files:
        label = infer_label_from_name(f.name)
        print(f"[INFO] Processing {f.name} (label={label})")
        t0 = time.time()

        flows = aggregate_flows(
            normalized_packets_from_file(f),
            timeout=args.timeout,
            include_packets=True,
            max_active_duration=args.active_duration
        )

        flow_count = 0
        nonspotify_kept = 0

        for row in extract_all_features(flows, label=label):
            row["Source_PCAP"] = f.name
            flow_count += 1

            if label == 1:
                spotify_rows.append(row)
            else:
                if args.max_nonspotify_per_pcap and args.max_nonspotify_per_pcap > 0:
                    if nonspotify_kept >= args.max_nonspotify_per_pcap:
                        continue
                    nonspotify_kept += 1
                nonspotify_rows.append(row)

        print(f"[OK] Extracted {flow_count} flows from {f.name} in {time.time()-t0:.1f}s")

    n_spotify = len(spotify_rows)
    n_non = len(nonspotify_rows)

    target_non = int(max(0, round(args.nonspotify_ratio * n_spotify)))

    # Subsampling global NON-Spotify
    if target_non > 0 and n_non > target_non:
        df_non = pd.DataFrame(nonspotify_rows).sample(n=target_non, random_state=42)
        nonspotify_rows = df_non.to_dict(orient="records")

    df = pd.DataFrame(spotify_rows + nonspotify_rows)

    print("\n[SUMMARY]")
    print(f"  Spotify rows     : {len(spotify_rows)}")
    print(f"  Non-Spotify rows : {len(nonspotify_rows)}")
    print(f"  Total rows       : {len(df)}")

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path, index=False)
    print(f"[OK] Wrote CSV: {out_path}")


if __name__ == "__main__":
    main()
