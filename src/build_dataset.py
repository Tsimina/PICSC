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
    # Spotify = 1, restul = 0 (bazat pe numele fișierului)
    return 1 if "spotify" in name.lower() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap_dir", type=str, required=True, help="Director (poate conține subfoldere) cu .pcap/.pcapng")
    ap.add_argument("--out", type=str, required=True, help="Calea CSV de ieșire")
    ap.add_argument("--timeout", type=float, default=60.0, help="Flow timeout în secunde (default 600)")
    ap.add_argument("--active_duration", "--flow_duration", dest="active_duration", type=float, default=30.0,
                    help="Segment active flows into fixed-duration windows in seconds (e.g., 15). Omit or set to None to disable.")

    # Balancing
    ap.add_argument("--nonspotify_ratio", type=float, default=1.0,
                    help="Câte exemple NON-Spotify păstrezi relativ la Spotify (1.0=egal, 2.0=de 2 ori mai multe).")
    ap.add_argument("--max_nonspotify_per_pcap", type=int, default=0,
                    help="Limitează câte flow-uri NON-Spotify păstrezi per fișier (0=fără limită).")
    ap.add_argument("--seed", type=int, default=42, help="Seed pentru subsampling (reproductibil).")
    args = ap.parse_args()

    random.seed(args.seed)


    pcap_dir = Path(args.pcap_dir)

    # IMPORTANT: căutare RECURSIVĂ (dataseturile au des subfoldere)
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

    if n_spotify == 0:
        print("[WARN] Nu am găsit niciun flow Spotify (Label=1). Verifică numele fișierelor (trebuie să conțină 'spotify').")
        target_non = n_non
    else:
        target_non = int(max(0, round(args.nonspotify_ratio * n_spotify)))

    # Subsampling global NON-Spotify
    if target_non > 0 and n_non > target_non:
        df_non = pd.DataFrame(nonspotify_rows).sample(n=target_non, random_state=args.seed)
        nonspotify_rows = df_non.to_dict(orient="records")

    df = pd.DataFrame(spotify_rows + nonspotify_rows)

    print("\n[SUMMARY]")
    print(f"  Spotify rows     : {len(spotify_rows)}")
    print(f"  Non-Spotify rows : {len(nonspotify_rows)}")
    print(f"  Total rows       : {len(df)}")

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        print("[WARN] DataFrame gol. Nu am ce salva (verifică dacă PCAP-urile au IP/TCP/UDP).")
    else:
        df.to_csv(out_path, index=False)
        print(f"[OK] Wrote CSV: {out_path}")


if __name__ == "__main__":
    main()
