#!/usr/bin/env python3
"""
build_dataset.py

Construiește un CSV cu feature-uri la nivel de flux pentru clasificare binară:
  Spotify (Label=1) vs Restul (Label=0)

Regula de etichetare (implicit):
  dacă numele fișierului conține "spotify" (case-insensitive) => Label=1, altfel 0.

Exemplu:
  python build_dataset.py --pcap_dir /cale/catre/pcapuri --out dataset.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

import sys
from pathlib import Path as _Path

# Adaugă folderul 'src' în PYTHONPATH ca să putem importa modulele fără pachet (__init__.py)
_THIS = _Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS / "src"))

from pcap_ingest import pcap_reader_generator, normalize_flow_key
from flow_aggregator import aggregate_flows
from feature_extractor import extract_all_features

def normalized_packets_from_file(pcap_path: Path):
    for pkt in pcap_reader_generator(str(pcap_path)):
        yield normalize_flow_key(pkt), pkt

def infer_label_from_name(name: str) -> int:
    return 1 if "spotify" in name.lower() else 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap_dir", type=str, required=True, help="Directorul cu fișiere .pcap/.pcapng")
    ap.add_argument("--out", type=str, required=True, help="Calea CSV de ieșire")
    ap.add_argument("--timeout", type=float, default=600.0, help="Flow timeout în secunde (default 600)")
    args = ap.parse_args()

    pcap_dir = Path(args.pcap_dir)
    files = sorted([*pcap_dir.glob("*.pcap"), *pcap_dir.glob("*.pcapng")])

    if not files:
        raise SystemExit(f"Nu am găsit fișiere .pcap/.pcapng în: {pcap_dir}")

    all_rows = []
    for f in files:
        label = infer_label_from_name(f.name)
        print(f"[INFO] Processing {f.name} (label={label})")
        flows = aggregate_flows(normalized_packets_from_file(f), timeout=args.timeout, include_packets=True)
        for row in extract_all_features(flows, label=label):
            row["Source_PCAP"] = f.name
            all_rows.append(row)

    df = pd.DataFrame(all_rows)

    # Ordine coloane: label la final + metadata
    cols = [c for c in df.columns if c not in ("Label",)]
    cols = cols + ["Label"]
    df = df[cols]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] Scris {len(df)} rânduri în {out_path}")

if __name__ == "__main__":
    main()
