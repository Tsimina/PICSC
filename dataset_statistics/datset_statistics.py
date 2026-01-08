import argparse
import os
from collections import defaultdict

def safe_id(s: str) -> str:
    # Graphviz node IDs can't have many special chars reliably
    return "".join(c if c.isalnum() else "_" for c in s)

def scan_structure(root: str, max_depth: int = 3):
    """
    Returns:
      tree[split][traffic_class] -> set(applications)
    """
    tree = defaultdict(lambda: defaultdict(set))

    # splits are first-level directories
    for split in sorted(os.listdir(root)):
        split_path = os.path.join(root, split)
        if not os.path.isdir(split_path):
            continue

        # traffic classes are second-level directories
        for tc in sorted(os.listdir(split_path)):
            tc_path = os.path.join(split_path, tc)
            if not os.path.isdir(tc_path):
                continue

            # applications are third-level directories (or inferred from files)
            has_app_dirs = False
            for app in sorted(os.listdir(tc_path)):
                app_path = os.path.join(tc_path, app)
                if os.path.isdir(app_path):
                    has_app_dirs = True
                    tree[split][tc].add(app)

            # fallback: if no app subdirs, infer "application" by filename prefixes
            if not has_app_dirs:
                # if there are files directly under tc_path, treat tc as application
                # (keeps script resilient)
                files = [f for f in os.listdir(tc_path) if os.path.isfile(os.path.join(tc_path, f))]
                if files:
                    tree[split][tc].add(tc)  # app == tc (best-effort)

    return tree

def build_full_dot(tree, out_dot: str):
    """
    Full structure diagram: Dataset -> Split -> Traffic Class -> Application
    """
    dataset_node = "Dataset"

    lines = []
    lines.append('digraph G {')
    lines.append('  rankdir=LR;')
    lines.append('  node [shape=box, fontsize=11];')
    lines.append(f'  "{dataset_node}" [shape=folder, style="filled", fillcolor="#eeeeee"];')

    for split, tcs in tree.items():
        split_node = f"split::{split}"
        lines.append(f'  "{split_node}" [shape=folder];')
        lines.append(f'  "{dataset_node}" -> "{split_node}";')

        for tc, apps in tcs.items():
            tc_node = f"tc::{split}::{tc}"
            lines.append(f'  "{tc_node}" [shape=box];')
            lines.append(f'  "{split_node}" -> "{tc_node}";')

            for app in sorted(apps):
                app_node = f"app::{split}::{tc}::{app}"
                # highlight Spotify if present
                if app.lower() == "spotify":
                    lines.append(f'  "{app_node}" [shape=box, style="filled", fillcolor="#c9f7c9", label="Spotify"];')
                else:
                    lines.append(f'  "{app_node}" [shape=box, label="{app}"];')
                lines.append(f'  "{tc_node}" -> "{app_node}";')

    lines.append('}')
    with open(out_dot, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def build_spotify_binary_dot(tree, out_dot: str, positive_name="Spotify"):
    """
    Binary diagram: Dataset -> Split -> {Spotify, Other}
    """
    dataset_node = "Dataset"

    lines = []
    lines.append('digraph G {')
    lines.append('  rankdir=LR;')
    lines.append('  node [shape=box, fontsize=11];')
    lines.append(f'  "{dataset_node}" [shape=folder, style="filled", fillcolor="#eeeeee"];')

    for split, tcs in tree.items():
        split_node = f"split::{split}"
        lines.append(f'  "{split_node}" [shape=folder];')
        lines.append(f'  "{dataset_node}" -> "{split_node}";')

        # detect whether Spotify exists anywhere under this split
        found_spotify = False
        other_apps = set()

        for tc, apps in tcs.items():
            for app in apps:
                if app.lower() == positive_name.lower():
                    found_spotify = True
                else:
                    other_apps.add(app)

        spotify_node = f"bin::{split}::Spotify"
        other_node = f"bin::{split}::Other"

        # Spotify node (green)
        if found_spotify:
            lines.append(f'  "{spotify_node}" [style="filled", fillcolor="#c9f7c9", label="Spotify (positive)"];')
        else:
            lines.append(f'  "{spotify_node}" [style="filled", fillcolor="#ffe6e6", label="Spotify (missing in split)"];')

        # Other node (light)
        other_label = "Other apps (negative)"
        if other_apps:
            # keep label short; list up to 8 apps
            sample = ", ".join(sorted(list(other_apps))[:8])
            if len(other_apps) > 8:
                sample += ", …"
            other_label += f"\\n{sample}"

        lines.append(f'  "{other_node}" [style="filled", fillcolor="#f5f5f5", label="{other_label}"];')

        lines.append(f'  "{split_node}" -> "{spotify_node}";')
        lines.append(f'  "{split_node}" -> "{other_node}";')

    lines.append('}')
    with open(out_dot, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def try_render(dot_path: str, out_path: str):
    """
    Render using `dot` if available. Otherwise leave .dot for manual render.
    """
    import shutil
    import subprocess

    dot_bin = shutil.which("dot")
    if not dot_bin:
        print(f"[WARN] Graphviz 'dot' not found. Wrote: {dot_path}")
        print("       Install Graphviz, then run:")
        print(f"       dot -Tpng {dot_path} -o {out_path}")
        return False

    subprocess.run([dot_bin, "-Tpng", dot_path, "-o", out_path], check=True)
    print(f"[OK] Rendered: {out_path}")
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root folder (contains nonVPN_1, nonVPN_2, ...)")
    ap.add_argument("--outdir", default=".", help="Output directory")
    ap.add_argument("--positive", default="Spotify", help="Positive class application name (default: Spotify)")
    args = ap.parse_args()

    tree = scan_structure(args.root)

    os.makedirs(args.outdir, exist_ok=True)

    full_dot = os.path.join(args.outdir, "structure_full.dot")
    full_png = os.path.join(args.outdir, "structure_full.png")
    build_full_dot(tree, full_dot)
    try_render(full_dot, full_png)

    bin_dot = os.path.join(args.outdir, "structure_spotify_binary.dot")
    bin_png = os.path.join(args.outdir, "structure_spotify_binary.png")
    build_spotify_binary_dot(tree, bin_dot, positive_name=args.positive)
    try_render(bin_dot, bin_png)

    print("[DONE] Outputs written to:", os.path.abspath(args.outdir))

if __name__ == "__main__":
    main()