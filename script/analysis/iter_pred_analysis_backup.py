#!/usr/bin/env python3
"""End-to-end analyses for TaH labeled datasets.

This script performs the following tasks:

1) Compute P(iter1_decision=1 | iter1_pred=x) from tah_decision.labeled_correct.csv
   - Applies a minimum support threshold (default 0.1% of rows)
   - Prints the top-K tokens by conditional probability

2) Use only tah_decision.labeled_correct.csv for all subsequent analyses
   - Define iter2_pred as existing only when iter1_decision==1; in that case
     iter2_pred is equal to final_pred. For iter1_decision==0, iter2_pred does
     not exist (ignored in analyses that need iter2).

3) Compute P(iter1_decision=1 | final_pred=x) from the corrected CSV
   - Uses the same minimum support threshold and prints the top-K tokens

4) For selected iter1_pred seeds (auto-chosen from top-3 unless overridden),
   prints their most common iter2 (i.e., final_pred on rows with iter1_decision==1)
   values with counts and percentages, using only the corrected CSV.

All default file paths target the 1.7B analysis directory.
"""

import argparse
import csv
import os
from collections import Counter, defaultdict
from math import ceil


def format_token(token: str) -> str:
    if token is None:
        return ""
    return token.replace("\n", "\\n").replace("\t", "\\t")


def compute_conditional_probability(
    csv_path: str,
    group_col: str,
    decision_col: str = "iter1_decision",
    min_support_ratio: float = 0.001,
    top_k: int = 50,
    row_filter=None,
):
    """Compute P(decision_col==1 | group_col=x) with min-support filter.

    Returns: (total_rows, base_rate, min_support_count, leaderboard)
      leaderboard: list of tuples (prob, count, positives, token)
    """
    counts = defaultdict(int)
    positives = defaultdict(int)

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if group_col not in reader.fieldnames or decision_col not in reader.fieldnames:
            raise ValueError(
                f"Required columns missing in {csv_path}: {group_col} or {decision_col}"
            )
        for row in reader:
            if row_filter is not None and not row_filter(row):
                continue
            key = row.get(group_col)
            val = row.get(decision_col)
            try:
                decision_is_one = int(val) == 1
            except Exception:
                # Skip malformed decision values
                continue
            counts[key] += 1
            if decision_is_one:
                positives[key] += 1

    total_rows = sum(counts.values())
    base_rate = (sum(positives.values()) / total_rows) if total_rows else 0.0
    min_support = ceil(min_support_ratio * total_rows)

    leaderboard = []
    for token, n in counts.items():
        if n < min_support:
            continue
        k = positives[token]
        p = k / n if n else 0.0
        leaderboard.append((p, n, k, token))

    leaderboard.sort(key=lambda r: (r[0], r[1], r[3]), reverse=True)
    return total_rows, base_rate, min_support, leaderboard[:top_k]


def iter_rows_with_iter2_from_corrected(corrected_csv: str):
    """Yield rows from corrected CSV, attaching an implicit iter2_pred only when
    iter1_decision == 1 (then iter2_pred == final_pred). Rows with iter1_decision == 0
    are yielded with iter2_pred = None.
    """
    with open(corrected_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                dec = int(row.get("iter1_decision"))
            except Exception:
                continue
            row = dict(row)
            row["iter2_pred"] = row.get("final_pred") if dec == 1 else None
            yield row


def most_common_iter2_for_iter1(
    combined_csv: str,
    iter1_values: list,
    iter1_col: str = "iter1_pred",
    iter2_col: str = "iter2_pred",
    top_n: int = 20,
):
    """For each iter1 in iter1_values, list most common iter2 values with counts.

    Returns dict: iter1_value -> list[(iter2, count, pct)]
    """
    counters = {val: Counter() for val in iter1_values}
    totals = Counter()

    with open(combined_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            i1 = row.get(iter1_col)
            i2 = row.get(iter2_col)
            if i1 in counters and i2 is not None:
                counters[i1][i2] += 1
                totals[i1] += 1

    result = {}
    for val in iter1_values:
        total = totals[val]
        ranked = []
        for tok, c in counters[val].most_common(top_n):
            pct = (c / total) if total else 0.0
            ranked.append((tok, c, pct))
        result[val] = (total, ranked)
    return result


def print_leaderboard(title: str, total_rows: int, base_rate: float, min_support: int, leaderboard: list):
    print(title)
    print(f"Total rows={total_rows}, base_rate={base_rate:.6f}, min_support={min_support}")
    print("rank\tprob\tcount\tpositives\ttoken")
    for i, (prob, count, pos, token) in enumerate(leaderboard, 1):
        print(f"{i}\t{prob:.6f}\t{count}\t{pos}\t{format_token(token)}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Iter prediction analyses and CSV combiner")
    parser.add_argument("--root", default="/share/futianyu/cloud/repo/TaH/local/data/analysis/1.7B", help="Base directory for input/output CSVs")
    parser.add_argument("--min_support_ratio", type=float, default=0.00423, help="Minimum support ratio for leaderboards (default 0.1%)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K rows to display in leaderboards")
    parser.add_argument("--seeds", nargs="*", default=["\\(", "maybe", "But"], help="iter1_pred seed tokens to analyze for most common iter2_pred (ignored if auto seeds enabled)")
    parser.add_argument("--auto_seeds_top_k", type=int, default=2, help="If >0, automatically pick top-K tokens from iter1 leaderboard as seeds")
    parser.add_argument("--sankey_top_iter1", type=int, default=2, help="Top-N iter1 seeds to visualize in Sankey (default 2)")
    parser.add_argument("--sankey_top_iter2", type=int, default=3, help="Top-N iter2 destinations per seed for Sankey (default 3)")
    parser.add_argument("--sankey_html_out", default=None, help="Path to write Sankey HTML (default: <root>/iter_flow_sankey.html)")
    args = parser.parse_args()

    # Paths
    path_correct = os.path.join(args.root, "tah_decision.labeled_correct.csv")

    # 1) P(iter1_decision=1 | iter1_pred=x) [is_response=True]
    tot_r, base_r, min_sup_r, board_r = compute_conditional_probability(
        csv_path=path_correct,
        group_col="iter1_pred",
        decision_col="iter1_decision",
        min_support_ratio=args.min_support_ratio,
        top_k=args.top_k,
        row_filter=lambda r: r.get("is_response") in ("True", True),
    )
    print_leaderboard(
        title="P(iter1_decision=1 | iter1_pred=x) [is_response=True]",
        total_rows=tot_r,
        base_rate=base_r,
        min_support=min_sup_r,
        leaderboard=board_r,
    )

    # 2) No external join; derive iter2_pred from corrected only (where decision==1)
    # Show a short verification of counts with iter2 present
    iter2_present = 0
    total_rows_tmp = 0
    for row in iter_rows_with_iter2_from_corrected(path_correct):
        total_rows_tmp += 1
        if row.get("iter2_pred") is not None:
            iter2_present += 1
    print({"rows_in_corrected": total_rows_tmp, "rows_with_iter2_pred": iter2_present})
    print()

    # 3) P(iter1_decision=1 | final_pred (iter2)=x) using corrected only [is_response=True]
    #    Since iter2_pred exists only when decision==1, we equivalently group by final_pred
    #    but still compute P(decision==1 | final_pred=x)
    tot2, base2, min_sup2, board2 = compute_conditional_probability(
        csv_path=path_correct,
        group_col="final_pred",
        decision_col="iter1_decision",
        min_support_ratio=args.min_support_ratio,
        top_k=args.top_k,
        row_filter=lambda r: r.get("is_response") in ("True", True),
    )
    print_leaderboard(
        title="P(iter1_decision=1 | final_pred=x) [is_response=True]",
        total_rows=tot2,
        base_rate=base2,
        min_support=min_sup2,
        leaderboard=board2,
    )

    # 4) Most common iter2_pred for selected iter1_pred values
    # Determine seeds automatically from the top-K iter1 leaderboard if enabled
    if args.auto_seeds_top_k and args.auto_seeds_top_k > 0:
        auto_seeds = [t for (_, _, _, t) in board_r[: args.auto_seeds_top_k]]
        seeds_to_use = auto_seeds
        print(f"Selected seeds from top{args.auto_seeds_top_k} iter1 leaderboard: {[format_token(s) for s in seeds_to_use]}")
    else:
        seeds_to_use = args.seeds
        print(f"Using provided seeds: {[format_token(s) for s in seeds_to_use]}")

    print("Most common iter2_pred given iter1_pred seeds [is_response=True & iter1_decision=1]:")
    # Compute most common "iter2" as final_pred but only on rows where
    # is_response==True and iter1_decision==1, restricted to the selected seeds.
    seed_counters = {s: Counter() for s in seeds_to_use}
    seed_totals = Counter()
    with open(path_correct, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("is_response") not in ("True", True):
                continue
            try:
                dec = int(row.get("iter1_decision"))
            except Exception:
                continue
            if dec != 1:
                continue
            i1 = row.get("iter1_pred")
            if i1 not in seed_counters:
                continue
            i2 = row.get("final_pred")
            seed_counters[i1][i2] += 1
            seed_totals[i1] += 1

    for seed in seeds_to_use:
        total = seed_totals[seed]
        print(f"iter1_pred={format_token(seed)} total={total}")
        for tok, c in seed_counters[seed].most_common(3):
            pct = (c / total) if total else 0.0
            print(f"  {c}\t{pct:.4f}\t{format_token(tok)}")
        print()

    # 5) Two-column Sankey: Pred@Iter1 (passed) -> Pred@Iter2 (+ Others)
    sankey_seeds = [t for (_, _, _, t) in board_r[: args.sankey_top_iter1]]
    # Aggregate counts for per-destination for passing rows
    seed_dest = {s: Counter() for s in sankey_seeds}
    with open(path_correct, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("is_response") not in ("True", True):
                continue
            i1 = row.get("iter1_pred")
            if i1 not in seed_dest:
                continue
            try:
                dec = int(row.get("iter1_decision"))
            except Exception:
                continue
            if dec == 1:
                i2 = row.get("final_pred")
                seed_dest[i1][i2] += 1

    # Two-column Sankey: Pred@Iter1 (passed) -> Pred@Iter2 (+ Others)
    #    - Column 1: top-2 seeds (passed only)
    #    - Column 2: union of specified destinations plus an 'Others' bucket
    #    - Colors: red for 'But', blue for 'So', grey for other tokens, white for 'Others'

    # Decide union of destinations explicitly, defaulting to computed union
    explicit_union = ["Wait", "But", "Therefore", "So"]
    # Filter to those that actually appear; if none, fall back to computed dest_union
    dest_union_2col = [d for d in explicit_union if any(seed_dest[s].get(d, 0) > 0 for s in sankey_seeds)]
    if not dest_union_2col:
        dest_union = sorted({d for s in sankey_seeds for d in seed_dest.get(s, {}).keys()})
        dest_union_2col = dest_union[:]
    others_label = "Others"

    # Label formatting for 2-col Sankey: quote all tokens except Others
    def label_for_2col_token(token: str) -> str:
        if token == others_label:
            return others_label
        return f'"{token}"'

    # Pre-compute totals for counts on nodes
    total_passed_per_seed = {}
    dest_total_counts = {d: 0 for d in dest_union_2col}
    others_total_count = 0
    for s in sankey_seeds:
        dc = seed_dest.get(s, Counter())
        total_passed = sum(dc.values())
        total_passed_per_seed[s] = total_passed
        allocated = 0
        for d in dest_union_2col:
            c = dc.get(d, 0)
            dest_total_counts[d] += c
            allocated += c
        others_total_count += max(total_passed - allocated, 0)

    # Sort Column 2 by total amount (descending)
    dest_sorted = sorted(dest_union_2col, key=lambda d: dest_total_counts.get(d, 0), reverse=True)

    # Build node labels without counts
    col1_nodes_2 = [label_for_2col_token(s) for s in sankey_seeds]
    col2_nodes_2 = [label_for_2col_token(d) for d in dest_sorted] + [label_for_2col_token(others_label)]
    labels2 = col1_nodes_2 + col2_nodes_2

    # Color mapping function per token content
    def color_for_token(token: str) -> str:
        if token == "But":
            return "rgba(214, 39, 40, 0.8)"  # red
        if token == "So":
            return "rgba(31, 119, 180, 0.8)"  # blue
        if token == others_label:
            return "rgba(255, 255, 255, 1.0)"  # white for Others
        return "rgba(127, 127, 127, 0.8)"      # grey

    node_colors2 = []
    # Column 1 colors (map from seed token)
    for s in sankey_seeds:
        node_colors2.append(color_for_token(s))
    # Column 2 colors (map from destination token)
    for d in dest_sorted:
        node_colors2.append(color_for_token(d))
    # Add Others color
    node_colors2.append(color_for_token(others_label))

    # Build links seeds->destinations using passed counts only
    sources2, targets2, values2, link_labels2, link_colors2 = [], [], [], [], []
    def idx_c1(i):
        return i
    def idx_c2(i):
        return len(col1_nodes_2) + i

    for si, s in enumerate(sankey_seeds):
        dest_counter = seed_dest.get(s, Counter())
        allocated = 0
        for di, d in enumerate(dest_sorted):
            v = dest_counter.get(d, 0)
            if v > 0:
                sources2.append(idx_c1(si))
                targets2.append(idx_c2(di))
                values2.append(v)
                link_labels2.append("")  # No labels on links
                # Color flows to But/So as red/blue; others semi-transparent grey
                if d == "But":
                    link_colors2.append("rgba(214, 39, 40, 0.6)")
                elif d == "So":
                    link_colors2.append("rgba(31, 119, 180, 0.6)")
                else:
                    link_colors2.append("rgba(0,0,0,0.2)")
                allocated += v
        # Others
        total_passed = sum(dest_counter.values())
        v_other = total_passed - allocated
        if v_other > 0:
            sources2.append(idx_c1(si))
            targets2.append(idx_c2(len(dest_sorted)))  # index of Others node
            values2.append(v_other)
            link_labels2.append("")  # No labels on links
            link_colors2.append("rgba(0,0,0,0.1)")

    out_html2 = os.path.join(args.root, "iter_flow_sankey_2col.html")
    out_pdf2 = os.path.join(args.root, "iter_flow_sankey_2col.pdf")
    
    try:
        import plotly.graph_objects as go
        fig2 = go.Figure(
            data=[
                go.Sankey(
                    arrangement="snap",
                    node=dict(
                        label=labels2,
                        pad=15,
                        thickness=20,
                        color=node_colors2,
                        line=dict(color="black", width=0.5),
                    ),
                    link=dict(
                        source=sources2,
                        target=targets2,
                        value=values2,
                        label=link_labels2,
                        color=link_colors2,
                        hovertemplate="%{label}<extra></extra>",
                    ),
                )
            ]
        )
        
        width=600
        height=375
        fig2.update_layout(
            font_size=18,
            height=height,  # 3.5 inches * 300 DPI
            width=width,   # 6 inches * 300 DPI
        )
        
        # Save HTML
        fig2.write_html(out_html2, include_plotlyjs="cdn")
        print({"sankey_2col_saved": out_html2, "nodes": len(labels2), "links": len(values2)})
        
        # Export PDF at 300 DPI with size 6x3.5 inches
        try:
            fig2.write_image(out_pdf2, format="pdf", width=width, height=height, scale=1)
            print({"sankey_2col_pdf_saved": out_pdf2, "dpi": 300, "size_inches": [6, 3.5]})
        except Exception as e_img:
            print({"sankey_2col_pdf_error": str(e_img), "hint": "pip install -U kaleido for static image export"})
            
    except Exception as e:
        out_csv2 = os.path.join(args.root, "iter_flow_sankey_2col_links.csv")
        with open(out_csv2, "w", encoding="utf-8", newline="") as fo:
            w = csv.writer(fo)
            w.writerow(["source_label", "target_label", "value"])
            for s_idx, t_idx, v in zip(sources2, targets2, values2):
                w.writerow([labels2[s_idx], labels2[t_idx], v])
        print({"sankey_2col_links_csv": out_csv2, "error": str(e)})


if __name__ == "__main__":
    main()
