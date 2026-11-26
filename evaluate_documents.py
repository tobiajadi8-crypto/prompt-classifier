"""Compare prefix and full-document inference on the same fixed position probe."""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from benchmarks.cases import cases
from scanner import DocumentScanner, OVERLAP


@torch.inference_mode()
def prefix_scan(scanner, text):
    tokens = scanner.tokenizer(text, truncation=True, max_length=scanner.limit, return_tensors="pt")
    return scanner.model(**tokens).logits.softmax(-1)[0, 1].item()


def benchmark(scanner, repeats):
    sample = next(cases())[2]
    prefix_scan(scanner, sample)
    scanner.scan(sample)
    records = []
    for title, position, text, passage in cases():
        if position in ("middle", "end"):
            before = text[:text.index(passage)]
            assert len(scanner.tokenizer.encode(before, add_special_tokens=False, verbose=False)) >= scanner.limit
        row = {"document": title, "position": position, "expected": "attack" if passage else "safe"}
        timings = {"prefix": [], "full": []}
        for repeat in range(repeats):
            for mode in (("prefix", "full") if repeat % 2 == 0 else ("full", "prefix")):
                start = time.perf_counter()
                if mode == "prefix":
                    score = prefix_scan(scanner, text)
                else:
                    result = scanner.scan(text)
                    score = result["attack_score"]
                timings[mode].append((time.perf_counter() - start) * 1000)
                row[mode] = {"flagged": score >= scanner.threshold, "score": round(score, 6)}
        row["passages"] = result["passages_checked"]
        if passage:
            offset = text.index(passage)
            peak = result["highest_scoring_passage"]
            row["peak_contains_labeled_passage"] = peak["start"] <= offset and offset + len(passage) <= peak["end"]
        for mode in timings:
            row[mode]["median_ms"] = round(statistics.median(timings[mode]), 3)
        records.append(row)
    summary = {}
    for mode in ("prefix", "full"):
        counts = {p: sum(r[mode]["flagged"] for r in records if r["position"] == p)
                  for p in ("clean", "start", "middle", "end")}
        summary[mode] = {"flagged_by_position": counts,
                         "median_ms": round(statistics.median(r[mode]["median_ms"] for r in records), 3)}
    summary["full"]["peak_contains_labeled_passage"] = {
        position: sum(r["peak_contains_labeled_passage"] for r in records if r["position"] == position)
        for position in ("start", "middle", "end")
    }
    return {"cases": len(records), "positions_per_document": 4, "window_tokens": scanner.limit,
            "overlap": OVERLAP, "threshold": scanner.threshold, "repeats": repeats, "threads": 2,
            "platform": platform.platform(), "torch": str(torch.__version__), "summary": summary, "rows": records}


def published_test(scanner, path):
    import pyarrow.parquet as pq

    counts = {mode: dict(tp=0, tn=0, fp=0, fn=0) for mode in ("prefix", "full")}
    changes = []
    long_inputs = 0
    for index, row in enumerate(pq.read_table(path).to_pylist()):
        full = scanner.scan(row["text"])
        long_inputs += full["passages_checked"] > 1
        predictions = {
            "prefix": prefix_scan(scanner, row["text"]) >= scanner.threshold,
            "full": full["label"] == "attack",
        }
        for mode, positive in predictions.items():
            correct = positive == bool(row["label"])
            key = ("t" if correct else "f") + ("p" if positive else "n")
            counts[mode][key] += 1
        if predictions["prefix"] != predictions["full"]:
            changes.append(index)
    return {"counts": counts, "multi_window_inputs": long_inputs, "changed_row_indices": changes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("document_evaluation.json"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--published-test", type=Path, help="Optional pinned deepset test parquet from fetch.py")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    torch.set_num_threads(2)
    scanner = DocumentScanner()
    report = benchmark(scanner, args.repeats)
    if args.published_test:
        report["published_test"] = published_test(scanner, args.published_test)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
