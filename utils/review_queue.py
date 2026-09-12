"""Interactive and automated resolution tool for review_queue.csv."""
import json
import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd


def load_review_queue(queue_path: str | Path = "output/review_queue.csv") -> pd.DataFrame:
    """Load the fuzzy matching review queue CSV."""
    path = Path(queue_path)
    if not path.exists():
        return pd.DataFrame(columns=["name1", "name2", "confidence", "platform1", "platform2", "merged"])
    return pd.read_csv(path)


def save_review_queue(df: pd.DataFrame, queue_path: str | Path = "output/review_queue.csv") -> None:
    """Save the updated review queue DataFrame."""
    path = Path(queue_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_overrides(overrides_path: str | Path = "config/match_overrides.json") -> dict[str, Any]:
    """Load curated match overrides mapping variant names to canonical names."""
    path = Path(overrides_path)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {k: v for k, v in data.items() if not k.startswith("//")}
    except Exception as e:
        logging.warning("Error reading %s: %s", path, e)
        return {}


def save_overrides(overrides: dict[str, Any], overrides_path: str | Path = "config/match_overrides.json") -> None:
    """Save match overrides dictionary to JSON."""
    path = Path(overrides_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=2, ensure_ascii=False)


def get_review_status(queue_path: str | Path = "output/review_queue.csv") -> dict[str, Any]:
    """Summarize the status of review queue items."""
    df = load_review_queue(queue_path)
    if df.empty:
        status = {
            "total_pairs": 0,
            "pending_pairs": 0,
            "merged_pairs": 0,
            "platforms": {},
            "confidence_distribution": {},
        }
        print("Review queue is empty or does not exist.")
        return status

    # Ensure merged column exists
    if "merged" not in df.columns:
        df["merged"] = False

    total = len(df)
    merged_count = int(df["merged"].astype(bool).sum())
    pending_count = total - merged_count

    conf = df["confidence"].astype(float)
    high = int(((conf >= 0.88) & (~df["merged"].astype(bool))).sum())
    medium = int(((conf >= 0.84) & (conf < 0.88) & (~df["merged"].astype(bool))).sum())
    low = int(((conf < 0.84) & (~df["merged"].astype(bool))).sum())

    platform_counts = df[~df["merged"].astype(bool)]["platform1"].value_counts().to_dict()

    status = {
        "total_pairs": total,
        "pending_pairs": pending_count,
        "merged_pairs": merged_count,
        "platforms": platform_counts,
        "confidence_distribution": {
            ">= 0.88 (High confidence)": high,
            "0.84 - 0.87 (Moderate)": medium,
            "< 0.84 (Low)": low,
        },
    }

    print("=" * 60)
    print("REVIEW QUEUE STATUS")
    print("=" * 60)
    print(f"Total pairs in queue: {total}")
    print(f"Pending review:       {pending_count}")
    print(f"Already resolved:     {merged_count}")
    print("\nPending by confidence tier:")
    print(f"  High (>= 0.88):      {high}")
    print(f"  Moderate (0.84-0.87): {medium}")
    print(f"  Low (< 0.84):         {low}")
    if platform_counts:
        print("\nTop platforms with pending pairs:")
        for plat, count in list(platform_counts.items())[:5]:
            print(f"  {plat}: {count}")
    print("=" * 60)

    return status


def record_override(
    overrides: dict[str, Any],
    platform: str,
    variant_name: str,
    canonical_name: str,
) -> None:
    """Record a match override into the nested overrides dictionary."""
    if platform:
        if platform not in overrides or not isinstance(overrides[platform], dict):
            overrides[platform] = {}
        overrides[platform][variant_name] = canonical_name
    else:
        overrides[variant_name] = canonical_name


def auto_resolve_queue(
    queue_path: str | Path = "output/review_queue.csv",
    overrides_path: str | Path = "config/match_overrides.json",
    min_confidence: float = 0.88,
) -> int:
    """Auto-approve review queue pairs having confidence >= min_confidence.

    Prefers the cleaner title (without subtitles or shorter) as canonical.
    """
    df = load_review_queue(queue_path)
    if df.empty:
        print("No items in review queue to auto-resolve.")
        return 0

    if "merged" not in df.columns:
        df["merged"] = False

    overrides = load_overrides(overrides_path)
    resolved_count = 0

    for idx, row in df.iterrows():
        if row["merged"]:
            continue

        conf = float(row["confidence"])
        if conf >= min_confidence:
            n1, n2 = str(row["name1"]).strip(), str(row["name2"]).strip()
            plat = str(row.get("platform1", "") or "").strip()

            # Choose canonical: shorter name or the one without trailing tags/subtitles
            if len(n1) <= len(n2):
                canonical, variant = n1, n2
            else:
                canonical, variant = n2, n1

            record_override(overrides, plat, variant, canonical)
            df.at[idx, "merged"] = True
            resolved_count += 1

    if resolved_count > 0:
        save_overrides(overrides, overrides_path)
        save_review_queue(df, queue_path)
        print(f"Auto-resolved {resolved_count} pairs with confidence >= {min_confidence}.")
        print(f"Saved overrides to {overrides_path} and updated {queue_path}.")
    else:
        print(f"No pending pairs met the minimum confidence threshold ({min_confidence}).")

    return resolved_count


def interactive_review(
    queue_path: str | Path = "output/review_queue.csv",
    overrides_path: str | Path = "config/match_overrides.json",
    limit: Optional[int] = None,
) -> int:
    """Interactive CLI flow to manually inspect and approve ambiguous duplicate titles."""
    df = load_review_queue(queue_path)
    if df.empty:
        print("No review queue found or queue is empty.")
        return 0

    if "merged" not in df.columns:
        df["merged"] = False

    pending_indices = df[~df["merged"].astype(bool)].index.tolist()
    if not pending_indices:
        print("All pairs in review queue have already been resolved!")
        return 0

    if limit:
        pending_indices = pending_indices[:limit]

    overrides = load_overrides(overrides_path)
    resolved_count = 0

    print(f"\nStarting interactive review for {len(pending_indices)} pending pairs.")
    print("Options: [1] Keep Name 1 | [2] Keep Name 2 | [s] Skip | [q] Quit\n")

    for i, idx in enumerate(pending_indices, 1):
        row = df.loc[idx]
        n1 = str(row["name1"]).strip()
        n2 = str(row["name2"]).strip()
        plat = str(row.get("platform1", "") or "").strip()
        conf = float(row["confidence"])

        print("-" * 60)
        print(f"[{i}/{len(pending_indices)}] Platform: {plat or 'Unknown'} | Confidence: {conf:.4f}")
        print(f"  [1] {n1}")
        print(f"  [2] {n2}")

        choice = input("Decision [1 / 2 / s(skip) / q(quit)]: ").strip().lower()

        if choice == "q":
            print("\nExiting review early.")
            break
        elif choice == "1":
            record_override(overrides, plat, n2, n1)
            df.at[idx, "merged"] = True
            resolved_count += 1
            print(f"✓ Approved: '{n2}' -> '{n1}'")
        elif choice == "2":
            record_override(overrides, plat, n1, n2)
            df.at[idx, "merged"] = True
            resolved_count += 1
            print(f"✓ Approved: '{n1}' -> '{n2}'")
        elif choice == "s":
            print("Skipped.")
        else:
            print("Unrecognized option, skipping.")

    if resolved_count > 0:
        save_overrides(overrides, overrides_path)
        save_review_queue(df, queue_path)
        print(f"\nSuccessfully resolved {resolved_count} pairs.")
        print(f"Saved overrides to {overrides_path} and updated {queue_path}.")

    return resolved_count
