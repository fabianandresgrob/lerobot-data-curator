"""
filter_dataset.py — top-level entry point for lerobot-data-curator.

Scores a LeRobot dataset on both technical and semantic dimensions and saves
a filtered copy containing only episodes that pass both thresholds.

Usage:
    python filter_dataset.py \\
        --repo_id your-hf-username/your-dataset \\
        --task_description "pick up the orange cube and place it in the blue container" \\
        --output filtered_dataset/

The pre-trained FS block weights for pick-and-place are downloaded automatically
from HuggingFace. To use custom weights, pass --fs_weights path/to/weights.pt.
To train your own, see docs/train_your_own_fs_blocks.md.
"""

import argparse
import sys
from pathlib import Path

# Make score_lerobot_episodes importable from the submodule
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "score_lerobot_episodes" / "src"))
sys.path.insert(0, str(REPO_ROOT / "score_lerobot_episodes"))
sys.path.insert(0, str(REPO_ROOT / "I-FailSense" / "src"))

# Default pre-trained weights on HuggingFace
DEFAULT_WEIGHTS_REPO = "fabiangrob/failsense-fs-blocks-pick-place"
DEFAULT_WEIGHTS_FILE = "best_model.pt"


def download_weights(cache_dir: Path) -> Path:
    """Download pre-trained FS block weights from HuggingFace if not cached."""
    from huggingface_hub import hf_hub_download
    weights_path = hf_hub_download(
        repo_id=DEFAULT_WEIGHTS_REPO,
        filename=DEFAULT_WEIGHTS_FILE,
        cache_dir=str(cache_dir),
    )
    return Path(weights_path)


def main():
    ap = argparse.ArgumentParser(
        description="Filter a LeRobot dataset by technical and semantic quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter using pre-trained pick-and-place weights (auto-downloaded)
  python filter_dataset.py \\
      --repo_id fabiangrob/my_pick_place_dataset \\
      --task_description "pick up the orange cube and place it in the blue container" \\
      --output filtered/

  # Use local dataset root (no HF download)
  python filter_dataset.py \\
      --repo_id fabiangrob/my_pick_place_dataset \\
      --root /data/lerobot_datasets \\
      --task_description "pick up the orange cube..." \\
      --output filtered/

  # Use custom FS block weights (trained on your own data)
  python filter_dataset.py \\
      --repo_id fabiangrob/my_dataset \\
      --task_description "your task description" \\
      --fs_weights checkpoints/my_fs_blocks.pt \\
      --output filtered/

  # Technical filtering only (no GPU required)
  python filter_dataset.py \\
      --repo_id fabiangrob/my_dataset \\
      --task_description "..." \\
      --no_semantic \\
      --output filtered/
        """,
    )
    ap.add_argument("--repo_id", required=True,
                    help="HuggingFace dataset repo ID (e.g. your-user/your-dataset).")
    ap.add_argument("--task_description", required=True,
                    help="Natural language description of the task the robot should perform.")
    ap.add_argument("--output", required=True,
                    help="Path to save the filtered dataset.")
    ap.add_argument("--root", default=None,
                    help="Local dataset root directory. If omitted, downloads from HuggingFace.")
    ap.add_argument("--fs_weights", default=None,
                    help="Path to FS block weights (.pt). If omitted, downloads pre-trained "
                         f"weights from {DEFAULT_WEIGHTS_REPO}.")
    ap.add_argument("--technical_threshold", type=float, default=0.5,
                    help="Aggregate technical score threshold (default: 0.5).")
    ap.add_argument("--semantic_threshold", type=float, default=0.5,
                    help="Semantic score threshold (default: 0.5).")
    ap.add_argument("--no_semantic", action="store_true",
                    help="Disable semantic scoring. Runs technical filter only (no GPU needed).")
    ap.add_argument("--nominal", type=float, default=None,
                    help="Expected episode length in frames, used by the runtime scorer. "
                         "If omitted, estimated from the dataset median.")
    ap.add_argument("--video_backend", default="pyav",
                    help="Video backend for LeRobot dataset loading (default: pyav).")
    args = ap.parse_args()

    # Resolve FS weights
    fs_weights_path = None
    if not args.no_semantic:
        if args.fs_weights:
            fs_weights_path = Path(args.fs_weights)
            if not fs_weights_path.exists():
                print(f"Error: --fs_weights path does not exist: {fs_weights_path}")
                sys.exit(1)
        else:
            print(f"Downloading pre-trained FS block weights from {DEFAULT_WEIGHTS_REPO}...")
            cache_dir = REPO_ROOT / ".weights_cache"
            cache_dir.mkdir(exist_ok=True)
            fs_weights_path = download_weights(cache_dir)
            print(f"Weights cached at: {fs_weights_path}")

    # Build sys.argv for score_dataset.py and call its main()
    score_argv = [
        "score_dataset.py",
        "--repo_id", args.repo_id,
        "--output", args.output,
        "--threshold", str(args.technical_threshold),
        "--semantic_threshold", str(args.semantic_threshold),
    ]
    if args.root:
        score_argv += ["--root", args.root]
    if args.nominal is not None:
        score_argv += ["--nominal", str(args.nominal)]
    if not args.no_semantic:
        score_argv += [
            "--semantic",
            "--task_description", args.task_description,
            "--semantic_fs_weights", str(fs_weights_path),
        ]

    # Import and run score_dataset.main() with the constructed args
    sys.argv = score_argv
    import score_dataset
    score_dataset.main()


if __name__ == "__main__":
    main()
