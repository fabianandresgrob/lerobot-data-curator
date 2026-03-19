# lerobot-data-curator

A data quality filtering tool for [LeRobot](https://github.com/huggingface/lerobot) datasets that combines **technical scoring** (visual clarity, smoothness, collision detection) with **semantic scoring** (did the robot actually do the right task?).

Built on top of [score_lerobot_episodes](https://github.com/fabianandresgrob/score_lerobot_episodes) and [I-FailSense](https://github.com/fabianandresgrob/I-FailSense).

---

## Why

Standard data quality tools catch technical issues: blur, shakiness, bad lighting. They miss semantic failures — episodes where the robot grabbed the wrong object, failed to complete the task, or had distractors in the scene. These episodes look fine technically but will silently degrade your trained policy.

This tool adds a VLM-based semantic scorer on top of the existing technical scorer. In our experiments on a pick-and-place task:

| Filter | Semantic failure recall | Technical failure recall |
|---|---|---|
| HF technical tool alone | ~0% | ~2% |
| I-FailSense alone | **100%** | **74–83%** |
| Combined | **100%** | **74–83%** |

Pre-trained FS block weights for a pick-and-place task are available on HuggingFace: [`fabiangrob/failsense-fs-blocks-pick-place`](https://huggingface.co/fabiangrob/failsense-fs-blocks-pick-place)

For other tasks, see [Train FS blocks on your own data](docs/train_your_own_fs_blocks.md).

---

## Quick start

```bash
# 1. Clone with submodules
git clone --recurse-submodules https://github.com/fabianandresgrob/lerobot-data-curator
cd lerobot-data-curator

# 2. Set up environment
bash setup.sh

# 3. Filter your dataset
source score_lerobot_episodes/.venv/bin/activate
python filter_dataset.py \
    --repo_id your-hf-username/your-lerobot-dataset \
    --task_description "pick up the orange cube and place it in the blue container" \
    --output filtered_dataset/
```

The script downloads the pre-trained FS block weights automatically and saves the filtered dataset to `filtered_dataset/`.

---

## Installation

### Requirements

- Python 3.10+
- `uv` package manager (`pip install uv`)
- CUDA GPU with ≥8 GB VRAM for semantic scoring (inference runs on GPU)
- macOS or Linux

### Setup

```bash
git clone --recurse-submodules https://github.com/fabianandresgrob/lerobot-data-curator
cd lerobot-data-curator
bash setup.sh
```

`setup.sh` creates a single virtual environment in `score_lerobot_episodes/.venv` and installs both tools into it. I-FailSense is installed with `--no-deps` to avoid a torch version conflict.

---

## Usage

### Filter a dataset

Scores every episode on technical and semantic dimensions, then saves a filtered copy containing only episodes that pass both thresholds.

```bash
python filter_dataset.py \
    --repo_id your-hf-username/your-dataset \
    --task_description "your task description" \
    --output path/to/filtered_dataset/ \
    [--root /local/dataset/root] \
    [--fs_weights path/to/custom_weights.pt] \
    [--technical_threshold 0.5] \
    [--semantic_threshold 0.5]
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--repo_id` | required | HuggingFace dataset repo ID |
| `--task_description` | required | Natural language description of the task |
| `--output` | required | Path to save the filtered dataset |
| `--root` | None | Local dataset root (skips HF download) |
| `--fs_weights` | auto-download | Path to FS block weights (.pt). If omitted, downloads from HuggingFace |
| `--technical_threshold` | 0.5 | Aggregate technical score threshold |
| `--semantic_threshold` | 0.5 | Semantic score threshold |
| `--no_semantic` | False | Disable semantic scoring (technical filter only) |

The output is a valid LeRobot dataset that can be used directly for policy training.

### Score only (no filtering)

To inspect scores without saving a filtered dataset:

```bash
cd score_lerobot_episodes
source .venv/bin/activate
python score_dataset.py \
    --repo_id your-hf-username/your-dataset \
    --task_description "your task description" \
    --semantic \
    --semantic_fs_weights path/to/weights.pt
```

This prints a per-episode table with technical scores, semantic scores, and a status column (`GOOD` / `FAIL_SEMANTIC` / `FAIL_TECHNICAL` / `FAIL_BOTH`).

### Use pre-trained weights for pick-and-place

The weights at [`fabiangrob/failsense-fs-blocks-pick-place`](https://huggingface.co/fabiangrob/failsense-fs-blocks-pick-place) were trained on an SO-101 arm performing pick-and-place (orange cube into blue container) with two cameras (top + wrist). They will work best for:

- Similar pick-and-place tasks
- Similar camera setups (top + wrist)
- Similar scene complexity

For different tasks or robot setups, [train your own FS blocks](docs/train_your_own_fs_blocks.md).

---

## How it works

Each episode is converted to a flat list of 8 frames: 4 evenly-spaced timesteps × 2 cameras (top, wrist). This is passed along with the task description string to I-FailSense, which outputs a success probability in [0, 1].

**I-FailSense architecture:**
1. **VLM backbone**: PaliGemma2-3B with a LoRA adapter (frozen, loaded from `ACIDE/FailSense-Calvin-2p-3b`)
2. **FS blocks**: three lightweight MLP classifiers attached to intermediate VLM layers via forward hooks — the only components that are task-specific and trainable

The backbone provides rich visual representations. The FS blocks learn a decision boundary in that feature space from a small amount of task-specific data.

---

## Results

Evaluated on 7 conditions of an SO-101 pick-and-place task (120 clean episodes, 50 episodes per failure condition):

| Condition | Type | VLM-only accuracy | Full model accuracy |
|---|---|---|---|
| clean | success | 0.47 | **1.00** |
| wrong_cube | semantic failure | 0.62 | **1.00** |
| task_fail | semantic failure | 0.56 | **1.00** |
| extra_objects | semantic failure | 0.30 | **1.00** |
| bad_lighting | technical failure | 0.00 | 0.72 |
| shakiness | technical failure | 0.46 | 0.76 |
| occluded_top_cam | technical failure | 0.76 | 0.74 |

The FS blocks were trained only on semantic failure conditions. The 72–76% accuracy on technical conditions is emergent generalization — the VLM backbone clusters visually degraded frames near the failure class without supervision.

---

## Repository structure

```
lerobot-data-curator/
├── README.md
├── setup.sh                          # environment setup
├── filter_dataset.py                 # main entry point
├── docs/
│   └── train_your_own_fs_blocks.md  # guide for custom FS block training
├── score_lerobot_episodes/           # submodule (semantic-scoring branch)
└── I-FailSense/                      # submodule (semantic-scoring branch)
```

---

## Citation

If you use this tool, please cite the original I-FailSense paper and the score_lerobot_episodes tool.
