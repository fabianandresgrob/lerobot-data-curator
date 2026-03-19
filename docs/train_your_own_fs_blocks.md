# Train FS Blocks on Your Own Data

The pre-trained FS block weights included with this tool were trained on a specific pick-and-place task (SO-101 arm, orange cube into blue container). For a different task or robot setup, you need to train the FS blocks from scratch. This is straightforward and requires only a small dataset.

---

## What you need

- A GPU with ≥8 GB VRAM (the VLM backbone is loaded in bfloat16)
- A LeRobot dataset with **positive episodes** (task executed correctly)
- One or more LeRobot datasets with **negative episodes** (task failed semantically)
- A task description string

**Important — what counts as a semantic negative:**

Use episodes where the robot did the wrong thing semantically: wrong object, task not completed, distractors that change the scene semantics. Do NOT use episodes where the robot did the right thing but the video quality was poor (bad lighting, shakiness). Those are technical failures and will confuse the semantic classifier.

Examples:
- ✅ Robot picks wrong object → negative
- ✅ Robot drops the object → negative
- ✅ Extra objects in the scene → negative
- ❌ Dark or blurry video → do NOT use as negative (use technical scorer for these)

---

## Camera setup

The adapter samples 4 evenly-spaced frames from each episode and expects two camera views:
- A **top/exocentric** camera (scene overview)
- A **wrist/egocentric** camera (gripper view)

Check your dataset's camera key names before training:
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("your/dataset")
print(list(ds.features.keys()))
```

Pass the correct key names to the training script via `--top_camera_key` and `--wrist_camera_key`.

---

## Training

```bash
cd score_lerobot_episodes
source .venv/bin/activate

python scripts/train_fs_blocks.py \
    --vlm_model_id ACIDE/FailSense-Calvin-2p-3b \
    --positive_repo_id your-user/your-clean-dataset \
    --negative_repo_ids \
        your-user/your-failure-dataset-1 \
        your-user/your-failure-dataset-2 \
    --task_description "your task description" \
    --top_camera_key observation.images.top \
    --wrist_camera_key observation.images.wrist \
    --num_epochs 10 \
    --output_dir checkpoints/my_fs_blocks \
    --video_backend pyav
```

**With local datasets:**
```bash
python scripts/train_fs_blocks.py \
    --positive_repo_id your-user/your-clean-dataset \
    --positive_root /path/to/local/datasets \
    --negative_repo_ids your-user/failure-dataset \
    --negative_roots /path/to/local/datasets \
    --task_description "your task description" \
    --output_dir checkpoints/my_fs_blocks
```

**Dry run first** (processes 3 episodes to verify everything works before committing to a full run):
```bash
python scripts/train_fs_blocks.py \
    ... \
    --dry_run
```

### Key hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `--num_epochs` | 10 | More epochs rarely help — FS blocks converge fast |
| `--batch_size` | 1 | Do not increase beyond 2 on 10 GB VRAM |
| `--gradient_accumulation_steps` | 4 | Effective batch = batch_size × grad_accum |
| `--learning_rate` | 1e-4 | Works well across tasks |
| `--val_split` | 0.2 | 80/20 train/val split |
| `--seed` | 42 | For reproducibility |

### W&B logging

Add `--wandb --wandb_project my-project` to log loss curves and accuracy to Weights & Biases.

### What gets trained

Only the FS blocks (attention pooling + MLP classifiers attached to 3 intermediate VLM layers). The VLM backbone is fully frozen. Trainable parameters: ~107M out of 3.16B (3.4%).

Expected training time on RTX 3080: 1–3 hours depending on dataset size.

---

## Using your trained weights

```bash
# With filter_dataset.py
python filter_dataset.py \
    --repo_id your-user/your-dataset \
    --task_description "your task description" \
    --fs_weights checkpoints/my_fs_blocks/best_model.pt \
    --output filtered/

# With score_dataset.py directly
cd score_lerobot_episodes
python score_dataset.py \
    --repo_id your-user/your-dataset \
    --task_description "your task description" \
    --semantic \
    --semantic_fs_weights ../checkpoints/my_fs_blocks/best_model.pt
```

---

## How much data do you need?

In our experiments, **270 episodes total** (120 positives + 150 negatives across 3 conditions) was sufficient to reach 100% validation accuracy by epoch 2. The frozen VLM backbone provides rich representations, so the FS blocks need very few examples to find the decision boundary.

A practical minimum is around **50 positives + 50 negatives**. Below that, results become unstable.

---

## Sharing your trained weights

If you train FS blocks for a new task and want to share them:

```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("your-user/failsense-fs-blocks-your-task", repo_type="model", exist_ok=True)
api.upload_file(
    path_or_fileobj="checkpoints/my_fs_blocks/best_model.pt",
    path_in_repo="best_model.pt",
    repo_id="your-user/failsense-fs-blocks-your-task",
    repo_type="model",
)
```

Then others can use your weights with:
```bash
python filter_dataset.py \
    --fs_weights your-user/failsense-fs-blocks-your-task \
    ...
```
