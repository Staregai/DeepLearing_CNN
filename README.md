# DeepLearning_CNN

End-to-end CINIC-10 experimentation project with:
- CNN baseline and grid experiments,
- EfficientNet transfer-learning baseline,
- Prototypical Networks (few-shot learning),
- reduced-data robustness analysis,
- soft-voting ensemble evaluation.

## 1. Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download and extract CINIC-10:

https://www.kaggle.com/datasets/mengcius/cinic10

Expected folder layout:

```text
cinic10/
	train/
		airplane/
		automobile/
		...
	valid/
		airplane/
		...
	test/
		airplane/
		...
```

Default dataset path used by scripts: `src/dataset`

You can override it with:

```bash
--data-dir /custom/path
```

`src.data.cinic10` supports split aliases:
- train/training
- valid/val/validation
- test/testing

Optional training subset controls (all training scripts):
- `--train-subset-ratio 0.25` to use 25% of train split
- `--train-subset-size 5000` to use exactly 5000 train samples

## 2. Project Structure

```text
configs/                         # currently empty (reserved for config files)
outputs/                         # training/evaluation artifacts
scripts/                         # runnable experiment entrypoints
	run_all.py
	run_cnn.py
	run_cnn_grid.py
	run_efficientnet.py
	run_ensemble.py
	run_fewshot.py
	run_reduced_data.py
src/
	config.py
	data/
		augmentations.py
		cinic10.py
	dataset/                       # local CINIC-10 data root (default)
	models/
		cnn_baseline.py
		efficientnet.py
		protonet.py
		prototypical_classifier.py
	training/
		early_stopping.py
		ensemble.py
		fewshot.py
		metrics.py
		supervised.py
	utils/
		device.py
		io.py
		reproducibility.py
```

## 3. Output Artifacts

Training/evaluation scripts write to `outputs/...` (or a custom path via `--out-dir` / `--out-root`).

Typical training output directory contains:
- `best.pt` - model weights with best validation metric
- `epoch_<N>.pt` - periodic checkpoints (`checkpoint_every`, default 5)
- `train_state.pt` - full resumable training state
- JSON summaries such as `result.json`, `history.json`, `summary.json`, `eval_result.json` (script-dependent)
- `tb/` - TensorBoard logs

## 4. Run Experiments

### CNN grid search

```bash
python scripts/run_cnn_grid.py --out-dir outputs/cnn_grid

# Example: train each run on 20% of train data
python scripts/run_cnn_grid.py --out-dir outputs/cnn_grid --train-subset-ratio 0.2
```

Important: current `run_cnn_grid.py` is configured with a reduced internal grid:
- augmentation: `autoaugment`
- optimizer: `sgd`
- momentum: `0.7`
- label smoothing: `0.2`, `0.0`
- dropout: `0.3`, `0.0`

So by default it runs 4 combinations. To expand the grid, edit lists in `scripts/run_cnn_grid.py`.

### Single CNN with Early Stopping

Train a single CNN configuration with early stopping (stops training when validation loss improvement falls below threshold):

```bash
python scripts/run_cnn.py --out-dir outputs/cnn_single
```

Customize hyperparameters and early stopping:

```bash
python scripts/run_cnn.py \
  --out-dir outputs/cnn_single \
  --epochs 200 \
  --learning-rate 1e-3 \
  --optimizer adam \
  --dropout 0.3 \
  --label-smoothing 0.1 \
  --aug-profile combo \
  --patience 10 \
  --min-delta 1e-4
```

Early stopping options:
- `--patience 10` - stop if no improvement for 10 epochs
- `--min-delta 1e-4` - minimum loss decrease to count as improvement
- `--aug-profile {baseline|color_jitter|autoaugment|cutout|compression|combo}` - augmentation strategy

Also supports subset training:
```bash
python scripts/run_cnn.py --train-subset-ratio 0.3 --patience 15
```

### EfficientNet baseline

```bash
python scripts/run_efficientnet.py

# Example: train on exactly 10000 samples
python scripts/run_efficientnet.py --out-dir outputs/efficientnet --train-subset-size 10000
```

Default output directory in this script is currently `outputs/efficientnet51`.

### Few-shot Prototypical Network

```bash
python scripts/run_fewshot.py --out-dir outputs/fewshot
```

### Reduced training set comparison

```bash
python scripts/run_reduced_data.py --out-dir outputs/reduced_data --ratio 0.3

# Example: first cap train split to 30%, then run full-vs-reduced comparison
python scripts/run_reduced_data.py --out-dir outputs/reduced_data --ratio 0.3 --train-subset-ratio 0.3
```

### Ensemble (soft voting)

```bash
python scripts/run_ensemble.py \
  --cnn-ckpt outputs/cnn_grid/<best_run>/best.pt \
  --effnet-ckpt outputs/efficientnet/best.pt \
  --fewshot-ckpt outputs/fewshot/best.pt \
  --out-file outputs/ensemble/result.json
```

Soft voting follows:

$$
P_{final} = \frac{1}{3}\sum_{i=1}^{3} P_i
$$

### Run Full Pipeline (run_all)

```bash
python scripts/run_all.py
```

Execution order in `run_all.py`:
1. few-shot
2. efficientnet
3. cnn grid
4. reduced-data
5. ensemble

Custom output root:

```bash
python scripts/run_all.py --out-root outputs_full
```

Run full pipeline on subset of train data:

```bash
python scripts/run_all.py --train-subset-ratio 0.3
```

### Pause and Resume Training

Training scripts save resumable state in `train_state.pt`, with periodic checkpoints controlled by `--checkpoint-every` (where available).

Saved files:
- `best.pt`: best validation model weights
- `epoch_<N>.pt`: periodic model snapshots
- `train_state.pt`: full resume state (epoch, model, optimizer, history, RNG state)

Resume a specific training script:

```bash
python scripts/run_efficientnet.py --out-dir outputs/efficientnet --resume
python scripts/run_cnn_grid.py --out-dir outputs/cnn_grid --resume
python scripts/run_fewshot.py --out-dir outputs/fewshot --resume
python scripts/run_reduced_data.py --out-dir outputs/reduced_data --resume
python scripts/run_cnn.py --out-dir outputs/cnn_single --resume
```

Resume the full pipeline:

```bash
python scripts/run_all.py --resume
```

Notes:
- Resume loads `train_state.pt` from each script's output directory
- If you stop between save points, training resumes from the last saved state



## 5. TensorBoard

```bash
tensorboard --logdir outputs
```

## 6. Notes

- CINIC-10 official statistics are used for normalization.
- AutoAugment uses CIFAR10 policy as a practical approximation for CINIC-10.
- `run_all.py` automatically reads best CNN checkpoint from `outputs/.../cnn_grid/summary.json` before running ensemble.
