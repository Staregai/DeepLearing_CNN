# DeepLearning_CNN

End-to-end implementation of the CINIC-10 project described in the documentation:
- CNN hyperparameter grid experiments,
- EfficientNet transfer learning baseline,
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

Default dataset path used by scripts: src/dataset
You can still override it with --data-dir /custom/path.

Optional training subset controls (all training scripts):
- --train-subset-ratio 0.25 to use 25% of train split
- --train-subset-size 5000 to use exactly 5000 train samples

## 2. Project Structure

```text
src/
	data/
		augmentations.py
		cinic10.py
	models/
		cnn_baseline.py
		efficientnet.py
		protonet.py
		prototypical_classifier.py
	training/
		supervised.py
		fewshot.py
		ensemble.py
		metrics.py
	utils/
		reproducibility.py
		device.py
		io.py
scripts/
	run_cnn_grid.py
	run_efficientnet.py
	run_fewshot.py
	run_cnn.py
	run_reduced_data.py
	run_ensemble.py
	run_all.py
```

## 3. Reproducibility

- Deterministic seed initialization is applied in each run.
- All runs save metrics, configs, TensorBoard logs, and checkpoints.

## 4. Run Experiments

### CNN grid search

```bash
python scripts/run_cnn_grid.py --out-dir outputs/cnn_grid

# Example: train each run on 20% of train data
python scripts/run_cnn_grid.py --out-dir outputs/cnn_grid --train-subset-ratio 0.2
```

Investigated dimensions include:
- training hyperparameters: optimizer (SGD/Adam), momentum,
- regularization hyperparameters: dropout, label smoothing,
- augmentation profiles: baseline + advanced methods.

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
- `--patience 10` – Stop if no improvement for 10 epochs
- `--min-delta 1e-4` – Minimum loss decrease to count as improvement
- `--aug-profile {baseline|color_jitter|autoaugment|cutout|compression|combo}` – Augmentation strategy

Also supports subset training:
```bash
python scripts/run_cnn.py --train-subset-ratio 0.3 --patience 15
```

### EfficientNet baseline

```bash
python scripts/run_efficientnet.py --out-dir outputs/efficientnet

# Example: train on exactly 10000 samples
python scripts/run_efficientnet.py --out-dir outputs/efficientnet --train-subset-size 10000
```

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

Custom output root:

```bash
python scripts/run_all.py --out-root outputs_full
```

Run full pipeline on subset of train data:

```bash
python scripts/run_all.py --train-subset-ratio 0.3
```



## 5. TensorBoard

```bash
tensorboard --logdir outputs
```

## 6. Notes

- CINIC-10 official statistics are used for normalization.
- AutoAugment uses CIFAR10 policy as a practical approximation for CINIC-10.
- The script scripts/run_all.py provides an orchestration template and should be adjusted to your selected best CNN checkpoint path.
