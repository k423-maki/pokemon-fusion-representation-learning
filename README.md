# Pokémon Fusion Representation Learning

**Yi Yao · PyTorch course project**

This project explores image reconstruction and the information retained in a variational autoencoder's latent representation. It includes a convolutional VAE baseline, a ResNet18 encoder with an auxiliary classifier, and a PCA/logistic-regression probe. The original course experiments used 18,900 RGB Pokémon fusion images at 128 × 128 resolution with 170 type-combination labels.

My technical work involved the PyTorch data pipeline, VAE architecture and loss/training experiments, and classification-based analysis of the latent representation. The archived notebooks document these experiments. The course dataset and any starting materials are not claimed as my own dataset creation.

## Approach

- **Convolutional VAE:** convolutional encoder, spatial Gaussian bottleneck, reparameterization and convolutional decoder. Training combines reconstruction MSE and KL divergence with linear KL annealing. The baseline uses an 8 × 32 × 32 latent tensor.
- **ResNet18 prototype:** pretrained encoder through `layer3`, spatial Gaussian bottleneck and convolutional decoder. An auxiliary classifier predicts the **input image's type-combination label from its latent tensor**. It does not evaluate the visual quality of generated images.
- **Representation probe:** flatten latent means, fit PCA on training examples, and use logistic regression to examine whether the representation retains label information.

## Historical evidence and limits

The baseline notebook retains a 1,000-epoch training log. The later prototype retains only epoch 1 of a requested 20-epoch run, so this repository does not claim that the complete prototype training finished.

The saved probe output reports 4,096 input features, 256 PCA components and validation accuracy **0.0995 (9.95%)**. This is an exploratory archived result. Its exact split and matching checkpoint were not recovered, and the saved feature dimension differs from the notebook's final 32-channel instantiation (2,048 features). The notebooks were edited interactively; these outputs therefore cannot establish an exact reproducible configuration. The portable CLI uses the explicitly selected configuration and records split indices for new runs.

The saved `Validation MSE: 805.862802` is squared error **per image**, summed over pixels and channels. It should not be compared directly with per-pixel MSE. No full-data retraining was performed while preparing this public repository.

## Run a CPU software check

Use Python 3.11+ and run commands from this directory. Create and activate a virtual environment, then:

```console
python -m pip install -r requirements.txt
python make_toy_data.py
python run.py smoke
```

`make_toy_data.py` creates 64 deterministic synthetic images containing colored squares and noise. Their eight numeric labels are arbitrary. These are **software fixtures, not Pokémon data or a model-quality benchmark**. No course images, full dataset or pretrained project checkpoints are distributed here. The smoke check initializes both architectures without downloaded pretrained weights and checks shapes, finite losses/gradients, an optimizer update and deterministic evaluation.

For a short training and probe example:

```console
python run.py train --data data/sample.npz --model baseline --epochs 1 --batch-size 4 --max-batches 2 --output runs/example
python run.py probe --data data/sample.npz --checkpoint runs/example/model.pt --components 8 --output runs/example-probe
```

The training command writes a checkpoint, history and reconstruction grid. The probe fits PCA and logistic regression only on the training split. Scores on this tiny synthetic fixture are not evidence of performance on the original task. Exact versions used for software verification are in `requirements-tested-cpu.txt`.

## Running with authorized course data

The loader accepts an `.npz` file containing `images` (uint8, shape `(N, 3, 128, 128)`) and `labels` (integer vector, values 0–169). Supply your own authorized copy; it is not included in the repository.

```console
python run.py train --data PATH_TO/train.npz --model baseline --latent-channels 8 --epochs 1000 --batch-size 128 --learning-rate 0.001 --all-training-data --output runs/baseline
python run.py train --data PATH_TO/train.npz --model resnet --latent-channels 32 --pretrained --epochs 20 --batch-size 128 --output runs/resnet
python run.py probe --data PATH_TO/train.npz --checkpoint runs/resnet/model.pt --components 256 --output runs/probe
```

These are commands for new experiments, not a claim to reproduce the archived scores. Full training is computationally expensive. `--pretrained` may download torchvision's ResNet18 weights. The default training split is seeded 90/10; checkpoints record the dataset hash, model configuration and split indices. The probe checks the dataset hash before reusing the split. `--all-training-data` disables the holdout and cannot be used for a validation probe.

## Repository guide

| File | Purpose |
|---|---|
| `models.py` | Portable baseline and ResNet/classifier architectures |
| `run.py` | Data loading, smoke check, training, reconstruction export and representation probe |
| `make_toy_data.py` | Synthetic fixture generator for running without course data |
| `archive/baseline_1000_epochs.ipynb` | Historical baseline code and text training outputs |
| `archive/resnet_classifier_prototype.ipynb` | Historical prototype code and saved diagnostic outputs |
| `checks/` | Results and scope of public-package software verification |
| `PROVENANCE.json` | Preparation dates, original notebook hashes and changes |

## Preparation and attribution

The historical notebooks are records of exploration, not recommended “Run All” entry points: some helper cells contain stale dimensions or inconsistent output unpacking. Public copies retain code and text outputs while omitting embedded images, HTML and error traces.

`models.py` and `run.py` were organized from the notebook architectures with AI assistance on **23 November 2026**. The portable CLI, seeded splits and explicit metrics are preparation additions rather than claims about the original course work. The synthetic fixture generator and public documentation were added with AI assistance on **24 November 2026**. During the portable prototype warm-up, frozen decoder blocks also use evaluation mode to freeze BatchNorm buffers; this corrects the exploratory notebook's behavior and can change new results. The original unused scheduler is not presented as an active training method.

Pokémon characters and course-provided materials belong to their respective creators and providers. This repository shares project source for portfolio review and does not redistribute the course image dataset or assert a new license over third-party assets.
