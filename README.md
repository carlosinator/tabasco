<div align="center">


<img src="figures/tabasco_logo.png" width="600">

<h3 align="center">
  A Fast, Simplified Model for Molecular Generation with Improved Physical Quality
</h3>
<p align="center">
    <a href="https://carlosinator.github.io">Carlos Vonessen</a>*,
  <a href="https://cch1999.github.io">Charles Harris</a>*,
  <a href="https://scholar.google.com/citations?user=KgA_dpsAAAAJ&hl=en">Miruna Cretu</a>*,
  <a href="https://www.cl.cam.ac.uk/~pl219/">Pietro Liò</a><br><br>
  GenBio Workshop @ ICML 2025, *Core contributor
</p><br>

[![arXiv](https://img.shields.io/badge/PDF-arXiv-red)](https://arxiv.org/pdf/2507.00899)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
<br>
<!--
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)
-->
<br>

</div>

<p align="center">
<img src="figures/pareto_front.png" width="60%">
</p>


## Main Contributions:
* [State-of-the-art performance on PoseBusters](paperswithcode.com)
* 10x speed-up at sampling time (see Table 1)
* More parameter efficient (see Figure 1)
* Standard non-equivariant Transformer
* Lean and extensible implementation


## Getting Started

**Introduction to Repo:** This repository is based on the [lightning hydra template](https://github.com/ashleve/lightning-hydra-template), where you can find an introduction on hydra for pytorch and general usage instructions.

**Downloading datasets:** The processed datasets are available for [GEOM-Drugs](https://huggingface.co/datasets/carlosinator/tabasco-geom-drugs) and [QM9](https://huggingface.co/datasets/carlosinator/tabasco-qm9). Move all splits to `src/data` without renaming. Running `src/train.py` for the first time will generate the lmdb dataset, which only happens once and can take about an hour.

### Installation

```bash
conda env create -f environment.yaml
conda activate tabasco
```

### Training

The training configs are available under `configs/experiment`, which overwrite the defaults in the other `configs/*` folders. To train the `TABASCO-hot` model from the paper, you can run:

```python
python src/train.py experiment=hot_geom trainer=gpu
```

**Multi-GPU Training** is available via `torchrun` and trainer parameters are customizable in `configs/trainer`. You may want to pass additional command line arguments to `torchrun` depending on your setup. For example for two GPUs on one node using DDP (assuming a suitable `ddp.yaml` config) you can run

```python
torchrun --nproc_per_node=2 --nnodes=1 src/train.py experiment=hot_geom trainer=ddp
```

### Sampling
We provide two scripts for sampling from a model checkpoint, as well as some convenient parameters to modify. Unconditional sampling is called with:
```python
python src/sample.py \
    --num_mols 1000 --num_steps 100 \
    --checkpoint path/to/model.ckpt \
    --output_path path/to/output/folder
```

**Boosting Physical Plausibility**: This is a script for sampling molecules with boosted physical quality (Section 3.5). Where `guidance` encodes the step size of each gradient step, `step-switch` the point at which to switch to UFF bound guidance, and `to-center` whether to regress to the interval center.

```python
python src/sample_uff_bounds.py \
    --guidance 0.01 --step-switch 90 --to-center False \
    --ckpt path/to/model.ckpt --output-dir path/to/output/folder
```

## Repository Summary

<p align="center">
<img src="figures/main_figure.png" width="80%">
</p>

### Model Architecture

The model uses a deliberately simplified non-equivariant Transformer that treats molecular generation as a sequence modeling problem (see the [positional encodings](src/pocketsynth/models/components/positional_encoder.py)). Coordinates and atom types are jointly embedded with time and positional encodings, then processed through standard [Transformer blocks](src/pocketsynth/models/components/transformer.py). No explicit bond information is included and the model relies on generating physically sensible coordinates so that standard chemoinformatics tools can infer bonds reliably. Optional cross-attention layers allow separate processing of coordinate and atom type domains before final MLP heads predict the outputs. The full [model implementation](src/pocketsynth/models/components/transformer_module.py) is easily extensible compared to specialized equivariant architectures.

### Interpolant Class

We combine the required interpolant functionality in one base `Interpolant` class to make the code more readable and extensible. In practice, we found that this significantly increases iteration speed and improves verifiability. The `SDEMetricInterpolant` manages coordinate flows with configurable noise scaling and centering, while `DiscreteInterpolant` handles categorical atom types in the discrete diffusion framework. Each interpolant defines four key operations: noise sampling, path creation between data points, loss computation, and explicit-Euler stepping during generation. This modular design allows mixing different interpolation strategies for different molecular properties while maintaining a unified training loop.


## Citing this work

```
# todo
```
