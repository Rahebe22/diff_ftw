# Overview

The [Mapping Africa](https://mappingafrica.io) project has developed models and data geared towards field boundary mapping in various countries in Africa. The largest of these efforts was through the Lacuna Fund-based project led by [Farmerline](https://farmerline.co/) and collaboration with [Spatial Collective](https://spatialcollective.com/) to develop [A Region-Wide, Multi-Year Set of Field Boundary Labels for Africa](https://github.com/agroimpacts/lacunalabels) (the Lacuna labels), which are now hosted on both the [Registry of Open Data on AWS](https://registry.opendata.aws/africa-field-boundary-labels/) and [Zenodo](https://zenodo.org/records/11060871).

In addition to that dataset, an additional set of \~5000 labels collected through various Mapping Africa project activities are to be made available. These data were combined with the Lacuna labels to train a modified U-Net model (described in [Khallaghi et al, 2025](https://www.mdpi.com/2072-4292/17/3/474)) that has been applied to map several countries, including Zambia, Tanzania, Angola, Ghana, and Nigeria.

The goal of this project is to integrate these datasets and models with the broader [Fields of the World](https://github.com/fieldsoftheworld) project. This integration will revolve around two primary efforts:

1.  Integrate the Lacuna+ labels with the existing FTW labels.
2.  Train and evaluate models using various combinations of the integrated datasets. Models will include the existing Mapping Africa U-Net as well as FTW's U-Net variant, and potentially others.

## Set-up

We require `ftw-tools` to be installed (currently part of [ftw-baselines](https://github.com/fieldsoftheworld/ftw-baselines?tab=readme-ov-file#download-the-ftw-baseline-dataset)), which requires python 3.10-3.12. Using `pyenv` to manage and set up the environment:

``` bash
pyenv install -v 3.12.10
pyenv virtualenv 3.12.10 ftw-mapafrica
pyenv activate ftw-mapafrica
python -m pip install --upgrade pip
```

And then run `pip install -e .` to install the package in editable mode.

## Datasets

We retrieved the Mapping Africa/Lacuna+ labels from our own HPC storage, and the FTW dataset using the FTW cli.

``` bash
ftw download 
ftw data download -o ~/data/labels/cropland/
```

The Mapping Africa/Lacuna+ (hereafter MA) labels were resampled to 256x256. The image bands were re-ordered to RGB-NIR from their existing BGR-NIR order to be consistent with FTW. The 3-class label masks were reprocessed to have a 1-pixel wide field edge class, in keeping with the FTW labels.

A unified CSV catalog was created that provides the following information:

| Variable    | Description                                                                                                                                                                                                                   |
|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| name        | FTW AOI ID and Lacuna+ grid identifier                                                                                                                                                                                        |
| dataset     | ftw or mappingafrica                                                                                                                                                                                                          |
| version     | 1.0 for FTW; 1.3.0 for labels collected under other Mapping Africa projects; 2.0.0 for labels from Lacuna Fund project                                                                                                        |
| country     | Full names for FTW, abbreviations for MA                                                                                                                                                                                      |
| x           | Longitude in decimal degrees                                                                                                                                                                                                  |
| y           | Latitude in decimal degrees                                                                                                                                                                                                   |
| fld_prop    | Proportion of image covered by field classes (interior + edge)                                                                                                                          |
| nonfld_prop | Proportion of image covered by non-field/background class                                                                                                                               |
| null_prop   | Proportion of image covered by unknown (3) class (FTW only)                                                                                                                             |
| window_a    | Partial path and name for image collected during the local dry season/end of season time period under the FTW scheme. This is the only time image time point available at present for MA.                                     |
| window_b    | Path and name of early growing season image for FTW, not available for MA.                                                                                                              |
| mask        | Path and name of 3-class mask. For FTW, the mask filename (as with image names) is formed from the AOI ID. For MA, the mask file name consists of `<name>*<assignment_id>*<year>-<month>.*` (year/month match window_a image). |
| split       | train, validate, or test                                                                                                                                                                                                      |

The image and mask names have partial paths to each that provide the respective sub-folder structures particular to each dataset, i.e.:

-   For FTW:

    -   images: `ftw/<country>/s2_images/<window_a|window_b>`

    -   masks: `ftw/<country>/label_masks/semantic_3class`

-   MA is simpler: `mappingafrica-256/<images|labels>`

So the two datasets should be downloaded into a single common folder to facilitate their combined use.

To access the MA dataset, download it using the AWS CLI:

``` bash
cd /path/to/your/common/label/directory
aws s3 sync s3://africa-field-boundary-labels/mappingafrica-256/ . --dryrun
aws s3 sync s3://africa-field-boundary-labels/mappingafrica-256/ .
```

If the `--dryrun` variant shows a successful download, run the final line to download the data into the same folder holding the FTW dataset.

## Working with the data

Data classes based on those in `ftw-baselines` and `torchgeo` are used here, with modifications to provide additional augmentations and to read from a combined catalog file. See the [data-modules.ipynb](notebooks/data-modules.ipynb) for additional details.

## Diffusion self-supervised pretraining

The diffusion pipeline pretrains an image encoder without requiring field-boundary labels. It learns a denoising task on unlabeled NICFI image chips, then exports the learned EfficientNet-B7 encoder for supervised field-boundary mapping.

The large-scale experiment is configured in [`configs/custom/diff-ftw-224-ssl-300ep-es5.yaml`](configs/custom/diff-ftw-224-ssl-300ep-es5.yaml). The current AWS run uses approximately 29.3 million four-channel, 224 x 224 NICFI chips staged on EBS and trained with eight CUDA devices on an AWS P3 instance.

### Method at a glance

| Component | Current choice | Why it is used |
|-----------|----------------|----------------|
| Pretraining task | DDPM-style noise prediction | Uses unlabeled satellite imagery by asking the model to predict synthetic Gaussian noise added to each chip. |
| Image encoder | EfficientNet-B7 | Matches the downstream FTW/Mapping Africa segmentation backbone, so encoder weights transfer directly. |
| Denoiser | U-Net with EfficientNet-B7 encoder | Combines local boundary detail with multiscale agricultural texture. |
| Timestep conditioning | FiLM layers on encoder features and decoder blocks | Gives the denoiser explicit information about the current noise level at multiple feature scales. |
| Input scaling | Normalized to `[0, 1]`, then centered to `[-1, 1]` inside the diffusion task | Aligns image values with the zero-centered Gaussian noise process. |
| Noise schedule | 1000-step cosine schedule | Preserves signal more smoothly across the noising trajectory than the original linear schedule. |
| Active timestep range | `100-999` for the AWS run | Avoids the near-clean timestep regime that produced unstable raw validation losses while contributing little useful denoising signal. |
| Loss | Min-SNR weighted MSE, `gamma = 5.0` | Reduces domination by very easy high-SNR timesteps. |
| Validation weights | EMA model | Evaluates and exports smoother weights than the instantaneous online model. |
| Transfer output | `encoder_ema.pt` | Strict encoder-only checkpoint used by the supervised field-boundary model. |

### Canonical references

This implementation combines established ideas; it is not a verbatim reimplementation of a single paper.

| Idea used here | Reference | Authors | Model or method | Year | How it appears in this repo |
|----------------|-----------|---------|-----------------|------|-----------------------------|
| Diffusion modeling foundation | [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) | Sohl-Dickstein, Weiss, Maheswaranathan, Ganguli | Diffusion probabilistic modeling | 2015 | Conceptual forward corruption and learned reverse denoising process. |
| DDPM noise prediction | [Denoising Diffusion Probabilistic Models](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html) | Ho, Jain, Abbeel | DDPM | 2020 | Random timestep sampling and MSE prediction of the Gaussian noise. |
| Cosine schedule | [Improved Denoising Diffusion Probabilistic Models](https://proceedings.mlr.press/v139/nichol21a.html) | Nichol, Dhariwal | Improved DDPM | 2021 | 1000-step cosine beta schedule with `cosine_s = 0.008`. |
| Min-SNR loss weighting | [Efficient Diffusion Training via Min-SNR Weighting Strategy](https://arxiv.org/abs/2303.09556) | Hang et al. | Min-SNR-g weighting | 2023 | Timestep loss weights with `gamma = 5.0`. |
| Multiscale denoiser | [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) | Ronneberger, Fischer, Brox | U-Net | 2015 | Encoder-decoder denoiser with skip-connected multiscale features. |
| Encoder backbone | [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://proceedings.mlr.press/v97/tan19a.html) | Tan, Le | EfficientNet | 2019 | EfficientNet-B7 encoder retained for downstream segmentation transfer. |
| Feature-wise timestep conditioning | [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871) | Perez et al. | FiLM | 2018 | Timestep embedding produces feature-wise scale and shift values. |
| AWS P3 hardware | [Amazon EC2 P3 Instances](https://aws.amazon.com/about-aws/whats-new/2017/10/introducing-amazon-ec2-p3-instances/) | AWS | P3 / p3.16xlarge | 2017 | Eight NVIDIA Tesla V100 GPUs used for the large run. |

### Input and output

The SSL dataset reads image-only chips from a CSV catalog. Each row points to one raster image.

```text
input image on disk
  -> GeoTIFF / COG chip
  -> 4 channels: RGB-NIR
  -> 224 x 224 pixels
  -> no boundary label required
```

The data module normalizes imagery to finite `[0, 1]` tensors. Invalid pixels, negative values, infinities, and configured nodata values are ignored during normalization statistics and written back as finite zeros. The diffusion task then centers the batch to `[-1, 1]` before adding noise.

```text
DataLoader output:     image in [0, 1]
Diffusion task input:  x_0 = 2 * image - 1, so x_0 is in [-1, 1]
Training target:       sampled Gaussian noise epsilon
Model output:          predicted 4-channel noise epsilon_theta(x_t, t)
Final artifact:        EMA EfficientNet-B7 encoder checkpoint
```

The useful training artifact is:

```text
/home/ubuntu/working/models/diffusion_ssl_300ep_es5/encoder_ema.pt
```

That file contains only the EMA EfficientNet encoder state dictionary and is intended for supervised field-boundary segmentation fine-tuning.

### Diffusion objective

For each batch, the task samples a timestep `t`, samples Gaussian noise `epsilon`, and creates a noisy image `x_t`:

```text
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
```

The model receives `(x_t, t)` and predicts `epsilon`:

```text
loss_i = weight_t * mean((epsilon - epsilon_theta(x_t, t))^2)
```

With Min-SNR weighting:

```text
SNR_t    = alpha_bar_t / (1 - alpha_bar_t)
weight_t = min(SNR_t, gamma) / SNR_t
```

The current run uses `gamma = 5.0`. This makes the loss less dominated by nearly clean, high-SNR timesteps and gives the model a more balanced denoising curriculum across the 1000-step schedule.

This repository currently uses a discrete DDPM-style objective. It does not currently implement EDM continuous noise sampling, latent diffusion, learned reverse variance, or a full reverse sampler for image generation. The purpose here is representation learning for boundary mapping, not producing synthetic satellite images.

For the large AWS run, the active timestep range starts at `t = 100` instead
of `t = 0`. Very low timesteps are almost clean images, so the noisy input
contains very little information about the random Gaussian target noise. With
Min-SNR weighting, those examples are also heavily downweighted during
training. In practice, this made the first validation bin unstable while later
denoising timesteps remained healthy. Since the goal is encoder pretraining,
not a full image-generation sampler, the production SSL run focuses on
meaningful denoising timesteps `100-999`.

### Model architecture

The diffusion model is implemented in [`ftw_ma/diffusion_task.py`](ftw_ma/diffusion_task.py) as `FTWEfficientNetDiffusionModel`.

```text
noisy 4-channel chip x_t
  -> unchanged EfficientNet-B7 encoder
  -> multiscale encoder feature maps
  -> timestep FiLM on consumed encoder scales
  -> U-Net decoder with timestep FiLM on decoder blocks
  -> 4-channel predicted noise
```

The underlying U-Net is created with [`segmentation_models_pytorch`](https://github.com/qubvel-org/segmentation_models.pytorch):

```text
encoder_name: efficientnet-b7
encoder_weights: imagenet
in_channels: 4
classes: 4
```

The EfficientNet-B7 encoder architecture is intentionally left untouched. The diffusion-specific information is added through FiLM conditioning around the U-Net feature maps, not by changing the encoder blocks themselves.

FiLM applies a feature-wise scale and shift:

```text
FiLM(feature, t) = feature * (1 + scale(t)) + shift(t)
```

The timestep path is:

```text
timestep t
  -> sinusoidal timestep embedding, 128 dimensions
  -> MLP, 128 -> 512
  -> FiLM scale and shift parameters
```

The first raw input-like encoder feature is left unconditioned because the SMP U-Net decoder does not consume it. SMP also keeps EfficientNet classification-tail layers in the encoder object, but those layers are not executed in the U-Net feature path; they are frozen during diffusion pretraining to avoid unused trainable parameters in DDP.

### EMA validation and encoder export

Training maintains two model copies:

```text
online model: updated by AdamW
EMA model:    theta_ema = decay * theta_ema + (1 - decay) * theta_online
```

The default EMA decay is `0.9999`. Validation uses the EMA model, and training end exports the EMA encoder.

Validation reports:

- `val/loss`: EMA denoising loss used by checkpointing and early stopping.
- `val/loss_t_0000_0099` through `val/loss_t_0900_0999`: timestep-binned validation loss.
- Best-sample diagnostics under `<default_root_dir>/best_samples`: clean image, noisy image, true noise, predicted noise, and reconstructed image.

The encoder export is validated by [`ftw_ma/checkpoints.py`](ftw_ma/checkpoints.py). Downstream supervised training should load it strictly so architecture or tensor-shape mismatches fail loudly.

```yaml
model:
  init_args:
    model: unet
    backbone: efficientnet-b7
    in_channels: 4
    weights: /home/ubuntu/working/models/diffusion_ssl_300ep_es5/encoder_ema.pt
```

### What changed from the first diffusion baseline

| Area | First baseline | Current pipeline |
|------|----------------|------------------|
| Encoder | EfficientNet-B7 | EfficientNet-B7, unchanged |
| Timestep context | Added once near the image input | FiLM conditioning at consumed encoder scales and decoder blocks |
| Input range | `[0, 1]` | DataLoader returns `[0, 1]`; diffusion task centers to `[-1, 1]` |
| Noise schedule | Long linear schedule | 1000-step IDDPM-style cosine schedule |
| Loss | Uniform MSE | Min-SNR weighted MSE with `gamma = 5.0` |
| Validation model | Online weights | EMA weights |
| Validation detail | Aggregate loss | Aggregate loss plus ten timestep bins and image diagnostics |
| Transfer artifact | Full checkpoint emphasis | Strict EMA encoder-only export |
| Data path | Mounted S3 | Full large run staged on EBS for faster small-file reads |

### Pretraining data

The full pretraining catalog contains 29,258,844 NICFI image chips. The current training catalog uses a reproducible 80/20 split in the `usage` column:

| Split | Chips |
|-------|------:|
| Train | 23,408,306 |
| Validate | 5,850,538 |
| Total | 29,258,844 |

The current AWS full-run config expects the copied EBS dataset:

```text
data_dir:          /mnt/ebs_pretrain
catalog:           /mnt/ebs_pretrain/catalog_train_validate_80_20.csv
catalog_cache_dir: /mnt/ebs_pretrain/ssl_catalog_cache
```

The mounted S3 dataset at `/mnt/s3_pretrain` is useful as the source of truth, but it was too slow for sustained random small-file training. The large run stages the imagery on a 10,000 GiB gp3 EBS volume mounted at `/mnt/ebs_pretrain`. In the EBS smoke and benchmark runs, first-image reads were about `0.04s`, and first validation batches were ready in about `0.33s`.

Build the local catalog cache once before the full run:

```bash
python scripts/build_ssl_catalog_cache.py \
  --catalog /mnt/ebs_pretrain/catalog_train_validate_80_20.csv \
  --output-dir /mnt/ebs_pretrain/ssl_catalog_cache
```

This cache avoids repeatedly expanding tens of millions of paths into Python objects across DDP ranks.

### GPU and runtime environment

The large diffusion experiment has been run on an AWS P3 eight-GPU instance. AWS lists the `p3.16xlarge` configuration as eight NVIDIA Tesla V100 GPUs, 64 vCPUs, 488 GiB memory, 25 Gbps network bandwidth, and 14 Gbps EBS bandwidth.

The full YAML is configured for:

```text
accelerator: cuda
devices: [0, 1, 2, 3, 4, 5, 6, 7]
strategy: ddp_find_unused_parameters_false
use_distributed_sampler: false
batch_size: 20 per rank
num_workers: 1 per rank
prefetch_factor: 1
pin_memory: true
```

Measured safe-run checkpoints from the AWS/EBS setup:

| Run | Setting | Observed result |
|-----|---------|-----------------|
| Short benchmark | 2000 train batches, 20 validation batches | 11 min 48 sec, about 2.84 it/s, `train/loss = 0.335`, `val/loss = 0.989` |
| Longer benchmark | 20000 train batches | 1 hr 56 min, about 2.90 it/s, `train/loss = 0.124`, `val/loss = 1.094` |
| Full epoch | 146,301 optimizer steps over the full train split | 14 hr 12 min, about 2.82 it/s, `train/loss = 0.054`, `val/loss = 0.462` |

The stable full-epoch run used safer runtime overrides than the YAML defaults:

```text
precision: 32-true
learning rate: 5e-5
gradient_clip_val: 1.0
```

These overrides were used after an earlier full-epoch attempt with mixed precision and `lr = 2e-4` produced NaN losses. The normalizer and dataloader checks showed finite image batches, so the safer run treats this as an optimization-stability issue rather than a confirmed data-corruption issue.

### How to run on AWS

Activate the environment and install the repo in editable mode:

```bash
cd /home/ubuntu/diff_ftw
source /home/ubuntu/venvs/ftw_ma312/bin/activate
pip install -e .
```

Confirm the EBS data path exists:

```bash
df -h /mnt/ebs_pretrain
ls /mnt/ebs_pretrain/catalog_train_validate_80_20.csv
ls /mnt/ebs_pretrain/ssl_catalog_cache
```

Run the tiny smoke test first. The smoke YAML intentionally uses the tiny
repo smoke catalog and may still point at `/mnt/s3_pretrain`; it is for DDP
and model-graph debugging, while the full run below uses EBS.

```bash
python run_lightning_fit.py fit \
  -c configs/custom/diff-ftw-smoke.yaml \
  --trainer.precision=32-true \
  --trainer.gradient_clip_val=1.0 \
  --model.lr=5e-5
```

Run a short full-catalog benchmark:

```bash
python run_lightning_fit.py fit \
  -c configs/custom/diff-ftw-224-ssl-300ep-es5.yaml \
  --trainer.limit_train_batches=2000 \
  --trainer.limit_val_batches=20 \
  --trainer.max_epochs=1 \
  --trainer.precision=32-true \
  --trainer.gradient_clip_val=1.0 \
  --model.lr=5e-5
```

Start the 300-epoch run in `tmux` so it survives disconnection:

```bash
tmux new -s diffusion300_ebs

python run_lightning_fit.py fit \
  -c configs/custom/diff-ftw-224-ssl-300ep-es5.yaml \
  --trainer.precision=32-true \
  --trainer.gradient_clip_val=1.0 \
  --model.lr=5e-5
```

Detach from `tmux` with `Ctrl-b`, then `d`. Reattach later with:

```bash
tmux attach -t diffusion300_ebs
```

To continue from a good diffusion checkpoint with a fresh optimizer, use
`init_from_checkpoint` instead of `--ckpt_path`. This loads model and EMA
weights only, while the new run uses the current config, learning rate, and
active timestep range.

```bash
python run_lightning_fit.py fit \
  -c configs/custom/diff-ftw-224-ssl-300ep-es5.yaml \
  --trainer.precision=32-true \
  --trainer.gradient_clip_val=1.0 \
  --trainer.sync_batchnorm=true \
  --model.lr=1e-5 \
  --model.init_from_checkpoint=/home/ubuntu/working/models/diffusion_ssl_300ep_es5/lightning_logs/version_X/checkpoints/epoch=1-val_loss=0.0428.ckpt
```

### Expected outputs

The full run writes under:

```text
/home/ubuntu/working/models/diffusion_ssl_300ep_es5
```

Important outputs are:

| Output | Meaning |
|--------|---------|
| `encoder_ema.pt` | EMA EfficientNet-B7 encoder for supervised transfer |
| `last.ckpt` | Resume checkpoint |
| best `epoch-*.ckpt` | Best checkpoint by `val_loss` |
| `best_samples/*.png` | Validation diagnostic images |
| Lightning logs | Training loss, validation loss, Min-SNR weights, timestep-binned losses |

### Relevant diffusion files

| File | Role |
|------|------|
| [`ftw_ma/diffusion_task.py`](ftw_ma/diffusion_task.py) | Denoiser, FiLM conditioning, weighted objective, EMA validation, diagnostics, and encoder export |
| [`diffusion/scheduler.py`](diffusion/scheduler.py) | Cosine schedule, forward noising process, timestep embeddings, SNR calculation, and Min-SNR weights |
| [`ftw_ma/normalize.py`](ftw_ma/normalize.py) | Finite image normalization and nodata handling |
| [`ftw_ma/ssl_datamodule.py`](ftw_ma/ssl_datamodule.py) | Image-only SSL dataset, memory-mapped catalog cache, constant-memory sampler, and loader diagnostics |
| [`scripts/build_ssl_catalog_cache.py`](scripts/build_ssl_catalog_cache.py) | One-time path-cache builder for the large SSL catalog |
| [`ftw_ma/checkpoints.py`](ftw_ma/checkpoints.py) | Strict encoder-export checkpoint format |
| [`run_lightning_fit.py`](run_lightning_fit.py) | Standalone Lightning DDP launcher for multi-GPU training |
| [`tests/test_diffusion_pipeline.py`](tests/test_diffusion_pipeline.py) | Schedule, weighting, transfer, forward-pass, and no-unused-parameter regression tests |
| [`tests/test_normalize.py`](tests/test_normalize.py) | NaN, infinity, negative-value, and nodata normalization tests |
| [`tests/test_ssl_catalog_cache.py`](tests/test_ssl_catalog_cache.py) | Catalog cache and loader behavior tests |

## Supervised training and evaluation

From the CLI, the model can be trained as follows:

```bash
ftw_ma model fit -c configs/<config-file>.yaml
```

See the [example config](configs/initial/example-config.yaml) for settings.

`<config-file>.yaml` should be named to be informative of the experiment, e.g. `fullcat-ftwbaseline-exp2.yaml` for the second experiment using the full combined catalog and FTW Baseline model.

To resume training from a specific checkpoint:

```bash
CKPT=/path/to/checkpoint/checkpoint.ckpt
ftw_ma model fit -c configs/<config-file>.yaml --ckpt_path $CKPT
```

To test the model:

```bash
CHKPT=/path/to/checkpoint/checkpoint.ckpt 
ftw_ma model test -c configs/config.yaml -m $CHKPT --gpu 0 -o metrics.json
```

Or run the `tester.sh` script:

```bash
# from project root
./scripts/tester.sh <model_dir_name> <version_number> <catalog>
```

`<model_dir_name>` is the name of the model directory under `~/working/models/`, e.g. `fullcat-ftwbaseline-exp2` where the model checkpoint is stored. `<version_number>` is the version number of the training run, specified explicitly as an integer, or if left empty the latest version is found and run. `<catalog>` is the path to the catalog CSV file.

This will produce an output metrics file in a specified directory, with a file name composed of the experiment name and catalog used in testing. The script is currently hard-coded for the validation split.
