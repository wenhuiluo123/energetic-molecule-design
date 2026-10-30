# Energetic Molecule Design

Research code for Transformer-based molecular generation and property prediction, including generator pretraining and fine-tuning, 3D graph and SMILES representations, multimodal fusion, knowledge distillation, and reinforcement learning.

## Project Structure

```text
generator/       Molecular generator, training scripts, and configurations
predictor/       Property prediction and distillation models
  gpt/
  graph/
  multimodal/
  graph_to_smiles/
  multimodal_to_smiles/
  common/
  3D_GNN/
RL/              Reinforcement learning, rewards, and metrics
SA_Score/        Synthetic accessibility scoring
data/            Datasets and graph caches
```

## Installation

Requires Python 3.10 or later. Main dependencies include PyTorch, Lightning, and RDKit.

```bash
pip install -r requirements.txt
```

## Data Preparation

The generator reads a CSV file containing a `smiles` column. Predictors use `data/metadata.csv`, which contains `molecule_id`, `smiles`, target property columns, and `split`. Split labels are `train`, `valid`, and `test`. Specify target columns through `target_columns` in each predictor configuration.

```bash
python -m predictor.prepare_data --input_csv data/properties.csv --targets D r0 --split_ratios 8 1 1 --seed 42
python -m predictor.build_graph_cache --metadata data/metadata.csv --output_dir data/graph_cache
```

Split ratios are ordered as training, validation, and test. Different SMILES representations of the same molecule remain in the same split. A fixed random seed makes the split reproducible. Graph cache files are named by `molecule_id`, and all predictors share the same metadata.

## Training

Run all commands from the project root. Dataset paths, model parameters, and training settings are defined in each module's YAML configuration. Example entry points:

```bash
python -m generator.train --config generator/configs/base.yaml
python -m predictor.gpt.train --config predictor/gpt/configs/D_gpt.yaml
python -m predictor.graph.train --config predictor/graph/configs/D_graph.yaml
python -m RL.train --config RL/configs/base.yaml
```

Multimodal and distillation training scripts are located in their respective model directories. Distillation requires the baseline and teacher checkpoints specified in the configuration. RL loads a trained generator and two final student predictors. Configure the task-specific reward function in `RL/rewards.py` before use.

The generator supports distributed data parallel (DDP) training. GPU selection and early stopping are configurable. For fine-tuning, set `pretrained_checkpoint` to load a pretrained model and reuse its vocabulary and architecture.

## Model Files

Training outputs are saved under `outputs/`. Generator checkpoints include the vocabulary. When transferring a predictor, keep its checkpoint together with `tokenizer.json`, `stats.json`, `data_schema.json`, and `resolved_config.yaml` from the training directory.

See [3D_GNN](predictor/3D_GNN/README.md) for pretrained graph models and source information, and [SA_Score](SA_Score/README) for synthetic accessibility scoring attribution.
