# Spec2PSM

A Framework for deep learning transformer encoder/decoder models to predict peptide sequence from spectra. 
This repo is a work in progress currently

<br><br>
<img src="./static/spec2PSM.png" alt="logo" width="300" style="display: block; margin: 0;" />
<br><br>

# Spec2PSM Models
Pre trained Spec2PSM models can be found at HuggingFace Hub here: https://huggingface.co/hinklet

# Spec2PSM CLI Documentation

## Overview
`spec2psm` is a command-line tool for managing and running various modes of the Spec2PSM pipeline. It supports parsing spectra, training models, fine-tuning, validating models, and running inference.

## Usage
```bash
spec2psm [mode] [options]
```

## Available Modes

### Convert Mode
Convert a spectrum file and/or a PSM file using the provided configuration.

#### Command
```bash
spec2psm convert -c <config_path> -m <mzml_paths> [-s <search_paths>] [-f <fdr_paths>] [-u <mods_path>] [-o <output_directory>]
```

#### Arguments
- `-c`, `--config_path` (required): Path to the configuration file for parsing.
- `-m`, `--mzml_paths` (required): One or more mzML file paths to convert. Provide as a space-separated list for multiple paths.
- `-s`, `--search_paths` (optional): One or more search result file paths (mzid or pepXML). Provide as a space-separated list for multiple paths. Default: `None`.
- `-f`, `--fdr_paths` (optional): One or more Percolator result file paths. Requires both mzML and pepXML paths to be provided. Default: `None`.
- `-u`, `--mods_path` (optional): Path to a file specifying modification settings. Default: `None`.
- `-o`, `--output_directory` (optional): Directory to save the Parquet files. If not specified, the files will be saved in the same directory as the mzML files.
---

### Train Mode
Train a model using the specified training and validation datasets.

#### Command
```bash
spec2psm train -t <train_parquet_paths> [-v <val_parquet_paths>] [-o <output_model_name>] [-c <config_path>] [-d <device>]
```

#### Arguments
- `-t`, `--train_parquet_paths` (required): Directories or Parquet file paths for training.
- `-v`, `--val_parquet_paths` (optional): Directories or Parquet file paths for validation.
- `-o`, `--output_model_name` (optional): Path to save the trained model weights.
- `-c`, `--config_path` (required): Path to the fine-tuning configuration file. Default: `None`.
- `-d`, `--device` (required): Device for training (`mps`, `cpu`, or `gpu`). Default: `None`.

---

### Fine-Tune Mode
Fine-tune a pre-trained model.

#### Command
```bash
spec2psm tune -m <model> -t <train_parquet_paths> [-v <val_parquet_paths>] [-o <output_model_name>] [-c <config_path>] [-d <device>]
```

#### Arguments
- `-m`, `--model` (required): Path to the model weights or Hugging Face model name.
- `-t`, `--train_parquet_paths` (required): Directories or Parquet file paths for training.
- `-v`, `--val_parquet_paths` (optional): Directories or Parquet file paths for validation.
- `-o`, `--output_model_name` (optional): Path to save the fine-tuned model weights.
- `-c`, `--config_path` (required): Path to the fine-tuning configuration file. Default: `None`.
- `-d`, `--device` (required): Device for fine-tuning (`mps`, `cpu`, or `gpu`). Default: `None`.

---

### Validate Mode
Run validation on a model.

#### Command
```bash
spec2psm validate -m <model> -p <parquet_paths> [-c <config_path>] [-d <device>]
```

#### Arguments
- `-m`, `--model` (required): Path to the model weights or Hugging Face model name.
- `-p`, `--parquet_paths` (required): Directories or Parquet file paths for validation.
- `-c`, `--config_path` (required): Path to the inference configuration file. Default: `None`.
- `-d`, `--device` (required): Device for validation (`mps`, `cpu`, or `gpu`). Default: `None`.

---

### Infer Mode
Run inference using a model.

#### Command
```bash
spec2psm infer -m <model> -p <parquet_paths> [-c <config_path>] [-d <device>]
```

#### Arguments
- `-m`, `--model` (required): Path to the model weights or Hugging Face model name.
- `-p`, `--parquet_paths` (required): Directories or Parquet file paths for inference.
- `-c`, `--config_path` (required): Path to the inference configuration file. Default: `None`.
- `-d`, `--device` (optional): Device for inference (`mps`, `cpu`, or `gpu`). Default: `None`.

---

## Examples

### Convert Examples
#### Convert a single mzML file
```bash
spec2psm convert -c config.yaml -m /path/to/file1.mzML
```

#### Convert Multiple mzML Files
```bash
spec2psm convert -c config.yaml -m /path/to/file1.mzML /path/to/file2.mzML
```

#### Convert mzML with Search Files
```bash
spec2psm convert -c config.yaml -m /path/to/file1.mzML -s /path/to/file1.pepXML
```

#### Convert with FDR and Mods
```bash
spec2psm convert -c config.yaml -m /path/to/file1.mzML /path/to/file2.mzML \
-s /path/to/file1.pepXML /path/to/file2.pepXML \
-f /path/to/fdr_results1.tsv /path/to/fdr_results2.tsv \
-o /path/to/output_directory -u /path/to/mods_config.txt
```

### Train Example
#### Train with one parquet file
```bash
spec2psm train -c config.yaml -t /path/to/train_data.parquet -v /path/to/val_data.parquet -o /path/to/model_output.pt -d mps
```

#### Train with multiple parquet files
```bash
spec2psm train -c config.yaml -t /path/to/train_data1.parquet /path/to/train_data2.parquet /path/to/train_data3.parquet -v /path/to/val_data.parquet -o /path/to/model_output.pt -d mps
```

### Fine-Tune Example
```bash
spec2psm tune -c config.yaml -m huggingface-model-name -t /path/to/train_data.parquet -v /path/to/val_data.parquet -o /path/to/model_output.pt -d mps
```

### Validate Example
```bash
spec2psm validate -c config.yaml -m /path/to/model.pt -p /path/to/val_data.parquet -d cpu
```

### Infer Example
#### Infer one parquet file
```bash
spec2psm infer -c config.yaml -m /path/to/model.pt -p /path/to/data.parquet -d cpu
```

#### Infer multiple parquet file
```bash
spec2psm infer -c config.yaml -m /path/to/model.pt -p /path/to/data1.parquet /path/to/data2.parquet -d cpu
```

---

## Notes
- Ensure all paths are correct and accessible.
- Select the appropriate device (`mps`, `cpu`, `gpu`) based on your hardware.
- For more details or troubleshooting, refer to the tool's help command:
  ```bash
  spec2psm --help
  

## Configuration

For default configuration files for small, medium, and large models see spec2psm/config
For an example modification configuration file also see spec2psm/config - This is to add more modification tokens for file parsing, model training, validation, and inference
  
### Main Spec2PSM Configuration
Configuration Sections
1. Tokens

   - `max_peptide_length`: Maximum length of peptide sequences to be predicted.
   Default: `62`

   - `spectra_length`: Maximum length of the input spectra.
   Default: `150`

   - `row_group_size`: Number of rows in each group for creating or reading Parquet files, impacting file I/O performance.
   Default: `500`

   - `token_map_size`: Specifies the granularity of token mapping.
   Options: small, medium, large.
   Default: `medium`

2. Model

   - `batch_size`: Batch size for training and validation.
Default: `32`

   - `d_model`: Dimensionality of the model’s embedding space.
Default: `512`

   - `ff_dim`: Dimensionality of the feed-forward layer in encoder and decoder layers.
Default: `1024`

   - `dropout`: Dropout rate for regularization.
Default: `0.1`

   - `nheads`: Number of attention heads in the multi-head attention mechanism.
Default: `8`

   - `decoder_layers`: Number of decoder layers in the transformer.
Default: `4`

   - `encoder_layers`: Number of encoder layers in the transformer.
Default: `4`

3. Train

   - `lr`: Learning rate for optimizer.
Default: `1e-4`

   - `weight_decay`: Weight decay for regularization.
Default: `1e-5`

   - `warmup_steps`: Number of warmup steps for learning rate scheduling.
Default: `1000`

   - `total_model_steps`: Total number of steps for the learning rate decay schedule.
Default: `30,000,000`

   - `num_epochs`: Number of epochs to train.
Default: `1`

   - `metric_window`: Window size for calculating rolling averages of metrics during training.
Default: `100`

   - `plot_per_x_batches`: Frequency (in batches) of updating training metric plots.
Default: `1000`

   - `val_per_x_batches`: Frequency (in batches) of performing model validation.
Default: `250,000`

Example Medium Sized Configuration snippet:

```yaml
tokens:
  max_peptide_length: 62 # Max length of peptides to be predicted
  spectra_length: 150 # Max length of an input spectra
  row_group_size: 500 # Parameter used for creating / reading parquet files
  token_map_size: "medium" # Set to small, medium, or large - Each setting adds more or less modifications
model:
  batch_size: 32 # Batch size for training and validation
  d_model: 512 # Transformer model dimension
  ff_dim: 1024 # Feed forward model dimension for the encoder / decoder layers
  dropout: 0.1 # Dropout percentage
  nheads: 8 # Number of attention heads
  decoder_layers: 4 # Number of decoder layers
  encoder_layers: 4 # Number of encoder layers
train:
  lr: 1e-4 # Learning rate
  weight_decay: 1e-5 # Weight decay
  warmup_steps: 1000 # Number of warmup steps
  total_model_steps: 30_000_000 # Total model steps (Used for the learning rate decay)
  num_epochs: 1 # Number of epochs to train
  metric_window: 100 # The window used for calculating rolling averages for metrics
  plot_per_x_batches: 1000 # How many batches to update metric plots
  val_per_x_batches: 250000 # How many batches to perform model validation
```

### Modification Configuration
TODO