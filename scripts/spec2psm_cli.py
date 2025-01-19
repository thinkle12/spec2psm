import os
import sys
import argparse
from pathlib import Path
import hydra
from omegaconf import OmegaConf
import torch
import random
import numpy as np
import logging
from spec2psm.inference import Inference
from spec2psm.datasets import Spec2PSMDataset
from spec2psm.train import Trainer
from spec2psm.peptide import PeptideManager
from spec2psm.tokenizer import Tokenizer
from spec2psm.model import Spec2Psm
from spec2psm.config import ModelConfig
from spec2psm.optimizer import Optimizer
from spec2psm.parser import Mzml, Mzid, PepXML, PercolatorOutput, SpectraToSearchMap

from torch.utils.data import DataLoader

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="Support for mismatched key_padding_mask and attn_mask is deprecated."
)

# Set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logger = logging.getLogger(__name__)

# set up our logger
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def convert_mode(config_path, spectra_paths, search_paths, fdr_paths, output_directory, mods_path=None):
    config = OmegaConf.load(config_path)
    logger.info("Running Convert Mode")
    logger.info("Config:", OmegaConf.to_yaml(config))
    if mods_path:
        mods_config = OmegaConf.load(mods_path)
        logger.info("Mods Config:", OmegaConf.to_yaml(mods_config))

    peptide_manager = PeptideManager(token_map_size=config["tokens"]["token_map_size"], config_path=mods_path)
    tokenizer = Tokenizer(
        peptide_manager=peptide_manager,
        max_peptide_length=config["tokens"]["max_peptide_length"],
        spectra_length=config["tokens"]["spectra_length"],
    )

    # If output directory not provided, write the parquet into the current mzML file in the loop
    dynamic_output_directory = False
    if not output_directory:
        dynamic_output_directory = True

    if spectra_paths and not search_paths and not fdr_paths:
        for i in range(len(spectra_paths)):
            mzml_file = spectra_paths[i]
            mzml = Mzml(filepath=mzml_file)
            mzml.parse_ms2s()
            if dynamic_output_directory:
                output_directory = os.path.dirname(mzml_file)
            mzml.results_to_parquet(directory=output_directory)

    if spectra_paths and search_paths and not fdr_paths:
        if len(spectra_paths) != len(search_paths):
            raise ValueError("Number of Spectral and Search files are not identical")
        for i in range(len(spectra_paths)):
            mzml_file = spectra_paths[i]
            mzml = Mzml(filepath=mzml_file)
            mzml.parse_ms2s()

            mzid_file = search_paths[i]
            mzid = Mzid(filepath=mzid_file, tokenizer=tokenizer)
            mzid.parse_results()

            mapper = SpectraToSearchMap(spectra_object=mzml, search_object=mzid)
            mapper.map_results()
            if dynamic_output_directory:
                output_directory = os.path.dirname(mzml_file)
            mapper.results_to_parquet(directory=output_directory)

    if spectra_paths and search_paths and fdr_paths:
        if len(spectra_paths) != len(search_paths) != len(fdr_paths):
            raise ValueError("Number of Spectral, Search, and PSM FDR files are not identical")
        for i in range(len(spectra_paths)):
            mzml_file = spectra_paths[i]
            mzml = Mzml(filepath=mzml_file)
            mzml.parse_ms2s()

            pepxml_file = search_paths[i]
            pepxml = PepXML(filepath=pepxml_file)
            pepxml.parse_results()

            pout_file = fdr_paths[i]
            percolator = PercolatorOutput(filepath=pout_file)
            percolator.parse_results()

            mapper = SpectraToSearchMap(spectra_object=mzml, search_object=pepxml, percolator_object=percolator)
            mapper.map_results()
            if dynamic_output_directory:
                output_directory = os.path.dirname(mzml_file)
            mapper.results_to_parquet(directory=output_directory)

def train_mode(model_filename, config_path, device, train_parquet_paths, val_parquet_paths=None, file_output_tag=None, mods_path=None):
    config = OmegaConf.load(config_path)

    logger.info("Running Train Mode")
    logger.info("Config:")
    logger.info(OmegaConf.to_yaml(config))

    peptide_manager = PeptideManager(token_map_size=config["tokens"]["token_map_size"], config_path=mods_path)
    tokenizer = Tokenizer(
        peptide_manager=peptide_manager,
        max_peptide_length=config["tokens"]["max_peptide_length"],
        spectra_length=config["tokens"]["spectra_length"],
    )

    device_object = ModelConfig.get_device(device)

    train_dataset = Spec2PSMDataset(
        train_parquet_paths,
        tokenizer=tokenizer,
        row_group_size=config["tokens"]["row_group_size"],
        swap_il=True,
        train=True,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=config["model"]["batch_size"], shuffle=True)

    model = Spec2Psm(
        vocab_size=tokenizer.vocab_size,  # e.g., 20 amino acids + start/end/pad tokens
        d_model=config["model"]["d_model"],  # model dimension
        ff_dim=config["model"]["ff_dim"],
        dropout=config["model"]["dropout"],
        max_peptide_length=tokenizer.max_peptide_length,
        # max sequence length for input spectra and peptides (usually 62)
        nhead=config["model"]["nheads"],  # number of heads in multi-head attention
        num_encoder_layers=config["model"]["encoder_layers"],  # number of encoder layers
        num_decoder_layers=config["model"]["decoder_layers"],
        precursor_fusion_position="post_encoder",
        learnable_positional_peptide_embedding=False,
        peak_layer_norm=False,
        latent_spectrum_type="singular",
    ).to(
        device_object
    )  # number of decoder layers

    # Auto file naming
    tag = ModelConfig.get_param_tag(model, config)
    if file_output_tag:
        combined_file_tag = "{}_{}".format(file_output_tag, tag)
    else:
        combined_file_tag = tag

    optimizer = Optimizer(
        model=model,
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
        warmup_steps=config["train"]["warmup_steps"],
        total_steps=config["train"]["total_model_steps"],
    )

    # Training loop
    train = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        device=device,
        metric_window=config["train"]["metric_window"],  # 100
        val_dataloader=None,
        plot_per_x_batches=config["train"]["plot_per_x_batches"],  # 1000
        val_per_x_batches=config["train"]["val_per_x_batches"],  # 250000 - Only used if val dataloader is passed
        tag=combined_file_tag,
    )

    train.train(num_epochs=config["train"]["num_epochs"], model_filename=model_filename, checkpoint_per_x_batches=1000)


def finetune_mode(
    model_filename,
    config_path,
    model_name_or_path,
    train_parquet_paths,
    device,
    resume=False,
    val_parquet_paths=None,
    file_output_tag=None,
    mods_path=None
):

    config = OmegaConf.load(config_path)
    logger.info("Running FineTune Mode")
    logger.info("Config:", OmegaConf.to_yaml(config))

    peptide_manager = PeptideManager(token_map_size=config["tokens"]["token_map_size"], config_path=mods_path)
    tokenizer = Tokenizer(
        peptide_manager=peptide_manager,
        max_peptide_length=config["tokens"]["max_peptide_length"],
        spectra_length=config["tokens"]["spectra_length"],
    )

    device_object = ModelConfig.get_device(device)

    train_dataset = Spec2PSMDataset(
        train_parquet_paths,
        tokenizer=tokenizer,
        row_group_size=config["tokens"]["row_group_size"],
        swap_il=True,
        train=True,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=config["model"]["batch_size"], shuffle=True)

    # Load from local OR hugginface hub
    if os.path.exists(model_name_or_path):
        model = Spec2Psm.from_local_checkpoint(model_name_or_path).to(device_object)
    else:
        model = Spec2Psm.from_pretrained(model_name_or_path).to(device_object)

    if resume and os.path.exists(model_name_or_path):
        # Get the model checkpoint if resuming from a local checkpoint
        checkpoint = torch.load(model_name_or_path, weights_only=False)
        logger.info("Loading Optimizer Details from Checkpoint")
        optimizer = Optimizer.load_from_state_dict(
            state_dict=checkpoint,
            model=model,
            lr=config["train"]["lr"],
            weight_decay=config["train"]["weight_decay"],
            warmup_steps=config["train"]["warmup_steps"],
            total_steps=config["train"]["total_model_steps"],
        )

        start_batch = checkpoint['iterations'] if 'iterations' in checkpoint else 0

    else:
        # If not resuming OR loading from huggingface (To finetune a pre trained model)
        # Then we must load optimizer from parameters
        logger.info("Loading Optimizer Details from Parameter File")
        optimizer = Optimizer(
            model=model,
            lr=config["train"]["lr"],
            weight_decay=config["train"]["weight_decay"],
            warmup_steps=config["train"]["warmup_steps"],
            total_steps=config["train"]["total_model_steps"],
        )
        start_batch = 0

    # Auto file naming
    tag = ModelConfig.get_param_tag(model, config)
    if file_output_tag:
        combined_file_tag = "{}_{}".format(file_output_tag, tag)
    else:
        combined_file_tag = tag

    # Training loop
    train = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        device=device,
        metric_window=config["train"]["metric_window"],  # 100
        val_dataloader=None,
        plot_per_x_batches=config["train"]["plot_per_x_batches"],  # 1000
        val_per_x_batches=config["train"]["val_per_x_batches"],  # 250000 - Only used if val dataloader is passed
        tag=combined_file_tag,
    )

    train.train(num_epochs=config["train"]["num_epochs"], model_filename=model_filename, start_batch=start_batch)


def validate_mode(config_path, model_name_or_path, parquet_path, device, mods_path=None):
    # Initialize classes needed for model validation

    if config_path:
        config = OmegaConf.load(config_path)
        logger.info("Running Infer Mode")
        logger.info("Config:", OmegaConf.to_yaml(config))
    else:
        logger.info("Running without extra tokens")
    # Add inference logic here

    peptide_manager = PeptideManager(token_map_size=config["tokens"]["token_map_size"], config_path=mods_path)
    tokenizer = Tokenizer(
        peptide_manager=peptide_manager,
        max_peptide_length=config["tokens"]["max_peptide_length"],
        spectra_length=config["tokens"]["spectra_length"],
    )

    # device_object = ModelConfig.get_device(device)
    device_object = ModelConfig.get_device("cpu")  # Need some logic here... validate / infer doesnt work on MPS...

    val_dataset = Spec2PSMDataset(
        parquet_path, tokenizer=tokenizer, row_group_size=config["tokens"]["row_group_size"], swap_il=True, train=True
    )

    # Load from local OR hugginface hub
    if os.path.exists(model_name_or_path):
        model = Spec2Psm.from_local_checkpoint(model_name_or_path).to(device_object)
    else:
        model = Spec2Psm.from_pretrained(model_name_or_path).to(device_object)

    val_dataloader = DataLoader(val_dataset, batch_size=config["model"]["batch_size"], shuffle=True)

    # Auto file naming
    tag = ModelConfig.get_param_tag(model, config)

    train = Trainer(
        train_dataloader=None,
        tokenizer=tokenizer,
        model=model,
        val_dataloader=val_dataloader,
        optimizer=None,
        device=device_object,
        metric_window=config["train"]["metric_window"],  # 100
        plot_per_x_batches=config["train"]["plot_per_x_batches"],  # 1000
        val_per_x_batches=config["train"]["val_per_x_batches"],  # 250000 - Only used if val dataloader is passed
        tag=tag,
    )

    train.validate(1)


def infer_mode(config_path, model_name_or_path, parquet_path, device, mods_path=None):

    config = OmegaConf.load(config_path)
    logger.info("Running Infer Mode")
    logger.info("Config:", OmegaConf.to_yaml(config))

    peptide_manager = PeptideManager(token_map_size=config["tokens"]["token_map_size"], config_path=mods_path)
    tokenizer = Tokenizer(
        peptide_manager=peptide_manager,
        max_peptide_length=config["tokens"]["max_peptide_length"],
        spectra_length=config["tokens"]["spectra_length"],
    )

    # device_object = ModelConfig.get_device(device)
    device_object = ModelConfig.get_device("cpu")

    infer_dataset = Spec2PSMDataset(
        parquet_path, tokenizer=tokenizer, row_group_size=config["tokens"]["row_group_size"], swap_il=False, train=True
    )

    # Load from local OR hugginface hub
    if os.path.exists(model_name_or_path):
        model = Spec2Psm.from_local_checkpoint(model_name_or_path).to(device_object)
    else:
        model = Spec2Psm.from_pretrained(model_name_or_path).to(device_object)

    infer = Inference(model, infer_dataset, tokenizer=tokenizer, device=device_object)
    infer.run_inference(beam=True)


def get_parquet_paths(input_paths):
    """
    Parse the input paths to collect Parquet file paths.
    Handles both directories and individual file paths.
    """
    parquet_paths = []

    for path in input_paths:
        if os.path.isdir(path):  # Check if it's a directory
            # Recursively find all parquet files in the directory
            parquet_paths.extend([str(p) for p in Path(path).rglob("*.parquet")])
        elif os.path.isfile(path) and path.endswith(".parquet"):  # Check if it's a file
            parquet_paths.append(path)
        else:
            logger.info(f"Warning: {path} is not a valid directory or Parquet file.")

    return parquet_paths


def main():
    parser = argparse.ArgumentParser(
        prog="spec2psm",
        description="Command-line interface for the Spec2PSM pipeline. Provides tools for converting spectral files to parquet, training models, fine-tuning, validation, and inference."
    )
    subparsers = parser.add_subparsers(dest="mode", help="Available modes")

    # Parse mode
    convert_parser = subparsers.add_parser("convert", help="Convert a spectral file and/or psm file to parquet using a configuration file. To be used for training and inference.")
    convert_parser.add_argument("-c", "--config_path", help="Path to the configuration file for parsing.")
    convert_parser.add_argument(
        "-o",
        "--output_directory",
        help="Optional: Directory to save Parquet files in. If not provided will save in the same directory mzML files live.",
        default=None
    )
    convert_parser.add_argument(
        "-m",
        "--mzml_paths",
        nargs="+",
        required=True,
        help="One or more mzML file paths for converting to Parquet. Provide a space-separated list if multiple paths are required."
    )
    convert_parser.add_argument(
        "-s",
        "--search_paths",
        nargs="+",
        required=False,
        help="One or more mzid OR pepxml file paths for converting to Parquet. Provide a space-separated list if multiple paths are required. idXML support coming in the future."
    )
    convert_parser.add_argument(
        "-f",
        "--fdr_paths",
        nargs="+",
        required=False,
        help="One or more target percolator result file paths for converting to Parquet. Provide a space-separated list if multiple paths are required. This is only compatible if mzML and pepXML paths are both provided as well."
    )
    convert_parser.add_argument(
        "-u",
        "--mods_path",
        help="Optional: Path to a file specifying modification settings.",
        default=None
    )

    # Train mode
    train_parser = subparsers.add_parser("train", help="Train a Spec2PSM model.")
    train_parser.add_argument(
        "-o",
        "--output_model_name",
        help="Path to save the trained model weights."
    )
    train_parser.add_argument(
        "-c",
        "--config_path",
        help="Path to the spec2psm configuration file.",
    )
    train_parser.add_argument(
        "-t",
        "--train_parquet_paths",
        nargs="+",
        required=True,
        help="One or more directories or Parquet file paths for training. Provide a space-separated list if multiple paths are required."
    )
    train_parser.add_argument(
        "-v",
        "--val_parquet_paths",
        nargs="+",
        required=False,
        help="Optional: One or more directories or Parquet file paths for validation. Provide a space-separated list if multiple paths are required."
    )
    train_parser.add_argument(
        "-d",
        "--device",
        help="Device to use for training: 'mps', 'cpu', or 'gpu'.",
        default=None
    )
    train_parser.add_argument(
        "-u",
        "--mods_path",
        help="Optional: Path to a file specifying additional modification settings.",
        default=None
    )

    # Finetune mode
    tune_parser = subparsers.add_parser("tune", help="Fine-tune a pre-trained Spec2PSM model.")
    tune_parser.add_argument(
        "-m",
        "--model",
        help="Path to the model weights or name of a Hugging Face model."
    )
    tune_parser.add_argument(
        "-o",
        "--output_model_name",
        help="Path to save the fine-tuned model weights."
    )
    tune_parser.add_argument(
        "-c",
        "--config_path",
        help="Path to the Spec2PSM configuration file.",
    )
    tune_parser.add_argument(
        "-t",
        "--train_parquet_paths",
        nargs="+",
        required=True,
        help="One or more directories or Parquet file paths for training. Use a space-separated list for multiple paths."
    )
    tune_parser.add_argument(
        "-v",
        "--val_parquet_paths",
        nargs="+",
        required=False,
        help="Optional: One or more directories or Parquet file paths for validation. Use a space-separated list for multiple paths."
    )
    tune_parser.add_argument(
        "-d",
        "--device",
        help="Device to use for fine-tuning: 'mps', 'cpu', or 'gpu'.",
        default=None
    )
    tune_parser.add_argument(
        "-r",
        "--resume",
        action="store_true",
        default=False,
        help="Optional: Resume training from a checkpoint. Includes optimizer settings."
    )
    tune_parser.add_argument(
        "-u",
        "--mods_path",
        help="Optional: Path to a file specifying modification settings.",
        default=None
    )

    # Validate mode
    validate_parser = subparsers.add_parser("validate", help="Validate a Spec2PSM model.")
    validate_parser.add_argument(
        "-m",
        "--model",
        help="Path to the model weights or name of a Hugging Face model."
    )
    validate_parser.add_argument(
        "-c",
        "--config_path",
        help="Path to the Spec2PSM configuration file.",
    )
    validate_parser.add_argument(
        "-p",
        "--parquet_paths",
        nargs="+",
        required=True,
        help="One or more directories or Parquet file paths for validation. Use a space-separated list for multiple paths."
    )
    validate_parser.add_argument(
        "-d",
        "--device",
        help="Device to use for validation: 'mps', 'cpu', or 'gpu'.",
        default=None
    )
    validate_parser.add_argument(
        "-u",
        "--mods_path",
        help="Optional: Path to a file specifying modification settings.",
        default=None
    )

    # Infer mode
    infer_parser = subparsers.add_parser("infer", help="Run inference using a Spec2PSM model.")
    infer_parser.add_argument(
        "-m",
        "--model",
        help="Path to the model weights or name of a Hugging Face model."
    )
    infer_parser.add_argument(
        "-c",
        "--config_path",
        help="Path to the Spec2PSM configuration file.",
    )
    infer_parser.add_argument(
        "-p",
        "--parquet_paths",
        nargs="+",
        required=True,
        help="One or more directories or Parquet file paths for inference. Use a space-separated list for multiple paths."
    )
    infer_parser.add_argument(
        "-d",
        "--device",
        help="Device to use for inference: 'mps', 'cpu', or 'gpu'.",
        default=None
    )
    infer_parser.add_argument(
        "-u",
        "--mods_path",
        help="Optional: Path to a file specifying modification settings.",
        default=None
    )

    args = parser.parse_args()

    if args.mode is None:
        # No mode provided; show the main help message
        parser.print_help()
        return

    if args.mode == "convert":
        convert_mode(config_path=args.config_path,
                     spectra_paths=args.mzml_paths,
                     search_paths=args.search_paths,
                     fdr_paths=args.fdr_paths,
                     output_directory=args.output_directory,
                     mods_path=args.mods_path)
    elif args.mode == "train":
        train_paths = get_parquet_paths(args.train_parquet_paths)
        if args.val_parquet_paths:
            val_paths = get_parquet_paths(args.val_parquet_paths)
        else:
            val_paths = None
        train_mode(
            model_filename=args.output_model_name,
            config_path=args.config_path,
            device=args.device,
            train_parquet_paths=train_paths,
            val_parquet_paths=val_paths,
            mods_path=args.mods_path,
        )
    elif args.mode == "infer":
        parquet_path = get_parquet_paths(args.parquet_path)
        infer_mode(
            config_path=args.config_path, model_name_or_path=args.model, parquet_path=parquet_path, device=args.device, mods_path=args.mods_path
        )
    elif args.mode == "validate":
        parquet_path = get_parquet_paths(args.parquet_path)
        validate_mode(
            config_path=args.config_path, model_name_or_path=args.model, parquet_path=parquet_path, device=args.device, mods_path=args.mods_path
        )
    elif args.mode == "tune":
        train_paths = get_parquet_paths(args.train_parquet_paths)
        if args.val_parquet_paths:
            val_paths = get_parquet_paths(args.val_parquet_paths)
        else:
            val_paths = None
        finetune_mode(
            model_filename=args.output_model_name,
            config_path=args.config_path,
            model_name_or_path=args.model,
            resume=args.resume,
            train_parquet_paths=train_paths,
            val_parquet_paths=val_paths,
            device=args.device,
            mods_path=args.mods_path,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
