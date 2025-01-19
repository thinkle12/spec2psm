import yaml
import torch
import logging

logger = logging.getLogger(__name__)


class ModelConfig:
    def __init__(self, yaml_file):
        """
        Initialize the ModelConfig by loading the configuration file.

        Parameters:
        - yaml_file (str): Path to the YAML file containing the configuration.
        """
        self.config = self._load_config(yaml_file)

    def _load_config(self, yaml_file):
        """
        Load configuration from a YAML file.

        Parameters:
        - yaml_file (str): Path to the YAML file.

        Returns:
        - dict: The parsed configuration dictionary.
        """
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)

        logger.info("Configurations for Model Training")
        logger.info(config)
        return config

    @classmethod
    def get_param_tag(cls, model, config):
        """
        Generate a parameter tag string for the model configuration.

        Parameters:
        - model (torch.nn.Module): The model instance to generate the tag for.

        Returns:
        - str: A formatted tag representing the model configuration.
        """
        num_params = sum(p.numel() for p in model.parameters())
        np_per_mil = f"{num_params / 1_000_000:.1f}M"

        max_peptide_length = int(config["tokens"]["max_peptide_length"])
        spectra_length = int(config["tokens"]["spectra_length"])

        # Model Params
        d_model = int(config["model"]["d_model"])
        ff_dim = int(config["model"]["ff_dim"])
        nheads = int(config["model"]["nheads"])
        decoder_layers = int(config["model"]["decoder_layers"])
        encoder_layers = int(config["model"]["encoder_layers"])
        num_epochs = int(config["train"]["num_epochs"])

        # Learning Rate Params
        lr = float(config["train"]["lr"])
        weight_decay = float(config["train"]["weight_decay"])
        warmup_steps = int(config["train"]["warmup_steps"])

        tag = (
            f"np{np_per_mil}_dm{d_model}_ffd{ff_dim}_nh{nheads}_dl{decoder_layers}"
            f"_el{encoder_layers}_ne{num_epochs}_lr{lr}_wm{warmup_steps}_dc{weight_decay}"
            f"_sl{spectra_length}_pl{max_peptide_length}"
        )

        return tag.replace(".", "_")

    @staticmethod
    def get_device(device_name):
        """
        Determine the appropriate torch device based on the device name.

        Parameters:
        - device_name (str): The preferred device name ('mps', 'cpu', 'gpu', 'cuda').

        Returns:
        - torch.device: The PyTorch device.
        """
        if device_name == "mps":
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        elif device_name == "cpu":
            device = torch.device("cpu")
        elif device_name in {"gpu", "cuda"}:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            raise ValueError(f"Unsupported device name: {device_name}")

        logger.info(f"Using device: {device}")
        return device
