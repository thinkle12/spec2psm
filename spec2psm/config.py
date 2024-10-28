import yaml

def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)

    print("Configurations for Model Training")
    print(config)
    return config

def get_param_tag(config, model):
    num_params = sum(p.numel() for p in model.parameters())
    np_per_mil = f"{num_params / 1_000_000:.1f}M"

    max_peptide_length = int(config["dataset"]["max_peptide_length"])
    spectra_length = int(config["dataset"]["spectra_length"])

    # Model Params
    d_model = int(config["model"]["d_model"])
    ff_dim = int(config["model"]["ff_dim"])
    nheads = int(config["model"]["nheads"])
    decoder_layers = int(config["model"]["decoder_layers"])
    encoder_layers = int(config["model"]["encoder_layers"])
    num_epochs = int(config["model"]["num_epochs"])

    # Learning Rate Params
    lr = float(config["optimizer"]["lr"])
    weight_decay = float(config["optimizer"]["weight_decay"])
    warmup_steps = int(config["optimizer"]["warmup_steps"])

    tag = "np{}_dm{}_ffd{}_nh{}_dl{}_el{}_ne{}_lr{}_wm{}_dc{}_sl{}_pl{}".format(np_per_mil, d_model, ff_dim, nheads, decoder_layers, encoder_layers, num_epochs, lr, warmup_steps, weight_decay, spectra_length, max_peptide_length)

    tag = tag.replace(".", "_")

    return tag
