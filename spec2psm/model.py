import torch
import torch.nn as nn
import torch.nn.functional as F
from depthcharge.components.encoders import FloatEncoder, PeakEncoder, PositionalEncoder
from huggingface_hub import PyTorchModelHubMixin
import math
import torch.optim as optim
import torch.optim.lr_scheduler
import logging

logger = logging.getLogger(__name__)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(LearnablePositionalEncoding, self).__init__()
        # Create a learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_length, d_model))

    def forward(self, embedded_peptides):
        # Add positional encodings to the embedded peptides
        seq_length = embedded_peptides.size(1)
        return embedded_peptides + self.positional_encoding[:, :seq_length, :]


class Spec2Psm(
    nn.Module, PyTorchModelHubMixin, library_name="spec2psm", repo_url="https://github.com/thinkle12/spec2psm"
):
    def __init__(
        self,
        vocab_size,
        d_model,
        max_peptide_length,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        ff_dim,
        dropout=0.1,
        max_charge=10,
        precursor_fusion_position="post_encoder",
        learnable_positional_peptide_embedding=False,
        peak_layer_norm=False,
        latent_spectrum_type="singular",
        pretrained_weights_path=None,
    ):
        super(Spec2Psm, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.max_charge = max_charge
        self.learnable_positional_peptide_embedding = learnable_positional_peptide_embedding
        self.peak_layer_norm = peak_layer_norm
        self.precursor_fusion_position = precursor_fusion_position
        self.max_peptide_length = max_peptide_length
        self.latent_spectrum_type = latent_spectrum_type

        # Transformer Encoder and Decoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=True
        )
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)

        # Embedding for the peptides
        # Assuming padding index is 0
        self.output_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0).to(
            torch.float32
        )  # Assuming vocab size

        # Learnable latent representations for each charge state and precursor mass bucket
        # Or one learnable latent spectrum
        if self.latent_spectrum_type == "singular":
            self.latent_spectrum = torch.nn.Parameter(torch.randn(1, 1, d_model))

        # self.peak_encoder = PeakEncoder(d_model) # TODO This is best
        # self.peak_encoder = PeakEncoderLinearReluSkipNet(input_dim=d_model) # TODO This is 2nd best barely...
        # self.peak_encoder = PeakEncoderFourier(dim_model=d_model) # This is the same as PeakEncoder... TODO This is 2nd/3rd best basically...
        # self.peak_encoder = MS2PeakEncoderBasic(d_model=d_model)
        self.peak_encoder = PeakEncoderLinearTanhSkipNetComplex(input_dim=d_model)  # This is as good as PeakEncoder

        self.charge_encoder = torch.nn.Embedding(max_charge, d_model)
        self.mz_encoder = FloatEncoder(d_model)

        # Layer normalization for input embeddings
        if self.peak_layer_norm:
            self.layer_norm_src = nn.LayerNorm(d_model)

        # Positional embeddings for the peptide
        if not self.learnable_positional_peptide_embedding:
            self.positional_encoding_peptide = PositionalEncoder(d_model)
        else:
            self.positional_encoding_peptide = LearnablePositionalEncoding(d_model, max_peptide_length)

        # Output linear layer to predict peptide sequence
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Call weight initialization function
        self._init_weights()

        # Load pre trained weights if they exist...
        if pretrained_weights_path:
            self.load_pretrained_weights(pretrained_weights_path)

        num_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Total number of parameters: {num_params}")

    def _init_weights(self):
        # Initialize all linear layers with Xavier uniform
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, src, tgt, tgt_padding_mask, tgt_mask, precursors):
        # src: (batch_size, seq_len, 2) --> spectra input
        # tgt: (batch_size, tgt_len) --> peptide sequence output (as indices)

        src = self.peak_encoder(src)

        # Apply layer normalization for the source embeddings
        if self.peak_layer_norm:
            src = self.layer_norm_src(src)

        # Encode the precursors (charge and m/z)
        masses = self.mz_encoder(precursors[:, None, 0])  # Assuming precursors contains m/z in the first column
        charges = self.charge_encoder(precursors[:, 1].int())  # Adjusting charge indices

        # Train a small latent representation of a spectrum
        # Either granular or singular
        if self.latent_spectrum_type == "singular":
            latent_spectra = self.latent_spectrum.expand(src.shape[0], -1, -1)

        precursors_encoded = torch.cat((latent_spectra, charges.unsqueeze(1), masses), dim=1)
        # Shape: (batch, 3, d_model)

        # Add the encoded precursors to the encoded spectra
        # Might play around with this and concat after transformer encoder instead
        if self.precursor_fusion_position == "pre_encoder":
            src = torch.cat([precursors_encoded, src], dim=1)
            src_padding_mask = src[:, :, 1] == 0  # Mask out zero values in the intensity feature
            # Shape: (batch, num_peaks+3, d_model)
        else:
            src_padding_mask = src[:, :, 1] == 0  # Mask out zero values in the intensity feature
            # Shape: (batch, num_peaks, d_model)

        # Pass through the transformer encoder
        encoder_output = self.transformer_encoder(src, src_key_padding_mask=src_padding_mask)

        if self.precursor_fusion_position == "post_encoder":
            # If fusing precursor post encoder
            encoder_output = torch.cat([precursors_encoded, encoder_output], dim=1)
            # Then we need to create a new src_padding mask
            src_padding_mask = encoder_output[:, :, 1] == 0  # Mask out zero values in the intensity feature
        else:
            pass

        # Apply output embeddings for peptide sequence
        tgt = self.output_embedding(tgt)

        # Apply sinusoidal or learnable positional encoding for target (peptide sequence)
        tgt_encoded = self.positional_encoding_peptide(tgt)

        # Pass through the transformer decoder
        output = self.transformer_decoder(
            tgt_encoded,
            encoder_output,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )

        # Final linear layer to predict the next token in the peptide sequence
        output = self.fc_out(output)

        return output

    def make_trg_mask(self, trg):
        # Create a mask for the target sequence
        trg_pad_mask = (trg != 0).unsqueeze(1).unsqueeze(2)  # Padding mask
        trg_len = trg.size(1)

        # Create a square mask for the target sequence (no future tokens)
        trg_subsequent_mask = torch.triu(torch.ones((trg_len, trg_len)), diagonal=1).bool().to(trg.device)

        # Combine the masks: Pad mask and subsequent mask
        trg_mask = trg_pad_mask & ~trg_subsequent_mask
        return trg_mask

    @classmethod
    def from_local_checkpoint(cls, checkpoint_path):
        """
        Custom method to load from a local .pth file.
        """
        logger.info("Loading Model from Local Checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = cls(**checkpoint["config"])  # Initialize model with config from checkpoint
        model.load_state_dict(checkpoint["model_state_dict"])  # Load model weights
        return model

    def save_model(self, model_filename, optimizer=None, scheduler=None, iterations=None):
        """
        Save both the model's state_dict and its configuration to the specified directory.
        """
        model_state_dict = self.state_dict()  # Save the model weights

        # Save optimizer and scheduler statedicts if provided
        if optimizer:
            optimizer = optimizer.state_dict()
        if scheduler:
            scheduler = scheduler.state_dict()
        # Gather configuration from the initialization parameters
        config = {
            "vocab_size": self.output_embedding.num_embeddings,  # Extract vocab size from embedding layer
            "d_model": self.d_model,
            "max_peptide_length": self.max_peptide_length,
            "nhead": self.nhead,
            "num_encoder_layers": len(self.transformer_encoder.layers),
            "num_decoder_layers": len(self.transformer_decoder.layers),
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
            "max_charge": self.charge_encoder.num_embeddings,
            "precursor_fusion_position": self.precursor_fusion_position,
            "learnable_positional_peptide_embedding": self.learnable_positional_peptide_embedding,
            "peak_layer_norm": self.peak_layer_norm,
            "latent_spectrum_type": self.latent_spectrum_type,
        }

        torch.save(
            {
                "config": config,
                "model_state_dict": model_state_dict,
                "iterations": iterations,
                "optimizer_state_dict": optimizer,
                "scheduler_state_dict": scheduler,
            },
            model_filename,
        )

    def to_huggingface(self, model_name):
        self.save_pretrained(model_name)
        self.push_to_hub(model_name)


class FourierFeatureEncoder(nn.Module):
    def __init__(self, dim_model=512, scale=10.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn((1, 1, dim_model // 2)) * scale, requires_grad=False)
        self.b = nn.Parameter(torch.randn((1, 1, dim_model // 2)) * 2 * torch.pi, requires_grad=False)

    def forward(self, x):
        sin_part = torch.sin(x * self.W + self.b)
        cos_part = torch.cos(x * self.W + self.b)
        return torch.cat([sin_part, cos_part], dim=-1)


class FloatEncoderWithComplexMLP(torch.nn.Module):
    """Float Encoder with sinusoidal encoding followed by a complex MLP for enhanced transformation."""

    def __init__(self, dim_model, min_wavelength=0.001, max_wavelength=10000, hidden_dims=(512, 256, 128)):
        """
        Initialize the FloatEncoderWithComplexMLP.

        Parameters
        ----------
        dim_model : int
            The number of features to output from the sinusoidal encoder.
        min_wavelength : float
            The minimum wavelength for the sinusoidal encoder.
        max_wavelength : float
            The maximum wavelength for the sinusoidal encoder.
        hidden_dims : tuple of int
            The dimensions of the hidden layers in the MLP.
        """
        super().__init__()

        # Initialize sinusoidal encoder
        self.float_encoder = FloatEncoder(dim_model, min_wavelength, max_wavelength)

        # Define MLP with multiple hidden layers
        layers = []
        input_dim = dim_model
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(0.1))  # Optional dropout for regularization
            input_dim = hidden_dim
        layers.append(torch.nn.Linear(input_dim, dim_model))  # Final layer maps back to dim_model

        # Combine into a Sequential module
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, X):
        """
        Forward pass for the FloatEncoderWithComplexMLP.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, seq_len)
            Input tensor with floating-point values.

        Returns
        -------
        torch.Tensor of shape (batch_size, seq_len, dim_model)
            The transformed sinusoidal encoding.
        """
        # Generate static sinusoidal encoding
        X = X[:, :, 0]
        static_encoding = self.float_encoder(X)  # Shape: (batch_size, seq_len, dim_model)

        # Pass through the complex MLP
        refined_encoding = self.mlp(static_encoding)  # Shape: (batch_size, seq_len, dim_model)
        return refined_encoding


class FourierEmbedding(nn.Module):
    """
    Fourier Embedding module for expanding scalar values into high-dimensional space
    using sine and cosine functions with different frequencies.

    Parameters
    ----------
    dim : int
        The dimensionality of the embedding space (output dimension).
    min_freq : float
        The minimum frequency for sine and cosine functions.
    max_freq : float
        The maximum frequency for sine and cosine functions.
    """

    def __init__(self, dim, min_freq=1.0, max_freq=1000.0):
        super(FourierEmbedding, self).__init__()
        self.dim = dim
        self.min_freq = min_freq
        self.max_freq = max_freq

        # Compute frequencies as a log-spaced range
        half_dim = dim // 2
        self.freqs = torch.exp(torch.linspace(math.log(min_freq), math.log(max_freq), half_dim))

    def forward(self, x):
        """
        Apply Fourier embeddings to the input tensor.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, seq_len, features)
            Input features to be embedded.

        Returns
        -------
        torch.Tensor of shape (batch_size, seq_len, dim)
            High-dimensional Fourier embeddings.
        """
        # Ensure x is (batch_size, seq_len, 1) for scalar input
        x = x.unsqueeze(-1) if x.ndim == 2 else x

        # Compute scaled input by frequencies
        x_scaled = x * self.freqs.to(x.device).view(1, 1, -1)

        # Compute sine and cosine embeddings
        sin_embed = torch.sin(x_scaled)
        cos_embed = torch.cos(x_scaled)

        # Concatenate sine and cosine embeddings
        return torch.cat([sin_embed, cos_embed], dim=-1)


class PeakEncoderFourier(torch.nn.Module):
    """Encode mass spectrum.

    Parameters
    ----------
    dim_model : int
        The number of features to output.
    dim_intensity : int, optional
        The number of features to use for intensity. The remaining features
        will be used to encode the m/z values.
    min_freq : float, optional
        The minimum frequency for sine and cosine embeddings.
    max_freq : float, optional
        The maximum frequency for sine and cosine embeddings.
    learned_intensity_encoding : bool, optional
        Use a learned intensity encoding as opposed to a sinusoidal encoding.
        Note that for the sinusoidal encoding, this encoder expects values
        between [0, 1].
    """

    def __init__(
        self,
        dim_model,
        dim_intensity=None,
        min_freq=1.0,
        max_freq=1000.0,
        learned_intensity_encoding=True,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.dim_mz = dim_model
        self.learned_intensity_encoding = learned_intensity_encoding

        if dim_intensity is not None:
            if dim_intensity >= dim_model:
                raise ValueError("'dim_intensity' must be less than 'dim_model'")
            self.dim_mz -= dim_intensity
            self.dim_intensity = dim_intensity
        else:
            self.dim_intensity = dim_model

        # Use FourierEmbedding for m/z encoding
        self.mz_encoder = FourierEmbedding(dim=self.dim_mz, min_freq=min_freq, max_freq=max_freq)

        # Choose between learned or sinusoidal intensity encoding
        if self.learned_intensity_encoding:
            self.int_encoder = torch.nn.Linear(1, self.dim_intensity, bias=False)
        else:
            self.int_encoder = FourierEmbedding(dim=self.dim_intensity, min_freq=min_freq, max_freq=max_freq)

    def forward(self, X):
        """Encode m/z values and intensities.

        Note that we expect intensities to fall within the interval [0, 1].

        Parameters
        ----------
        X : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra to embed. Axis 0 represents a mass spectrum, axis 1
            contains the peaks in the mass spectrum, and axis 2 is essentially
            a 2-tuple specifying the m/z-intensity pair for each peak. These
            should be zero-padded, such that all of the spectra in the batch
            are the same length.

        Returns
        -------
        torch.Tensor of shape (n_spectra, n_peaks, dim_model)
            The encoded features for the mass spectra.
        """
        # Separate m/z and intensity
        m_over_z = X[:, :, 0]  # (batch_size, n_peaks)
        encoded = self.mz_encoder(m_over_z)  # Fourier embedding for m/z

        if self.learned_intensity_encoding:
            int_input = X[:, :, [1]]  # Keep intensity shape as (batch, n_peaks, 1)
        else:
            int_input = X[:, :, 1]  # (batch, n_peaks)

        # Encode intensity
        intensity = self.int_encoder(int_input)

        # Combine m/z and intensity encodings
        if self.dim_intensity == self.dim_model:
            return encoded + intensity

        return torch.cat([encoded, intensity], dim=2)


class MS2PeakEncoderBasic(nn.Module):
    """Encode MS2 spectra with learnable embeddings and positional encoding."""

    def __init__(self, d_model=512, max_len=150):
        super(MS2PeakEncoderBasic, self).__init__()

        self.d_model = d_model

        # Linear layers for learnable embeddings
        self.mz_embedding = FloatEncoder(dim_model=d_model)  # Try this again but with tanh??
        self.intensity_embedding = nn.Linear(1, d_model)  # For intensity values

    def forward(self, X):
        """
        Encode MS2 spectra with learnable embeddings and positional encoding.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, seq_len, 2)
            Input tensor with m/z and intensity values.

        Returns
        -------
        torch.Tensor of shape (batch_size, seq_len, d_model)
            Encoded tensor with embedded m/z and intensity values.
        """
        # Split input into m/z and intensity
        mz = X[:, :, 0]  # Shape: (batch_size, seq_len, 1)
        intensity = X[:, :, [1]]  # Shape: (batch_size, seq_len, 1)

        # Apply learnable embeddings
        mz_embedded = self.mz_embedding(mz)  # Shape: (batch_size, seq_len, d_model)
        intensity_embedded = self.intensity_embedding(intensity)  # Shape: (batch_size, seq_len, d_model)

        # Combine m/z and intensity embeddings
        combined = mz_embedded + intensity_embedded  # Shape: (batch_size, seq_len, d_model)

        return combined


class PeakEncoderLinearTanhSkipNetComplex(nn.Module):
    def __init__(self, input_dim=512, hidden_dims=(768, 1024, 512), output_dim=512):
        """
        Extend the PeakEncoder with a skip connection and a deeper MLP for complex transformations.

        Parameters
        ----------
        input_dim : int
            Dimensionality of the input features.
        hidden_dims : tuple of int
            Dimensions of the hidden layers in the MLP.
        output_dim : int
            Dimensionality of the final output.
        """
        super(PeakEncoderLinearTanhSkipNetComplex, self).__init__()
        # PeakEncoder is assumed to output (batch, seq_len, input_dim)
        self.peak_encoder = PeakEncoder(input_dim)

        # Define MLP layers
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(0.1))  # Optional dropout
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))  # Final layer

        self.mlp = nn.Sequential(*layers)

        # Skip connection layer to project input_dim to output_dim
        self.skip_fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Forward pass for the extended PeakEncoder model.

        Parameters
        ----------
        x : torch.Tensor of shape (batch, seq_len, input_dim)
            Input tensor to be encoded.

        Returns
        -------
        torch.Tensor of shape (batch, seq_len, output_dim)
            Encoded output with added complexity from the MLP layers.
        """
        # First layer: PeakEncoder outputs (batch, seq_len, input_dim)
        out = F.tanh(self.peak_encoder(x))

        # Skip connection
        skip_out = self.skip_fc(out)  # Match the dimensions of the MLP output

        # Pass through the MLP
        out = self.mlp(out)

        # Add the skip connection to the final output
        out = out + skip_out

        return out
