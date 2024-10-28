import torch
import torch.nn as nn
import torch.nn.functional as F
from depthcharge.components.encoders import FloatEncoder, PeakEncoder, PositionalEncoder

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(LearnablePositionalEncoding, self).__init__()
        # Create a learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_length, d_model))

    def forward(self, embedded_peptides):
        # Add positional encodings to the embedded peptides
        seq_length = embedded_peptides.size(1)
        return embedded_peptides + self.positional_encoding[:, :seq_length, :]

class Spec2Psm(nn.Module):
    def __init__(self, vocab_size, d_model, max_peptide_length, nhead, num_encoder_layers, num_decoder_layers, ff_dim, dropout=0.1, max_charge=10, max_precursor_mass=2500, spectra_sequence_length=150, precursor_fusion_position="post_encoder", learnable_positional_peptide_embedding=False, peak_layer_norm=False, latent_spectrum_type="singular"):
        super(Spec2Psm, self).__init__()

        self.learnable_positional_peptide_embedding = learnable_positional_peptide_embedding
        self.peak_layer_norm = peak_layer_norm
        self.max_precursor_mass = max_precursor_mass
        self.precursor_fusion_position = precursor_fusion_position
        self.max_peptide_length = max_peptide_length
        self.spectra_sequence_length = spectra_sequence_length
        self.latent_spectrum_type = latent_spectrum_type

        # Transformer Encoder and Decoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)

        # Embedding for the peptides
        # Assuming padding index is 0
        self.output_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0).to(torch.float32)  # Assuming vocab size

        # Learnable latent representations for each charge state and precursor mass bucket
        # Or one learnable latent spectrum
        if self.latent_spectrum_type == "granular":
            self.latent_spectrum = nn.Embedding(max_charge * max_precursor_mass * 10, d_model)
        if self.latent_spectrum_type == "singular":
            self.latent_spectrum = torch.nn.Parameter(torch.randn(1, 1, d_model))

        self.peak_encoder = PeakEncoder(d_model)
        # self.peak_encoder_enhanced = EnhancedPeakEncoder(d_model=d_model, n_peaks=spectra_sequence_length)
        # self.peak_encoder_enhanced = PeakEncoderLinearReluSkipNet(input_dim=d_model)
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

    def _init_weights(self):
        # Initialize all linear layers with Xavier uniform
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, src, tgt, tgt_padding_mask, tgt_mask, precursors):
        # src: (batch_size, seq_len, 2) --> spectra input
        # tgt: (batch_size, tgt_len) --> peptide sequence output (as indices)

        src = self.peak_encoder(src)
        # src = self.peak_encoder_enhanced(src)
        # TODO after peak encoder
        # TODO consider adding an activiation function
        # TODO then consider adding another linear layer
        # TODO then consider adding layer norm
        # TODO then feed to transformer encoder
        # TODO maybe go peak encoder (512) -> GELU/RELU -> linear 1024 -> GELU/RELU -> Linear 512 -> Layer Norm
        # TODO or maybe go peak encoder (768/1024) -> GELU/RELU -> Linear (512) -> Layer Norm
        # TODO also consider using EnhancedPeakEncoder defined below
        # TODO also consider using 200 or 250 input peaks
        # TODO also check chatgpt email for the skip linear unit that should go AFTER PeakEncoder

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

        if self.latent_spectrum_type == "granular":
            # Below is a learned latent representation by binned precursor mz AND charge
            precursor_buckets = ((precursors[:, None, 0]) * 10).long()  # Adjust for bucketing
            precursor_buckets = precursor_buckets.clamp(0,self.max_precursor_mass*10 - 1)  # Ensure we stay within valid bucket range
            # Combine charge states and precursor bucket indices to get a unique index
            latent_indices = precursors[:, 1] * precursor_buckets  # This assumes charge states can overlap with buckets directly
            latent_indices = latent_indices.long()
            # Get the latent spectrum representation
            latent_spectra = self.latent_spectrum_granular(latent_indices)

        precursors_encoded = torch.cat((latent_spectra, charges.unsqueeze(1), masses), dim=1)
        # Shape: (batch, 3, d_model)

        # Add the encoded precursors to the encoded spectra
        # Might play around with this and concat after transformer encoder instead
        if self.precursor_fusion_position == "pre_encoder":
            src = torch.cat([precursors_encoded, src], dim=1)
            src_padding_mask = (src[:, :, 1] == 0)  # Mask out zero values in the intensity feature
            # Shape: (batch, num_peaks+3, d_model)
        else:
            src_padding_mask = (src[:, :, 1] == 0)  # Mask out zero values in the intensity feature
            # Shape: (batch, num_peaks, d_model)


        # Pass through the transformer encoder
        encoder_output = self.transformer_encoder(src, src_key_padding_mask=src_padding_mask)

        if self.precursor_fusion_position == "post_encoder":
            # If fusing precursor post encoder
            encoder_output = torch.cat([precursors_encoded, encoder_output], dim=1)
            # Then we need to create a new src_padding mask
            src_padding_mask = (encoder_output[:, :, 1] == 0)  # Mask out zero values in the intensity feature
        else:
            pass


        # Apply output embeddings for peptide sequence
        tgt = self.output_embedding(tgt)

        # Apply sinusoidal or learnable positional encoding for target (peptide sequence)
        tgt_encoded = self.positional_encoding_peptide(tgt)

        # Pass through the transformer decoder
        output = self.transformer_decoder(tgt_encoded, encoder_output,
                                          tgt_mask=tgt_mask,
                                          memory_key_padding_mask=src_padding_mask,
                                          tgt_key_padding_mask=tgt_padding_mask)

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


class Spectrum2Vec(nn.Module):
    def __init__(self, input_dim=2, embedding_dim=512, hidden_dim=256, num_layers=2):
        """
        Spectrum2Vec model to embed mass spectra with (m/z, intensity) pairs.

        Args:
        - input_dim: Dimension of input (2 for m/z and intensity)
        - embedding_dim: Dimension of the dense embedding for each (m/z, intensity) pair
        - hidden_dim: Hidden dimension of the network for contextual embedding
        - num_layers: Number of layers in the feed-forward network for contextual embedding
        """
        super(Spectrum2Vec, self).__init__()

        # Linear transformation for embedding (m/z, intensity) pairs
        self.linear = nn.Linear(input_dim, embedding_dim)

        # Feed-forward network to capture relationships between peaks
        layers = []
        layers.append(nn.Linear(embedding_dim, hidden_dim))
        for _ in range(num_layers - 1):
            layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.contextual_network = nn.Sequential(*layers)

        # Final projection layer if you need to project to another size
        self.output_layer = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, spectra, apply_positional_encoding=True):
        """
        Forward pass through Spectrum2Vec.

        Args:
        - spectra: Tensor of shape (batch_size, sequence_length, 2) representing (m/z, intensity) pairs
        - apply_positional_encoding: Whether to apply positional encoding

        Returns:
        - Embedded spectra: Tensor of shape (batch_size, sequence_length, embedding_dim)
        """

        device = spectra.device
        # Linear transformation to get initial embeddings for (m/z, intensity)
        embedded_spectra = self.linear(spectra)  # Shape: (batch_size, sequence_length, embedding_dim)

        if apply_positional_encoding:
            # Ensure positional encoding is on the same device as the input tensor
            positional_encoding = self._generate_positional_encoding(spectra.size(1), device)
            embedded_spectra += positional_encoding.unsqueeze(0)
        # Pass through contextual feed-forward network
        contextual_embeddings = self.contextual_network(embedded_spectra)

        # Optional: Pass through a final output layer
        output_embeddings = self.output_layer(contextual_embeddings)

        return output_embeddings

    def _generate_positional_encoding(self, max_len, device):
        """
        Generate positional encodings for the sequences to retain information about position.
        """
        pe = torch.zeros(max_len, self.linear.out_features).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.linear.out_features, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.linear.out_features))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


class EnhancedPeakEncoder(torch.nn.Module):
    def __init__(self, d_model, n_peaks, dim_intensity=None, learned_intensity_encoding=True):
        super().__init__()
        # Same as before, with added layers for improvement
        self.d_model = d_model
        self.dim_intensity = d_model
        self.learned_intensity_encoding = learned_intensity_encoding
        self.mz_encoder = FloatEncoder(dim_model=d_model,
                                       min_wavelength=0.001,
                                       max_wavelength=10000)

        # Multi-layer feed-forward for intensity
        if self.learned_intensity_encoding:
            self.int_encoder = torch.nn.Sequential(
                torch.nn.Linear(1, self.dim_intensity),
                torch.nn.ReLU(),
                torch.nn.Linear(self.dim_intensity, self.dim_intensity)
            )

        # Positional encoding for peaks
        self.positional_encoding = torch.nn.Parameter(torch.randn(1, n_peaks, d_model))

        # Attention layer for peak selection
        self.attention = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=4)

        # Optional normalization layer
        self.norm = torch.nn.LayerNorm(d_model)

        self.combiner = torch.nn.Linear(2 * d_model, d_model, bias=False)

    def forward(self, X):
        m_over_z = X[:, :, 0]
        encoded_mz = self.mz_encoder(m_over_z)

        int_input = X[:, :, [1]]
        intensity = self.int_encoder(int_input)

        # Positional encoding
        encoded_peaks = encoded_mz + self.positional_encoding

        # Attention mechanism
        attn_output, _ = self.attention(encoded_peaks, encoded_peaks, encoded_peaks)

        # Combine m/z and intensity
        combined = torch.cat([attn_output, intensity], dim=-1)

        condensed = self.combiner(combined)

        # Optional normalization
        return self.norm(condensed)


class PeakEncoderLinearReluSkipNet(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=768, output_dim=512):
        super(PeakEncoderLinearReluSkipNet, self).__init__()
        # PeakEncoder is assumed to output (batch, 150, 512)
        self.peak_encoder = PeakEncoder(
            input_dim)  # input_dim should match the input to PeakEncoder, typically 2 in your case

        # Define the linear layers
        self.fc2 = nn.Linear(input_dim, hidden_dim)  # From (batch, 150, 512) -> (batch, 150, 768)
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # From (batch, 150, 768) -> (batch, 150, 512)

        # Skip connection: it should match the output from PeakEncoder
        self.skip_fc = nn.Linear(input_dim, hidden_dim)  # From (batch, 150, 512) -> (batch, 150, 512)

    def forward(self, x):
        # First layer: PeakEncoder outputs (batch, 150, 512)
        out = F.relu(self.peak_encoder(x))  # Apply ReLU on the output of PeakEncoder

        # Skip connection
        skip_out = self.skip_fc(out)  # Apply skip connection (match the output dimensions of PeakEncoder)

        # Second layer: Linear -> ReLU
        out = F.relu(self.fc2(out))  # Output shape will be (batch, 150, 768)

        # Final linear layer and add the skip connection
        out = out + F.relu(skip_out)  # Add the skip connection after activation

        # Final linear layer
        out = self.fc3(out)  # Final output shape (batch, 150, 512)

        return out