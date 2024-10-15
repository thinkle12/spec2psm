import torch
import torch.nn as nn
import math
from depthcharge.components.encoders import FloatEncoder, PeakEncoder, PositionalEncoder

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        # Create a learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_length, d_model))

    def forward(self, embedded_peptides):
        # Add positional encodings to the embedded peptides
        seq_length = embedded_peptides.size(1)
        return embedded_peptides + self.positional_encoding[:, :seq_length, :]

class Spec2Psm(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, nhead, num_encoder_layers, num_decoder_layers, ff_dim, dropout=0.2, max_charge=10, max_precursor_mass=2500):
        super(Spec2Psm, self).__init__()

        # Transformer Encoder and Decoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)

        # Embedding layers for input spectra and output peptide
        self.input_embedding = nn.Linear(150, d_model).to(torch.float32)  # Assuming 2 features in spectra
        self.output_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0).to(torch.float32)  # Assuming vocab size

        # Learnable latent representations for each charge state and precursor mass bucket
        self.latent_spectrum_granular = nn.Embedding(max_charge * max_precursor_mass * 10, d_model)

        self.latent_spectrum = torch.nn.Parameter(torch.randn(1, 1, d_model))
        self.peak_encoder = PeakEncoder(d_model)
        self.charge_encoder = torch.nn.Embedding(max_charge, d_model)
        self.mz_encoder = FloatEncoder(d_model)

        # Layer normalization for input embeddings
        # self.layer_norm_src = nn.LayerNorm(d_model)

        # Positional embeddings
        self.positional_encoding_peptide = PositionalEncoder(d_model)

        # Learnable positional encoding initialized as nn.Parameter
        self.learnable_positional_encoding_peptide = PositionalEncoding(d_model, max_seq_len)

        # Output linear layer to predict peptide sequence
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Call weight initialization function
        self._init_weights()

        self.max_precursor_mass = max_precursor_mass

    def _init_weights(self):
        # Initialize all linear layers with Xavier uniform
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, src, tgt, tgt_padding_mask, tgt_mask, precursors):
        # src: (batch_size, seq_len, 2) --> spectra input
        # tgt: (batch_size, tgt_len) --> peptide sequence output (as indices)

        # Apply input embeddings for spectra
        # src = self.input_embedding(src)

        src = self.peak_encoder(src)

        # Apply layer normalization for the source embeddings
        # src = self.layer_norm_src(src)

        # Encode the precursors (charge and m/z)
        masses = self.mz_encoder(precursors[:, None, 0])  # Assuming precursors contains m/z in the first column
        charges = self.charge_encoder(precursors[:, 1].int())  # Adjusting charge indices
        # latent_spectra = self.latent_spectrum.expand(src.shape[0], -1, -1)

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
        encoded_spectra_and_precursors = torch.cat([precursors_encoded, src], dim=1)
        src_padding_mask = (encoded_spectra_and_precursors[:, :, 1] == 0)  # Mask out zero values in the intensity feature
        # Shape: (batch, num_peaks+3, d_model)

        # Pass through the transformer encoder
        encoder_output = self.transformer_encoder(encoded_spectra_and_precursors, src_key_padding_mask=src_padding_mask)

        # Apply output embeddings for peptide sequence
        tgt = self.output_embedding(tgt)

        # Apply sinusoidal positional encoding for target (peptide sequence)
        # tgt_encoded = self.positional_encoding_peptide(tgt)

        # Add the learnable positional encoding
        tgt_encoded = self.learnable_positional_encoding_peptide(tgt)

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
