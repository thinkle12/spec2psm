# This file is for defining the model, cost functions, etc

# TODO write the model here...
import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        # Create a matrix of [max_len, d_model]
        self.encoding = torch.zeros(max_len, d_model)

        for pos in range(max_len):
            for i in range(0, d_model, 2):
                self.encoding[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
                if i + 1 < d_model:
                    self.encoding[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))

        self.encoding = self.encoding.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)

class Spec2Psm(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, nhead, num_encoder_layers, num_decoder_layers, ff_dim, dropout=0.2):

        super(Spec2Psm, self).__init__()

        # Transformer Encoder and Decoder
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          batch_first=True,
                                          dim_feedforward=ff_dim,
                                          dropout=dropout)

        # Embedding layers for input spectra and output peptide
        self.input_embedding = nn.Linear(4, d_model).to(torch.float32)  # Assuming 4 features in spectra
        self.output_embedding = nn.Embedding(vocab_size, d_model).to(torch.float32) # Assuming vocab size

        # Layer normalization for input embeddings
        self.layer_norm_src = nn.LayerNorm(d_model)

        # Positional embeddings
        # self.positional_embedding_peptide = nn.Embedding(max_seq_len, d_model)
        # self.positional_embedding_spectra = nn.Embedding(100, d_model)

        # Positional encoding layers
        self.positional_encoding_peptide = SinusoidalPositionalEncoding(d_model, max_seq_len)
        self.positional_encoding_spectra = SinusoidalPositionalEncoding(d_model, 100)

        # Output linear layer to predict peptide sequence
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Call weight initialization function
        self._init_weights()

    def _init_weights(self):
        # Initialize all linear layers with Xavier uniform
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, src, tgt, src_padding_mask, tgt_padding_mask, tgt_mask, training=True):
        # src: (batch_size, seq_len, 4)  --> spectra input
        # tgt: (batch_size, tgt_len) --> peptide sequence output (as indices)

        # Apply input embeddings for spectra
        src = self.input_embedding(src)

        # Apply layer normalization for the source embeddings
        src = self.layer_norm_src(src)

        # Apply sinusoidal positional encoding for spectra embedding
        src = self.positional_encoding_spectra(src)

        # Apply output embeddings for peptide sequence
        tgt = self.output_embedding(tgt)

        # Apply sinusoidal positional encoding for target (peptide sequence)
        tgt = self.positional_encoding_peptide(tgt)

        # Pass the input through the transformer layers
        output = self.transformer(src, tgt,
                                  src_key_padding_mask=src_padding_mask,
                                  tgt_key_padding_mask=tgt_padding_mask,
                                  memory_key_padding_mask=src_padding_mask,
                                  tgt_mask=tgt_mask)

        # Apply the dense layer between transformer blocks
        # output = self.dense(output)  # This is the extra dense layer

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