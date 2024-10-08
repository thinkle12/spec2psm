# This file is for defining the model, cost functions, etc

# TODO write the model here...
import torch
import torch.nn as nn

class Spec2Psm(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, nhead, num_encoder_layers, num_decoder_layers):

        super(Spec2Psm, self).__init__()

        # Transformer Encoder and Decoder
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          batch_first=True)

        # Embedding layers for input spectra and output peptide
        self.input_embedding = nn.Linear(4, d_model).to(torch.float32)  # Assuming 4 features in spectra
        self.output_embedding = nn.Embedding(vocab_size, d_model).to(torch.float32)

        # Positional embeddings
        self.positional_embedding = nn.Embedding(max_seq_len, d_model)

        # Additional dense layer between transformer blocks
        self.dense = nn.Sequential(
            nn.Linear(d_model, d_model),  # Example dense layer with same input/output dims
            nn.ReLU(),
            nn.Dropout(0.1)  # Add dropout to prevent overfitting
        )

        # Output linear layer to predict peptide sequence
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_padding_mask, tgt_padding_mask, tgt_mask, training=True):
        # src: (batch_size, seq_len, 4)  --> spectra input
        # tgt: (batch_size, tgt_len) --> peptide sequence output (as indices)

        # Apply input embeddings
        src = self.input_embedding(src)

        # Generate positional indices and add positional embeddings for source
        batch_size, seq_len, _ = src.size()
        src_positions = torch.arange(0, seq_len, device=src.device).unsqueeze(0).repeat(batch_size, 1)

        # Apply output embeddings for peptide sequence
        tgt = self.output_embedding(tgt)

        # Add positional embeddings to the target (peptide sequence)
        tgt_len = tgt.size(1)
        tgt_positions = torch.arange(0, tgt_len, device=tgt.device).unsqueeze(0).repeat(batch_size, 1)
        tgt = tgt + self.positional_embedding(tgt_positions)

        # Pass the input through the transformer layers
        output = self.transformer(src, tgt,
                                  src_key_padding_mask=src_padding_mask,
                                  tgt_key_padding_mask=tgt_padding_mask,
                                  memory_key_padding_mask=src_padding_mask,
                                  tgt_mask=tgt_mask)

        # Apply the dense layer between transformer blocks
        output = self.dense(output)  # This is the extra dense layer

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