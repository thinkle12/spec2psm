import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader


class Inference:
    def __init__(self, model, dataset, tokenizer, device, batch_size=1):
        """
        Initialize the Inference class for autoregressive decoding.

        Args:
            model: The trained transformer model for peptide prediction.
            dataset: An instance of the Dataset class for input transformation.
            device: The device to run inference on (e.g., "cpu" or "cuda").
            batch_size: Batch size for DataLoader.
        """
        self.model = model
        self.model.eval()  # Set model to evaluation mode
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.sos_token = tokenizer.sos_token
        self.eos_token = tokenizer.eos_token
        self.max_seq_len = tokenizer.max_peptide_length
        self.vocab = tokenizer.vocab
        self.device = device
        self.batch_size = batch_size

        # Initialize DataLoader for batch processing
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def decode(self, spectra_batch, precursor_batch, mass, known_peptide):
        """
        Perform autoregressive decoding for a batch of spectra.
        """
        # Initialize <SOS> token for all sequences in the batch
        tgt_input = torch.full((spectra_batch.size(0), 1), self.sos_token, dtype=torch.long, device=self.device)
        generated_tokens = []
        entropies = []

        # Initialize EOS mask (True means EOS has been reached for that sequence)
        eos_reached = torch.zeros(spectra_batch.size(0), dtype=torch.bool, device=self.device)

        for _ in range(self.max_seq_len):
            with torch.no_grad():
                # Forward pass for the batch
                output_logits = self.model(
                    src=spectra_batch, tgt=tgt_input, tgt_padding_mask=None, tgt_mask=None, precursors=precursor_batch
                )

            # Get logits for the last predicted token in each sequence
            next_token_logits = output_logits[:, -1, :]  # Logits for the last token in batch
            next_token_probs = F.softmax(next_token_logits, dim=-1)  # Convert logits to probabilities

            # Autoregressive prediction: Greedy decoding (most probable token)
            next_tokens = torch.argmax(next_token_probs, dim=-1)

            # Calculate entropy for the batch (for each sequence)
            entropy = -torch.sum(next_token_probs * torch.log(next_token_probs + 1e-10), dim=-1).cpu().numpy()
            entropies.append(entropy)

            # Store the predicted tokens for the batch
            generated_tokens.extend(next_tokens.cpu().numpy())

            # Check if <EOS> is predicted for each sequence
            eos_reached |= next_tokens == self.eos_token

            # Stop if all sequences have predicted <EOS>
            if eos_reached.all():
                break

            # Update tgt_input by appending the predicted tokens
            tgt_input = torch.cat([tgt_input, next_tokens.unsqueeze(1)], dim=1)

        # Convert generated token IDs to peptide strings
        peptide = self.tokenizer.peptide_manager.reconstruct_peptide_sequence(generated_tokens)
        calculated_mass = self.tokenizer.peptide_manager.calculate_peptide_mass(peptide)
        known_peptide = [int(x) for x in known_peptide]
        known_peptide_str = self.tokenizer.peptide_manager.reconstruct_peptide_sequence(known_peptide)

        # TODO - Make sure this output is what we want... We probably want average entropies as well
        # TODO - This is also de-novo mode. We need to write the other mode too (Figure out a name... where we want to integrate with percolator)
        return peptide, generated_tokens, entropies, calculated_mass, float(mass), known_peptide_str

    def beam_search_decode(self, spectra_batch, precursor_batch, mass, known_peptide, beam_width=3, temperature=1.0):
        batch_size = spectra_batch.size(0)
        device = self.device

        # Initial beam: start with <SOS>
        beams = [
            (
                torch.full((batch_size, 1), self.sos_token, dtype=torch.long, device=device),  # Sequence
                0.0,  # Score
                [[] for _ in range(batch_size)],  # Entropy scores
            )
        ]

        completed_sequences = [[] for _ in range(batch_size)]  # Store completed sequences

        # Initialize active_beams to match the size of beams
        active_beams = [True] * len(beams)

        for step in range(self.max_seq_len):
            new_beams = []

            for beam_idx, (seq, score, entropy_scores) in enumerate(beams):
                if not active_beams[beam_idx]:  # Skip inactive beams
                    new_beams.append((seq, score, entropy_scores))  # Keep the beam as-is
                    continue

                # Check if this beam has reached <EOS>
                if seq[:, -1].eq(self.eos_token).all():  # All batch sequences in this beam reached <EOS>
                    for batch_idx in range(batch_size):
                        completed_sequences[batch_idx].append((seq[batch_idx], score, entropy_scores[batch_idx]))
                    active_beams[beam_idx] = False  # Mark this beam as inactive
                    new_beams.append((seq, score, entropy_scores))
                    continue

                # Forward pass

                tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq.size(1)).to(self.device)
                # tgt_padding_mask = (seq == 0).to(device)

                with torch.no_grad():
                    output_logits = self.model(
                        src=spectra_batch, tgt=seq, tgt_padding_mask=None, tgt_mask=tgt_mask, precursors=precursor_batch
                    )

                # Get probabilities for the next token
                next_token_logits = output_logits[:, -1, :]

                # Apply temperature scaling to logits
                scaled_logits = next_token_logits / temperature  # Adjust logits with temperature
                next_token_probs = F.softmax(scaled_logits, dim=-1)

                # Calculate entropy
                batch_entropy = -torch.sum(next_token_probs * torch.log(next_token_probs + 1e-10), dim=-1).cpu().numpy()

                # Top-k predictions
                topk_probs, topk_tokens = next_token_probs.topk(beam_width, dim=-1)

                # Prevent selection of stop token if sequence length < 5
                if seq.shape[1] < 5:
                    stop_token_mask = topk_tokens.eq(self.eos_token)
                    topk_probs.masked_fill_(stop_token_mask, float('-inf'))

                # Expand beams for each token in top-k
                for i in range(beam_width):
                    new_seq = torch.cat([seq, topk_tokens[:, i].unsqueeze(1)], dim=1)
                    new_score = score + topk_probs[:, i].item()
                    new_entropy_scores = [
                        entropy_scores[batch_idx] + [batch_entropy[batch_idx]] for batch_idx in range(batch_size)
                    ]
                    new_beams.append((new_seq, new_score, new_entropy_scores))

            # Select top beam_width beams across all sequences
            new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            beams = new_beams
            active_beams = [True] * len(beams)  # Reset active_beams to match new beam size

            # Stop if all beams are inactive
            if not any(active_beams):
                break

        # Finalize output
        final_sequences = []
        final_entropies = []
        final_tokens = []
        final_masses = []
        precursor_masses = []
        final_known_peptides = []

        for batch_idx in range(batch_size):
            # Select the sequence with the highest score
            if completed_sequences[batch_idx]:
                completed_sequences[batch_idx].sort(key=lambda x: x[1], reverse=True)
                best_seq, _, entropy_scores = completed_sequences[batch_idx][0]
            else:
                # If no sequence reached <EOS>, take the best unfinished beam
                best_seq, _, entropy_scores = beams[0]

            # Convert token IDs to peptide sequence
            peptide = self.tokenizer.peptide_manager.reconstruct_peptide_sequence(best_seq.cpu().numpy().flatten())
            calculated_mass = self.tokenizer.peptide_manager.calculate_peptide_mass(peptide)
            known_peptide = self.tokenizer.peptide_manager.reconstruct_peptide_sequence(
                [int(x) for x in known_peptide[0]]
            )

            # Store final sequence and entropies
            final_sequences.append(peptide)
            final_entropies.append(entropy_scores)
            final_tokens.append(best_seq)
            final_masses.append(calculated_mass)
            precursor_masses.append(mass[batch_idx])
            final_known_peptides.append(known_peptide)

        return final_sequences, final_tokens, final_entropies, final_masses, precursor_masses, known_peptide

    def bulk_decode(self, spectra_batch, precursor_batch, mass, known_peptide):
        """
        Perform non-autoregressive decoding for a batch of spectra.
        :param spectra_batch: Tensor of input spectra (batch_size, input_dim).
        :param precursor_batch: Tensor of precursor masses (batch_size, precursor_dim).
        :return: Tuple (decoded_sequences, entropy_scores)
        """
        batch_size = spectra_batch.size(0)
        tgt_input = torch.full((spectra_batch.size(0), 1), self.sos_token, dtype=torch.long, device=self.device)

        # Forward pass through the model
        with torch.no_grad():
            output_logits = self.model(
                src=spectra_batch, tgt=tgt_input, tgt_padding_mask=None, tgt_mask=None, precursors=precursor_batch
            )  # Shape: (batch_size, max_seq_len, vocab_size)

        # Apply softmax to obtain probabilities
        output_probs = F.softmax(output_logits, dim=-1)  # Shape: (batch_size, max_seq_len, vocab_size)

        # Get the token predictions (argmax for non-autoregressive decoding)
        predicted_tokens = output_probs.argmax(dim=-1)  # Shape: (batch_size, max_seq_len)

        # Compute entropy for each position in the sequence
        token_entropies = -torch.sum(
            output_probs * torch.log(output_probs + 1e-10), dim=-1
        )  # Shape: (batch_size, max_seq_len)

        # Process predictions to handle padding and EOS
        decoded_sequences = []
        entropy_scores = []

        for batch_idx in range(batch_size):
            sequence = []
            entropies = []
            for token_idx in range(self.max_seq_len):
                token = predicted_tokens[batch_idx, token_idx].item()
                entropy = token_entropies[batch_idx, token_idx].item()

                if token == self.eos_token:  # Stop at EOS token
                    break
                if token != 0:  # Skip padding tokens
                    sequence.append(token)
                    entropies.append(entropy)

            # Append decoded sequence and entropy score for this batch index
            decoded_sequences.append(sequence)
            entropy_scores.append(entropies)

        peptide = self.tokenizer.peptide_manager.reconstruct_peptide_sequence(predicted_tokens[0])
        calculated_mass = self.tokenizer.peptide_manager.calculate_peptide_mass(peptide)
        known_peptide = [int(x) for x in known_peptide]
        known_peptide_str = self.tokenizer.peptide_manager.reconstruct_peptide_sequence(known_peptide)

        # TODO - Make sure this output is what we want... We probably want average entropies as well
        # TODO - This is also de-novo mode. We need to write the other mode too (Figure out a name... where we want to integrate with percolator)
        return peptide, predicted_tokens, token_entropies, calculated_mass, float(mass), known_peptide_str

    def run_inference(self, beam=True):
        """
        Run inference for the entire dataset using the DataLoader.

        Returns:
            List of tuples, each containing:
                - Predicted peptide string.
                - List of token IDs.
                - List of entropy scores for each peptide.
        """
        all_predictions = []

        for batch_idx, batch in enumerate(self.dataloader):
            spectra, known_peptide, precursor_mass, charge, precursor_masses = batch

            precursor_batch = torch.stack([precursor_mass, charge], dim=-1).to(self.device)
            spectra_batch = spectra.to(self.device)

            # Perform decoding for the current batch
            if beam:
                peptides, tokens, entropies, calc_mass, known_mass, known_peptide_str = self.beam_search_decode(
                    spectra_batch, precursor_batch, precursor_masses, known_peptide, beam_width=3, temperature=1.0
                )
            else:
                peptides, tokens, entropies, calc_mass, known_mass, known_peptide_str = self.decode(
                    spectra_batch, precursor_batch, precursor_masses, known_peptide
                )
            massdiff = calc_mass[0] - float(known_mass[0])
            all_predictions.append(
                [peptides, known_peptide_str, massdiff, calc_mass, known_mass, tokens, known_peptide, entropies]
            )

        return all_predictions

    def tokens_to_peptide(self, tokens):
        """
        Convert a list of token IDs into a peptide string.

        Args:
            tokens: List of token IDs.

        Returns:
            Peptide string.
        """
        return ''.join(self.vocab[token] for token in tokens if token != self.eos_token)

    def calculate_entropy_for_peptide(self, target_tokens, spectra, precursors):
        """
        Calculate entropy for a given target peptide sequence using autoregressive predictions.

        Args:
            target_tokens: List of token IDs for the target peptide.
            spectra: Input spectrum.
            precursors: Precursor mass info.

        Returns:
            The entropy score for the target peptide.
        """
        # Prepare initial input with <SOS> token for autoregressive generation
        tgt_input = torch.full((1, 1), self.sos_token, dtype=torch.long, device=self.device)

        # Initialize sequence and entropy tracking
        token_probs = []  # To track probabilities for entropy calculation

        # Autoregressive generation loop
        for i in range(len(target_tokens)):
            with torch.no_grad():
                # Perform model inference
                output_logits = self.model(
                    src=spectra.unsqueeze(0),  # Add batch dimension
                    tgt=tgt_input,  # Use current sequence as input
                    src_key_padding_mask=None,
                    tgt_key_padding_mask=None,
                    precursors=precursors,
                )

            # Get the token prediction for the current step
            next_token_logits = output_logits[:, -1, :]  # Logits for the last predicted token
            next_token_probs = F.softmax(next_token_logits, dim=-1)  # Convert logits to probabilities

            # Track the probabilities for entropy calculation
            token_probs.append(next_token_probs)

            # Update tgt_input by appending the predicted token for autoregressive generation
            next_token = torch.argmax(next_token_probs, dim=-1).item()  # Greedy decoding
            next_token_tensor = torch.tensor([[next_token]], device=self.device)
            tgt_input = torch.cat([tgt_input, next_token_tensor], dim=1)

        # Now calculate the entropy for the target peptide by comparing probabilities
        entropy = 0.0
        for i, prob in enumerate(token_probs):
            target_token = target_tokens[i]  # The true token at position i in the target peptide
            target_token_prob = prob[0, target_token].item()  # Probability of the true token at this position
            entropy -= torch.log(target_token_prob + 1e-10)  # Avoid log(0) with small constant

        return entropy
