import os
import torch
import torch.nn as nn
from spec2psm.utils import truncate_sequences, truncate_after_eos
from spec2psm.preprocess import TOKEN_TO_MASS
from spec2psm.metrics import calculate_levenshtein, calculate_total_correct_sequences, calculate_aa_accuracy, plot_metrics

# TODO need to add test set validation every X batches...
# TODO we would just pass two dataloaders I think, One train dataloader and one test dataloader
def train_model(model, model_filename, train_dataloader, optimizer, scheduler, vocab_size, device, val_dataloader=None, num_epochs=1, weight_tensor=None, eos_token=26, pad_token=0, val_per_x_batches=100000, plot_per_x_batches=10000, tag="plot", output_directory=os.getcwd()):
    model.train()  # Set the model to training mode
    losses = []
    per_residue_accuracies = []
    sequence_accuracies = []
    levenshtein_distances = []
    train_metrics = {}
    val_metrics = {
        "training_batch": [],
        "loss": [],
        "per_residue_acc": [],
        "sequence_acc": [],
        "lev_dist": []
    }
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        correct_per_residue = 0
        total_residues = 0
        correct_sequences = 0
        total_sequences = 0
        total_levenshtein_distance = 0

        for batch in train_dataloader:
            num_batches = num_batches + 1
            src = batch[0].to(device)  # (batch_size, seq_len, 4) --> input spectra
            tgt = batch[1].to(device)  # (batch_size, tgt_len) --> target peptide sequences
            precursors = torch.stack([batch[2], batch[3]], dim=-1).to(device)
            precursor_masses = batch[4]
            precursor_masses = precursor_masses.clone().detach().float().to(device)

            tgt_input = tgt[:, :-1]  # Teacher forcing: input is all but the last token
            tgt_output = tgt[:, 1:]  # Expected output is all but the first token

            # Padding masks and target masks
            tgt_padding_mask = (tgt_input == 0)  # Mask out padding tokens in tgt
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(device)

            # Forward pass
            output_logits = model(src, tgt_input, tgt_padding_mask, tgt_mask, precursors=precursors)

            # Step 1: Get the predicted tokens
            _, predicted = torch.max(output_logits, dim=-1)  # Shape: (batch_size, seq_len)

            output_logits = output_logits.reshape(-1, vocab_size)
            tgt_output_flat = tgt_output.reshape(-1)

            # Convert to sequences for sequence-level accuracy and Levenshtein distance
            predicted_seqs = predicted.view(-1, tgt_input.size(1)).tolist()
            target_seqs = tgt[:, 1:].view(-1, tgt[:, 1:].size(1)).tolist()

            truncated_sequences = truncate_sequences(predicted_seqs, eos_token)
            truncated_target_sequences = truncate_sequences(target_seqs, eos_token)

            # Calculate the mass for each sequence
            sequence_masses = []

            for sequence in truncated_sequences:
                total_mass = 0
                for token in sequence:
                    if token in TOKEN_TO_MASS:
                        total_mass += TOKEN_TO_MASS[token]
                    else:
                        pass
                sequence_masses.append(total_mass)

            # Compute loss (CrossEntropyLoss expects logits, not probabilities)
            loss = combined_loss(output_logits, tgt_output_flat, precursor_masses, sequence_masses, weight_tensor)

            # Reset gradients after updating
            loss.retain_grad()
            optimizer.zero_grad()

            # Backpropagation
            loss.backward()

            # Step
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            # Calculate sequence-level accuracy
            # Fixed to use truncated sequences
            correct_sequences += calculate_total_correct_sequences(truncated_sequences, truncated_target_sequences)
            total_sequences += len(predicted_seqs)

            # Calculate Levenshtein distance
            total_levenshtein_distance += calculate_levenshtein(truncated_sequences, truncated_target_sequences)

            # Calculate AA accuracy
            cur_aa_matches, cur_aa_total = calculate_aa_accuracy(predicted=predicted, target=tgt_output, eos_token=eos_token, pad_token=pad_token)
            correct_per_residue += cur_aa_matches
            total_residues += cur_aa_total

            avg_loss = epoch_loss / num_batches
            per_residue_accuracy = correct_per_residue / total_residues if total_residues > 0 else 0.0
            sequence_accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0.0
            avg_levenshtein_distance = total_levenshtein_distance / num_batches

            print(f"Epoch {epoch + 1}, Batch {num_batches}, Loss: {avg_loss:.4f}, "
                  f"Per-Residue Accuracy: {per_residue_accuracy:.4f}, "
                  f"Sequence Accuracy: {sequence_accuracy:.4f}, "
                  f"Levenshtein Distance: {avg_levenshtein_distance:.4f}")

            # Append metrics for plotting
            losses.append(avg_loss)
            per_residue_accuracies.append(per_residue_accuracy)
            sequence_accuracies.append(sequence_accuracy)
            levenshtein_distances.append(avg_levenshtein_distance)

            # Validate every X batches
            if num_batches % val_per_x_batches == 0 and val_dataloader:
                val_losses, val_per_residue_accuracies, val_sequence_accuracies, val_levenshtein_distances = validate_model(model=model, val_dataloader=val_dataloader, vocab_size=vocab_size, device=device, eos_token=eos_token, pad_token=pad_token, weight_tensor=weight_tensor)

                val_metrics["training_batch"].append(num_batches)
                val_metrics["loss"].append(val_losses)
                val_metrics["per_residue_acc"].append(val_per_residue_accuracies)
                val_metrics["sequence_acc"].append(val_sequence_accuracies)
                val_metrics["lev_dist"].append(val_levenshtein_distances)

            # Plot every X batches
            if num_batches % plot_per_x_batches == 0:
                train_metrics = {
                    "loss": losses,
                    "per_residue_acc": per_residue_accuracies,
                    "sequence_acc": sequence_accuracies,
                    "lev_dist": levenshtein_distances
                }
                pdf_filename = "{}.pdf".format(tag)
                pdf_filepath = os.path.join(output_directory, pdf_filename)
                plot_metrics(train_metrics=train_metrics, validation_metrics=val_metrics, filename=pdf_filepath)



        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}, Loss: {avg_epoch_loss:.4f}")

    torch.save(model.state_dict(), model_filename)
    return model, losses, per_residue_accuracies, sequence_accuracies, levenshtein_distances

def validate_model(model, val_dataloader, vocab_size, device, eos_token, pad_token, weight_tensor=None):
    model.eval()  # Set the model to evaluation mode

    val_loss = 0
    num_batches = 0
    correct_per_residue = 0
    total_residues = 0
    correct_sequences = 0
    total_sequences = 0
    total_levenshtein_distance = 0

    val_losses = []
    val_per_residue_accuracies = []
    val_sequence_accuracies = []
    val_levenshtein_distances = []

    with torch.no_grad():  # Disable gradient calculation
        for batch in val_dataloader:
            num_batches += 1

            # Move batch data to device (GPU/CPU)
            src = batch[0].to(device)  # Input spectra
            tgt = batch[1].to(device)  # Target peptide sequences
            precursors = torch.stack([batch[2], batch[3]], dim=-1).to(device)
            precursor_masses = batch[4].clone().detach().float().to(device)

            tgt_input = tgt[:, :-1]  # Teacher forcing: input is all but the last token
            tgt_output = tgt[:, 1:]  # Expected output is all but the first token

            # Padding masks and target masks
            tgt_padding_mask = (tgt_input == 0)  # Mask out padding tokens in tgt
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(device)

            # Forward pass
            output_logits = model(src, tgt_input, tgt_padding_mask, tgt_mask, precursors=precursors)

            # Get the predicted tokens
            _, predicted = torch.max(output_logits, dim=-1)

            output_logits = output_logits.reshape(-1, vocab_size)
            tgt_output_flat = tgt_output.reshape(-1)

            # Convert to sequences for sequence-level accuracy and Levenshtein distance
            predicted_seqs = predicted.view(-1, tgt_input.size(1)).tolist()
            target_seqs = tgt[:, 1:].view(-1, tgt[:, 1:].size(1)).tolist()

            truncated_sequences = truncate_sequences(predicted_seqs, eos_token)
            truncated_target_sequences = truncate_sequences(target_seqs, eos_token)

            # Calculate the mass for each sequence
            sequence_masses = []
            for sequence in truncated_sequences:
                total_mass = 0
                for token in sequence:
                    if token in TOKEN_TO_MASS:
                        total_mass += TOKEN_TO_MASS[token]
                    else:
                        pass
                sequence_masses.append(total_mass)

            # Compute loss
            loss = combined_loss(output_logits, tgt_output_flat, precursor_masses, sequence_masses, weight_tensor)

            val_loss += loss.item()

            # Calculate sequence-level accuracy
            correct_sequences += calculate_total_correct_sequences(truncated_sequences, truncated_target_sequences)
            total_sequences += len(predicted_seqs)

            # Calculate Levenshtein distance
            total_levenshtein_distance += calculate_levenshtein(truncated_sequences, truncated_target_sequences)

            # Calculate AA accuracy
            cur_aa_matches, cur_aa_total = calculate_aa_accuracy(predicted=predicted, target=tgt_output, eos_token=eos_token, pad_token=pad_token)
            correct_per_residue += cur_aa_matches
            total_residues += cur_aa_total

            # Update validation metrics per batch
            avg_val_loss = val_loss / num_batches
            val_per_residue_accuracy = correct_per_residue / total_residues if total_residues > 0 else 0.0
            val_sequence_accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0.0
            avg_val_levenshtein_distance = total_levenshtein_distance / num_batches

            # Append validation metrics for plotting
            val_losses.append(avg_val_loss)
            val_per_residue_accuracies.append(val_per_residue_accuracy)
            val_sequence_accuracies.append(val_sequence_accuracy)
            val_levenshtein_distances.append(avg_val_levenshtein_distance)

        # Average validation metrics for plotting
        avg_val_loss_across_set = sum(val_losses) / len(val_losses)
        avg_val_per_residue_accuracy_across_set = sum(val_per_residue_accuracies) / len(val_per_residue_accuracies)
        avg_val_sequence_accuracy_across_set = sum(val_sequence_accuracies) / len(val_sequence_accuracies)
        avg_val_levenshtein_distance_across_set = sum(val_levenshtein_distances) / len(val_levenshtein_distances)


    # Return validation metrics for plotting
    return avg_val_loss_across_set, avg_val_per_residue_accuracy_across_set, avg_val_sequence_accuracy_across_set, avg_val_levenshtein_distance_across_set


def combined_loss(predictions, targets, precursor_masses, sequence_masses, weight_tensor=None, cross_entropy_weight=1.0, mass_accuracy_weight=0.0):
    # Cross-entropy loss
    # ce_loss = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)(predictions, targets)
    if weight_tensor is not None:
        ce_loss = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=0, label_smoothing=0.1)(predictions, targets)
    else:
        ce_loss = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)(predictions, targets)

    # Mass accuracy loss (mean squared error)
    precursor_masses_tensor = precursor_masses.view(-1)  # Ensure it's a 1D tensor
    sequence_masses_tensor = torch.tensor(sequence_masses, dtype=torch.float32).to(targets.device)

    # Calculate the mass accuracy loss
    mass_accuracy_loss = torch.mean((precursor_masses_tensor - sequence_masses_tensor) ** 2)

    # Combine the losses
    total_loss = (cross_entropy_weight * ce_loss) + (mass_accuracy_weight * mass_accuracy_loss)

    return total_loss