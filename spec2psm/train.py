import os
import torch
import torch.nn as nn
from itertools import islice
from spec2psm.metrics import Metrics
import logging

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_dataloader,
        optimizer,
        device,
        metric_window=100,
        val_dataloader=None,
        output_directory=os.getcwd(),
        val_per_x_batches=100000,
        plot_per_x_batches=10000,
        tag="plot",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        if optimizer:
            self.optimizer = optimizer.optimizer
            self.scheduler = optimizer.scheduler
        else:
            self.optimizer = None
            self.scheduler = None
        self.vocab_size = tokenizer.vocab_size
        self.device = device
        self.eos_token = tokenizer.eos_token
        self.pad_token = tokenizer.pad_token
        self.val_dataloader = val_dataloader
        self.val_per_x_batches = val_per_x_batches
        self.plot_per_x_batches = plot_per_x_batches
        self.output_directory = output_directory
        self.tag = tag
        self.metrics = Metrics(tokenizer=self.tokenizer, metric_window=metric_window)

    def train(self, num_epochs=1, model_filename="model.pth", checkpoint_per_x_batches=50_000, start_batch=0):

        # TODO wrap this with torch.profiler...
        # TODO add the spec2psm_cli.py script to setup.py or .cfg not sure so that on install we can just call spec2psm... :)
        # TODO upload models to huggingface hub and make sure thats working... I think do that with Small and then make sure inference and fine tuning works from that model in HF
        # TODO fine-tune the spec2psm models on 20M more PSMs from my other dataset...
        # TODO - Implement logic for parser cli tool
        # TODO - Think of other todos...
        # TODO make sure setting a mods.yaml file works... we can test this with the npgq param I think...
        # TODO write the readme and make sure it works...
        # TODO create the two inference modes, de novo and search bolstering mode
        # TODO implement idXML parsing for peptides for training

        self.model.train()  # Set the model to training mode

        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0

            for num_batches, batch in enumerate(islice(self.train_dataloader, start_batch, None), start=start_batch):
                # num_batches += 1
                loss, cur_metrics = self._train_step(batch)
                epoch_loss += loss

                # logger.info current metrics
                logger.info(
                    f"Epoch {epoch + 1}, Batch {num_batches}, Loss: {cur_metrics['train']['loss']:.4f}, "
                    f"Per-Residue Accuracy: {cur_metrics['train']['per_residue_accuracy']:.4f}, "
                    f"Sequence Accuracy: {cur_metrics['train']['sequence_accuracy']:.4f}, "
                    f"Levenshtein Distance: {cur_metrics['train']['levenshtein_distance']:.4f}"
                )

                # Validate every X batches
                if num_batches % self.val_per_x_batches == 0 and self.val_dataloader:
                    self.validate(num_batches)

                # Plot every X batches
                if num_batches % self.plot_per_x_batches == 0:
                    self._plot_metrics()

                # Save every X batches
                if num_batches % checkpoint_per_x_batches == 0:
                    self.model.save_model(
                        model_filename, iterations=num_batches, optimizer=self.optimizer, scheduler=self.scheduler
                    )

            avg_epoch_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch + 1}, Loss: {avg_epoch_loss:.4f}")

        # Save the model - In the future add this to the batch loop above to add checkpointing ever X batches
        # torch.save(self.model.state_dict(), model_filename
        self.model.save_model(
            model_filename, iterations=num_batches, optimizer=self.optimizer, scheduler=self.scheduler
        )
        return self.model

    def validate(self, training_batch):
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            num_batches = 0
            for batch in self.val_dataloader:
                num_batches += 1
                loss, cur_metrics = self._validate_step(batch)

                logger.info(
                    f"Train Batch {training_batch}, Val Batch {num_batches}, Loss: {cur_metrics['val']['loss']:.4f}, "
                    f"Per-Residue Accuracy: {cur_metrics['val']['per_residue_accuracy']:.4f}, "
                    f"Sequence Accuracy: {cur_metrics['val']['sequence_accuracy']:.4f}, "
                    f"Levenshtein Distance: {cur_metrics['val']['levenshtein_distance']:.4f}"
                )

            # Optionally, compute and log averaged validation metrics across batches
            avg_val_metrics = self.metrics.average_val_metrics()
            logger.info(f"Validation Metrics at Batch {training_batch}: {avg_val_metrics}")
        self.model.train()  # Set the model back to training mode

    def _train_step(self, batch):
        # Unpack batch
        spectra, peptide, precursor_mass, charge, precursor_masses = batch
        src = spectra.to(self.device)
        tgt = peptide.to(self.device)
        tgt_input = tgt[:, :-1]  # Teacher forcing: input is all but the last token
        tgt_output = tgt[:, 1:]  # Expected output is all but the first token
        target_seqs = tgt[:, 1:].view(-1, tgt[:, 1:].size(1)).tolist()

        # Prepare precursors
        precursors = torch.stack([precursor_mass, charge], dim=-1).to(self.device)

        # Padding masks and target masks
        tgt_padding_mask = tgt_input == self.pad_token
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(self.device)

        # Forward pass
        output_logits = self.model(src, tgt_input, tgt_padding_mask, tgt_mask, precursors=precursors)

        # Get the predicted sequence with greedy search
        predicted = torch.argmax(output_logits, dim=-1)

        # Reshape tgt and output for loss calculation
        output_logits_flat = output_logits.reshape(-1, self.vocab_size)
        tgt_output_flat = tgt_output.reshape(-1)

        # Convert to sequences for sequence-level accuracy and Levenshtein distance
        predicted_seqs = predicted.view(-1, tgt_input.size(1)).tolist()

        # Remove all tokens after stop tokens, for correct metric calculation
        truncated_pred = Metrics.truncate_sequences(predicted_seqs, self.eos_token)
        truncated_target = Metrics.truncate_sequences(target_seqs, self.eos_token)

        # Compute loss TODO swap this with a flag to do either regular ce loss or ce loss plus massdiff
        # loss = self.ce_loss(output_logits_flat, tgt_output_flat, weight_tensor=None)

        predicted_masses = self.tokenizer.peptide_manager.detokenize_and_calculate_mass(tokenized_sequences=truncated_pred, charges=charge)

        predicted_masses_tensor = torch.tensor(predicted_masses).to(self.device)
        loss = self.combined_loss(
                predictions=output_logits_flat,
                targets=tgt_output_flat,
                precursor_masses=precursor_mass.to(self.device),
                sequence_masses=predicted_masses_tensor,
                weight_tensor=None,
                cross_entropy_weight=0.5,
                mass_accuracy_weight=0.5)

        # Backpropagation
        loss.retain_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        # Metric calculations
        sequence_accuracy = self.metrics.calculate_sequence_accuracy(truncated_pred, truncated_target)
        levenshtein_distance = self.metrics.calculate_levenshtein(truncated_pred, truncated_target)
        per_residue_accuracy = self.metrics.calculate_aa_accuracy(predicted, tgt_output)

        self.metrics.add_train_metrics(
            loss=loss.item(),
            sequence_accuracy=sequence_accuracy,
            levenshtein_distance=levenshtein_distance,
            per_residue_accuracy=per_residue_accuracy,
        )

        cur_metrics = self.metrics.get_latest_ravg_metrics()
        return loss.item(), cur_metrics

    def _validate_step(self, batch):
        spectra, peptide, precursor_mass, charge, precursor_masses = batch
        src = spectra.to(self.device)
        tgt = peptide.to(self.device)  # This is used for the target ground truth
        batch_size = src.size(0)
        max_len = tgt.size(1)  # Maximum sequence length

        # Prepare initial input tokens (start token)
        generated_tokens = torch.full(
            (batch_size, 1),
            self.tokenizer.peptide_manager.token_map[self.tokenizer.peptide_manager.START_SYMBOL],
            dtype=torch.long,
        ).to(self.device)
        eos_mask = torch.zeros(batch_size, dtype=torch.bool).to(self.device)  # Tracks sequences that reached eos

        precursors = torch.stack([precursor_mass, charge], dim=-1).to(self.device)

        # Autoregressive generation loop
        for step in range(max_len - 1):
            tgt_mask_step = nn.Transformer.generate_square_subsequent_mask(generated_tokens.size(1)).to(self.device)

            # Get logits for the current step
            output_logits = self.model(
                src,
                generated_tokens,
                tgt_padding_mask=None,  # No padding mask for generated tokens
                tgt_mask=tgt_mask_step,
                precursors=precursors,
            )

            # Get the predicted tokens for this step
            predicted_tokens = torch.argmax(output_logits[:, -1, :], dim=-1)  # Last token in the sequence

            # Ensure EOS token is kept in the sequence, but pad after it
            predicted_tokens[eos_mask] = self.pad_token  # Pad further predictions after EOS
            eos_mask |= predicted_tokens == self.eos_token  # Update EOS mask for sequences reaching EOS

            # Append the predicted tokens to the generated sequence
            generated_tokens = torch.cat([generated_tokens, predicted_tokens.unsqueeze(1)], dim=1)

            # Break if all sequences have reached EOS
            if eos_mask.all():
                break

            # Pad all sequences to max_len if generation ended early
        if generated_tokens.size(1) < max_len:
            token_padding = torch.full(
                (batch_size, max_len - generated_tokens.size(1)),
                self.pad_token,
                dtype=torch.long,
                device=self.device,
            )
            # Pad the generated tokens and output logits so we can appropriately calculate metrics...
            generated_tokens = torch.cat([generated_tokens, token_padding], dim=1)
            padding_shape_logits = (output_logits.size(0), max_len - output_logits.size(1) - 1, output_logits.size(2))
            padding_logits = torch.zeros(padding_shape_logits, device=output_logits.device)
            output_logits = torch.cat([output_logits, padding_logits], dim=1)

        # Calculate the loss
        tgt_output = tgt[:, 1:]  # Remove start token from target
        generated_tokens = generated_tokens[:, 1:]  # Remove start token from predictions

        output_logits_flat = output_logits.reshape(-1, self.vocab_size)
        tgt_output_flat = tgt_output.reshape(-1)
        loss = self.ce_loss(output_logits_flat, tgt_output_flat, weight_tensor=None)

        # Metrics calculation
        truncated_pred = Metrics.truncate_sequences(generated_tokens.tolist(), self.eos_token)
        truncated_target = Metrics.truncate_sequences(tgt_output.tolist(), self.eos_token)

        sequence_accuracy = self.metrics.calculate_sequence_accuracy(truncated_pred, truncated_target)
        levenshtein_distance = self.metrics.calculate_levenshtein(truncated_pred, truncated_target)
        per_residue_accuracy = self.metrics.calculate_aa_accuracy(generated_tokens, tgt_output)

        # Update validation metrics
        self.metrics.add_val_metrics(
            loss=loss.item(),
            sequence_accuracy=sequence_accuracy,
            levenshtein_distance=levenshtein_distance,
            per_residue_accuracy=per_residue_accuracy,
        )

        # Fetch the latest rolling average metrics
        cur_metrics = self.metrics.get_latest_metrics()

        # Return loss and current metrics
        return loss.item(), cur_metrics

    # def _validate_step_autoregressive(self, batch):
    #     spectra, peptide, precursor_mass, charge, precursor_masses = batch
    #     src = spectra.to(self.device)
    #
    #     # This tgt is the ACTUAL peptide and not to be fed to the model, but used during metric calculations
    #     tgt = peptide.to(self.device)
    #     tgt_output = tgt[:, 1:]
    #
    #     precursors = torch.stack([precursor_mass, charge], dim=-1).to(self.device)
    #
    #     # Autoregressive generation
    #     max_len = tgt.size(1)  # Maximum sequence length
    #     batch_size = tgt.size(0)
    #     total_loss = 0
    #
    #     # Start each generated sequence with the Start Token
    #     generated = torch.full((batch_size, 1), self.tokenizer.peptide_manager.token_map[self.tokenizer.peptide_manager.START_SYMBOL], device=self.device, dtype=torch.long)
    #
    #     # Loop over the maximum possible peptide length
    #     for t in range(max_len - 1):
    #         tgt_padding_mask = (generated == self.pad_token)
    #         tgt_mask = nn.Transformer.generate_square_subsequent_mask(generated.size(1)).to(self.device)
    #
    #         # Forward pass
    #         output_logits = self.model(src, generated, tgt_padding_mask, tgt_mask, precursors=precursors)
    #         next_token = torch.argmax(output_logits[:, -1, :], dim=-1, keepdim=True)  # Get the next token
    #
    #         # Compare against the next target token
    #         target_token_flat = tgt[:, t + 1].reshape(-1)
    #         output_logits_flat = output_logits[:, -1, :].reshape(-1, self.vocab_size)
    #
    #
    #         # Compute loss for the current token
    #         loss = self.ce_loss(output_logits_flat, target_token_flat)
    #         total_loss += loss.item()
    #
    #         # Combine the next token with the already generated tokens
    #         generated = torch.cat([generated, next_token], dim=1)
    #
    #         # Stop if all sequences have generated EOS
    #         if (generated[:, -1] == self.eos_token).all():
    #             break
    #
    #     # Truncate sequences after EOS
    #     predicted_seqs = Metrics.truncate_sequences(generated.tolist(), self.eos_token)
    #     target_seqs = Metrics.truncate_sequences(tgt_output.tolist(), self.eos_token)
    #
    #     # Calculate metrics
    #     batch_metrics = {
    #         "sequence_accuracy": self.metrics.calculate_sequence_accuracy(predicted_seqs, target_seqs),
    #         "levenshtein_distance": self.metrics.calculate_levenshtein(predicted_seqs, target_seqs),
    #         "per_residue_accuracy": self.metrics.calculate_aa_accuracy(generated, tgt_output),
    #         "loss": total_loss / t
    #     }

    def ce_loss(self, predictions, targets, weight_tensor=None):
        # Cross-entropy loss
        if weight_tensor is not None:
            ce_loss = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=0, label_smoothing=0.1)(
                predictions, targets
            )
        else:
            ce_loss = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)(predictions, targets)

        return ce_loss

    def combined_loss(
        self,
        predictions,
        targets,
        precursor_masses,
        sequence_masses,
        weight_tensor=None,
        cross_entropy_weight=1.0,
        mass_accuracy_weight=0.0,
    ):
        # Cross-entropy loss
        # ce_loss = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)(predictions, targets)
        if weight_tensor is not None:
            ce_loss = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=0, label_smoothing=0.1)(
                predictions, targets
            )
        else:
            ce_loss = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)(predictions, targets)

        # Mass accuracy loss (mean squared error)
        precursor_masses_tensor = precursor_masses.view(-1)  # Ensure it's a 1D tensor
        # sequence_masses_tensor = torch.tensor(sequence_masses, dtype=torch.float32).to(targets.device)
        sequence_masses_tensor = sequence_masses.clone().detach().to(targets.device)

        # Calculate the mass accuracy loss
        mass_accuracy_loss = torch.mean((precursor_masses_tensor - sequence_masses_tensor) ** 2)

        # Combine the losses
        total_loss = (cross_entropy_weight * ce_loss) + (mass_accuracy_weight * mass_accuracy_loss)

        return total_loss

    def _plot_metrics(self):
        pdf_filename = f"{self.tag}.pdf"
        pdf_filepath = os.path.join(self.output_directory, pdf_filename)
        self.metrics.plot_metrics(filename=pdf_filepath)
