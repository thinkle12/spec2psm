import numpy as np
from Levenshtein import distance as levenshtein_distance
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict


class Metrics:
    def __init__(self, tokenizer, metric_window=100):
        self.eos_token = tokenizer.eos_token
        self.pad_token = tokenizer.pad_token
        self.int_to_aa = {value: key for key, value in tokenizer.token_map.items()}
        self.metric_window = metric_window

        # Full and rolling average metrics
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        self.ravg_train_metrics = defaultdict(list)
        self.ravg_val_metrics = defaultdict(list)
        self.batch_average_val_metrics = defaultdict(list)

    def add_train_metrics(self, **kwargs):
        """Add metrics for the training set."""
        for key, value in kwargs.items():
            self.train_metrics[key].append(value)
        self._update_rolling_averages(self.train_metrics, self.ravg_train_metrics)

    def add_val_metrics(self, **kwargs):
        """Add metrics for the validation set."""
        for key, value in kwargs.items():
            self.val_metrics[key].append(value)
        self._update_rolling_averages(self.val_metrics, self.ravg_val_metrics)

    def average_val_metrics(self):
        """
        Computes the average of all stored validation metrics and appends to batch averages.
        """
        # Calculate average metrics across all batches
        avg_metrics = {key: sum(values) / len(values) for key, values in self.val_metrics.items()}

        # Update batch average validation metrics
        for key, value in avg_metrics.items():
            self.batch_average_val_metrics[key].append(value)

        return avg_metrics

    def _rolling_average(self, values):
        """Calculate rolling average over the last `metric_window` elements."""
        if len(values) > self.metric_window:
            return sum(values[-self.metric_window :]) / self.metric_window
        return sum(values) / len(values)

    def _update_rolling_averages(self, metrics, ravg_metrics):
        """Update rolling averages for a given set of metrics."""
        for key, values in metrics.items():
            ravg_metrics[key].append(self._rolling_average(values))

    def get_latest_metrics(self):
        """Retrieve the latest metrics for both train and validation."""
        return {
            "train": {key: values[-1] for key, values in self.train_metrics.items()},
            "val": {key: values[-1] for key, values in self.val_metrics.items()},
        }

    def get_latest_ravg_metrics(self):
        """Retrieve the latest rolling average metrics for both train and validation."""
        return {
            "train": {key: values[-1] for key, values in self.ravg_train_metrics.items()},
            "val": {key: values[-1] for key, values in self.ravg_val_metrics.items()},
        }

    def calculate_levenshtein(self, predicted_seqs, target_seqs):
        distances = [levenshtein_distance(pred_seq, tgt_seq) for pred_seq, tgt_seq in zip(predicted_seqs, target_seqs)]
        return np.mean(distances)

    def calculate_sequence_accuracy(self, predicted_seqs, target_seqs):
        correct_sequences = sum(1 for pred_seq, tgt_seq in zip(predicted_seqs, target_seqs) if pred_seq == tgt_seq)
        total_sequences = len(predicted_seqs)
        sequence_accuracy = correct_sequences / total_sequences
        return float(sequence_accuracy)

    def mask_batch_after_stop_token(self, batch):
        stop_token_mask = batch == self.eos_token
        stop_indices = stop_token_mask.int().argmax(dim=1)

        for i, idx in enumerate(stop_indices):
            batch[i, idx + 1 :] = self.pad_token

        return batch

    def calculate_aa_accuracy(self, predicted, target):
        masked_predicted = self.mask_batch_after_stop_token(predicted)
        mask = target != self.pad_token

        matches = (masked_predicted == target) & mask
        correct_per_residue = matches.sum().item()
        total_residues = mask.sum().item()

        aa_accuracy = correct_per_residue / total_residues
        return aa_accuracy

    def calculate_aa_substitutions(self, seqs1, seqs2):
        substitution_counts = defaultdict(int)

        for s1, s2 in zip(seqs1, seqs2):
            for a, b in zip(s1, s2):
                if a != b:
                    substitution_counts[(self.int_to_aa[a], self.int_to_aa[b])] += 1

        sorted_substitutions = sorted(substitution_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_substitutions

    def plot_metrics(self, filename):
        metrics = list(self.ravg_train_metrics.keys())
        overall_batches = range(len(self.ravg_train_metrics["loss"]))

        with PdfPages(filename) as pdf:
            for metric in metrics:
                train_values = self.ravg_train_metrics[metric]
                batches = list(range(len(train_values)))

                plt.figure()
                plt.plot(batches, train_values, label='Training {}'.format(metric))
                if metric == "levenshtein_distance":
                    plt.ylim(0, 20)
                plt.xlabel('Batch')
                plt.ylabel(metric)
                plt.title('Training and Validation {} per Batch'.format(metric))
                plt.legend()
                pdf.savefig()
                plt.close()

            metrics_table = np.array(
                [
                    overall_batches,
                    self.ravg_train_metrics["loss"],
                    self.ravg_train_metrics["per_residue_accuracy"],
                    self.ravg_train_metrics["sequence_accuracy"],
                    self.ravg_train_metrics["levenshtein_distance"],
                ]
            ).T

            column_labels = [
                'Batch',
                'Training Loss',
                'Per Residue Accuracy',
                'Sequence Accuracy',
                'Levenshtein Distance',
            ]
            filtered_table = metrics_table[metrics_table[:, 0] % 10000 == 0]
            filtered_table = filtered_table[filtered_table[:, 0] > 0]

            if len(filtered_table) == 0:
                filtered_table = metrics_table

            sorted_table = filtered_table[filtered_table[:, 0].argsort()[::-1]]
            top_10_table = sorted_table[:10] if len(sorted_table) > 10 else sorted_table
            top_10_table_rounded = np.round(top_10_table, 4)
            top_10_table_rounded[:, 0] = top_10_table_rounded[:, 0].astype(int)

            plt.figure(figsize=(8, 4))
            plt.axis('tight')
            plt.axis('off')
            table = plt.table(cellText=top_10_table_rounded, colLabels=column_labels, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.2)
            plt.title('Training Metrics Table', fontsize=14)

            pdf.savefig()
            plt.close()

    @classmethod
    def truncate_sequences(cls, sequences, stop_token):
        truncated_sequences = []
        for seq in sequences:
            if stop_token in seq:
                # Find the index of the first occurrence of the stop token
                index = seq.index(stop_token)
                # Append the truncated sequence including the stop token to the list
                truncated_sequences.append(seq[: index + 1])
            else:
                # If the stop token is not found, keep the whole sequence
                truncated_sequences.append(seq)
        return truncated_sequences
