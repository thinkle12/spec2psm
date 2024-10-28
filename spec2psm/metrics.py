import numpy as np
from Levenshtein import distance as levenshtein_distance
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Helper function to calculate Levenshtein distance for each sequence in a batch
def calculate_levenshtein(predicted_seqs, target_seqs):
    # This function calculates levenshtein distances
    distances = [levenshtein_distance(pred_seq, tgt_seq) for pred_seq, tgt_seq in zip(predicted_seqs, target_seqs)]
    return np.mean(distances)

def calculate_total_correct_sequences(predicted_seqs, target_seqs):
    # This function returns the number of correct peptide sequences
    correct_sequences = sum(
        [1 for pred_seq, tgt_seq in zip(predicted_seqs, target_seqs) if
         pred_seq == tgt_seq])

    return correct_sequences

def mask_batch_after_stop_token(batch, stop_token, pad_token):
    # This function replaces all tokens after the first stop token with a pad token

    # Get a mask where the token is equal to the stop token
    stop_token_mask = (batch == stop_token)

    # Get the indices of the first stop token per sequence (along the last dimension)
    stop_indices = stop_token_mask.int().argmax(dim=1)

    # Create a mask for positions after the stop token (broadcasted)
    for i, idx in enumerate(stop_indices):
        batch[i, idx + 1:] = pad_token  # Mask all positions after the stop token with 0

    return batch

def calculate_aa_accuracy(predicted, target, eos_token, pad_token):
    # This function calculates aa level accuracy
    masked_predicted = mask_batch_after_stop_token(predicted, stop_token=eos_token, pad_token=pad_token)

    mask = (target != pad_token)  # Create a mask for non-pad tokens

    matches = (masked_predicted == target) & mask  # Masked comparison, preserving shape

    # Only count the matches where tgt_output is not the pad token
    correct_per_residue = matches.sum().item()

    total_residues = mask.sum().item()

    return correct_per_residue, total_residues

def plot_metrics(train_metrics, validation_metrics, filename):

    metrics = list(train_metrics.keys())

    overall_batches = range(len(train_metrics["loss"]))

    with PdfPages(filename) as pdf:
        for metric in metrics:

            train_values = train_metrics[metric]
            batches = list(range(len(train_values)))

            plt.figure()
            plt.plot(batches, train_values, label='Training {}'.format(metric))
            # plt.plot(batches, val_loss, label='Validation Loss')
            plt.xlabel('Batch')
            plt.ylabel(metric)
            plt.title('Training and Validation {} per Batch'.format(metric))
            plt.legend()
            pdf.savefig()  # Save this plot in the PDF
            plt.close()

        # Create a table of values
        metrics_table = np.array([overall_batches, train_metrics["loss"], train_metrics["per_residue_acc"], train_metrics["sequence_acc"], train_metrics["lev_dist"]]).T
        column_labels = ['Batch', 'Training Loss', 'Per Residue Accuracy', 'Sequence Accuracy', 'Levenshtein Distance']

        # Step 1: Filter by batches that are multiples of 10k
        filtered_table = metrics_table[metrics_table[:, 0] % 10000 == 0]

        # Also filter out the first 0 batch
        filtered_table = filtered_table[filtered_table[:, 0] > 0]

        # Step 2: If filtered data is empty, use the original data
        if len(filtered_table) == 0:
            filtered_table = metrics_table

        # Step 3: Sort the data by batch (column 0) in descending order
        sorted_table = filtered_table[filtered_table[:, 0].argsort()[::-1]]

        # Step 4: Select the top 10 rows (or fewer if there are not enough)
        if len(sorted_table)>10:
            top_10_table = sorted_table[:10]
        else:
            top_10_table = sorted_table

        top_10_table_rounded = np.round(top_10_table, 4)

        top_10_table_rounded[:, 0] = top_10_table_rounded[:, 0].astype(int)

        plt.figure(figsize=(8, 4))
        plt.axis('tight')
        plt.axis('off')
        table = plt.table(cellText=top_10_table_rounded, colLabels=column_labels, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)  # Adjust table size if necessary
        plt.title('Training Metrics Table', fontsize=14)

        pdf.savefig()  # Save the table as the final page of the PDF
        plt.close()