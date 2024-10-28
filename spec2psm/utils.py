import torch

def truncate_after_eos(sequences, eos_token):
    truncated_sequences = []

    for seq in sequences:
        truncated_seq = []
        try:
            eos_index = seq.index(eos_token)
            if eos_index != -1:
                truncated_seq.append(seq[:eos_index + len(eos_token)])  # Include the <eos> token
            else:
                truncated_seq.append(seq)  # If <eos> not found, keep the original sequence
            truncated_sequences.append(truncated_seq)  # Append the truncated sublist
        except:
            truncated_sequences.append(seq)

    return truncated_sequences

def truncate_sequences(sequences, stop_token):
    truncated_sequences = []
    for seq in sequences:
        if stop_token in seq:
            # Find the index of the first occurrence of the stop token
            index = seq.index(stop_token)
            # Append the truncated sequence including the stop token to the list
            truncated_sequences.append(seq[:index + 1])
        else:
            # If the stop token is not found, keep the whole sequence
            truncated_sequences.append(seq)
    return truncated_sequences

def get_device(device_name):
    if device_name == "mps":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device_name == "cpu":
        device = torch.device("cpu")
    if device_name == "gpu" or device_name == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device
def calculate_aa_loss_weights(device):

    # These are the frequencies from the MassiveKB file
    amino_acid_frequencies = {21: 0.3395053140995869,
                              9: 6.67402677729377,
                              1: 4.338166818344472,
                              10: 6.645411129607654,
                              18: 5.590948819359133,
                              6: 2.3348583779830445,
                              12: 2.9140132484921444,
                              4: 4.089718799351087,
                              8: 5.75976582397702,
                              16: 8.119863509553824,
                              13: 6.088987618607931,
                              3: 7.681047180603522,
                              15: 5.468709325340789,
                              22: 1.044902006823297,
                              7: 3.883270586461217,
                              11: 5.370594347335413,
                              5: 4.705109057849412,
                              14: 2.595217485515268,
                              19: 3.32476400015402,
                              20: 1.4547737113605348,
                              2: 0.7872932512795231,
                              23: 0.5598131699879088}

    # These are AA frequencies in nature
    # amino_acid_frequencies = {
    #     16: 9.6,  # 'L' -> 16
    #     9: 8.2,   # 'A' -> 9
    #     13: 7.2,  # 'G' -> 13
    #     10: 7.0,  # 'S' -> 10
    #     18: 6.9,  # 'V' -> 18
    #     3: 6.5,   # 'E' -> 3
    #     7: 5.6,   # 'I' -> 7
    #     8: 5.0,   # 'P' -> 8
    #     5: 5.4,   # 'T' -> 5
    #     11: 5.4,  # 'D' -> 11
    #     15: 5.9,  # 'K' -> 15
    #     4: 5.5,   # 'R' -> 4
    #     12: 4.0,  # 'F' -> 12
    #     6: 3.2,   # 'Y' -> 6
    #     19: 4.1,  # 'N' -> 19
    #     1: 4.0,   # 'Q' -> 1
    #     14: 2.3,  # 'H' -> 14
    #     20: 2.4,  # 'M' -> 20
    #     2: 1.3,   # 'W' -> 2
    #     17: 1.3   # 'C' -> 17
    # }

    total_frequency = sum(amino_acid_frequencies.values())
    class_weights = {aa: total_frequency / freq for aa, freq in amino_acid_frequencies.items()}

    extra_weights = {
        17: 1,  # Cysteine no mod
        24: 1,  # Unknown
        25: 1,  # Start
        26: 1,  # End
        0: 1  # Pad
    }

    class_weights.update(extra_weights)

    max_weight = max(class_weights.values())
    min_weight = min(class_weights.values())

    # Scale weights to a more reasonable range
    scaled_weights = {k: (v - min_weight) / (max_weight - min_weight) + 1 for k, v in class_weights.items()}

    # Convert the weights into a tensor that can be used in CrossEntropyLoss
    # Assume the order of classes corresponds to your amino acid index
    weight_tensor = torch.tensor([scaled_weights[aa] for aa in sorted(scaled_weights.keys())]).to(device)

    return weight_tensor