import re
import pyarrow.parquet as pq
import numpy as np
import torch



from pyteomics.parser import std_amino_acids

UNIMOD_TO_SPEC2PSM = {
    '.(UniMod:737)(Unimod:1)': 'nta',  # TMT, Acetyl
    'K(UniMod:737)': 'Kt',  # TMT
    '.(UniMod:737)': 'nt',  # TMT
    'C(UniMod:4)': "Cc",  # Carbamidomethyl
    'M(UniMod:35)': "Mo",  # Oxidation
    'S(UniMod:21)': "Sp",  # Phosphorylation
    'T(UniMod:21)': "Tp",  # Phosphorylation
    'Y(UniMod:21)': "Yp",  # Phosphorylation
    '.(UniMod:737)(UniMod:5)': "ntcy",  # TMT, Carbamylation
    'N(UniMod:7)': "Nd",  # Deamidation
    'Q(UniMod:7)': "Qd",  # Deamidation
    '.(UniMod:737)(UniMod:28)': "nTpg",  # TMT, Pyro Glu from Q
    '.(UniMod:36)': "ndm",  # DiMethyl
    'K(UniMod:36)': "Kdm",  # DiMethyl
    '.(UniMod:510)': "ndmc",  # DiMethyl-C13HD2
    'K(UniMod:510)': "Kdmc",  # DiMethyl-C13HD2
    '.(UniMod:1)': "na",  # Acetyl
    '.(UniMod:5)': "ncy",  # Carbamylation
    '.(UniMod:28)': "npg",  # Pyro Glu from Q
    'K(UniMod:121)': "Ku",  # Ubiquitinylation
    'R(UniMod:267)': "Rs",  # Silac
    'K(UniMod:259)': "Ks",  # Silac
    '.(UniMod:510)(UniMod:28)': "ndmcpg",  # DiMethyl-C13HD2, Pyro Glu from Q
    'C(UniMod:108)': "Cn",  # Nethylmaleimide
    'Y(UniMod:737)': "Yt",
    'S(UniMod:1)': "Sa",  # Acetyl
}

PAD_TOKEN = "<pad>"
PEPTIDE_ENCODING_MAP = {'Q': 1, 'W': 2, 'E': 3, 'R': 4, 'T': 5, 'Y': 6, 'I': 7, 'P': 8, 'A': 9, 'S': 10, 'D': 11,
                        'F': 12, 'G': 13, 'H': 14, 'K': 15, 'L': 16, 'C': 17, 'V': 18, 'N': 19, 'M': 20, 'nta': 21,
                        'Kt': 22, 'nt': 23, 'Cc': 24, 'Mo': 25, 'Sp': 26, 'Tp': 27, 'Yp': 28, 'ntcy': 29, 'Nd': 30,
                        'Qd': 31, 'nTpg': 32, 'ndm': 33, 'Kdm': 34, 'ndmc': 35, 'Kdmc': 36, 'na': 37, 'ncy': 38,
                        'npg': 39, 'Ku': 40, 'Rs': 41, 'Ks': 42, 'ndmcpg': 43, 'Cn': 44, 'Yt': 45, 'Sa': 46,
                        "<unk>": 47, "<start>": 48, "<end>": 49, PAD_TOKEN: 0}



AMINO_ACIDS = list(std_amino_acids)
MOD_LIST = list(UNIMOD_TO_SPEC2PSM.values())

def tokenize_peptide(peptide, psm_tokens, max_peptide_length=30):

    split_peptide = split_peptide_by_tokens(peptide_string=peptide, tokens=psm_tokens)

    encoded_sequence = [PEPTIDE_ENCODING_MAP['<start>']] + \
                       [PEPTIDE_ENCODING_MAP.get(aa, PEPTIDE_ENCODING_MAP['<unk>']) for aa in split_peptide] + \
                       [PEPTIDE_ENCODING_MAP['<end>']]

    return encoded_sequence

def pad_with_zeros(array, target_length=100):
    """Pad the array with zeros to the target length."""
    padded_array = np.zeros(target_length)
    padded_array[:len(array)] = array  # Only fill with existing values
    return padded_array

def get_spec2psm_tokens(amino_acids=None, mods=None):
    if mods is None:
        mods = MOD_LIST
    if amino_acids is None:
        amino_acids = AMINO_ACIDS
    return sorted(amino_acids + mods, key=len, reverse=True)


def split_peptide_by_tokens(peptide_string, tokens):
    # Sort the tokens by length in descending order
    sorted_tokens = sorted(tokens, key=len, reverse=True)
    # Create a regex pattern by joining the sorted tokens with '|'
    pattern = '|'.join(map(re.escape, sorted_tokens))
    # Use re.findall to split the input string by matching tokens
    return re.findall(pattern, peptide_string)

def calculate_means(file_paths):
    total_intensity_sum = 0
    total_mz_sum = 0
    total_precursor_mass_sum = 0

    total_intensity_count = 0
    total_mz_count = 0
    total_precursor_mass_count = 0

    for file_path in file_paths:
        # Read the entire Parquet file
        df = pq.read_table(file_path).to_pandas()

        for index, row in df.iterrows():
            # Extract intensity and mz arrays
            intensities = row['intensity']
            mz = row['mz']

            # Get the indices of the top 100 intensity values
            if len(intensities) > 100:
                # Keep the top 100 intensity and corresponding mz values
                sorted_indices = sorted(range(len(intensities)), key=lambda i: intensities[i], reverse=True)[:100]
                top_intensities = [intensities[i] for i in sorted_indices]
                top_mz = [mz[i] for i in sorted_indices]
            else:
                top_intensities = intensities
                top_mz = mz

            # Update sums and counts
            total_intensity_sum += np.sum(top_intensities)
            total_mz_sum += np.sum(top_mz)
            total_precursor_mass_sum += row['precursor_mz_spectra']  # Assuming this is a scalar

            total_intensity_count += len(top_intensities)
            total_mz_count += len(top_mz)
            total_precursor_mass_count += 1  # Each row contributes 1 to precursor mass count

    # Calculate means
    intensity_mean = total_intensity_sum / total_intensity_count if total_intensity_count > 0 else 0
    mz_mean = total_mz_sum / total_mz_count if total_mz_count > 0 else 0
    precursor_mass_mean = total_precursor_mass_sum / total_precursor_mass_count if total_precursor_mass_count > 0 else 0

    return intensity_mean, mz_mean, precursor_mass_mean

def calculate_stds(file_paths, intensity_mean, mz_mean, precursor_mass_mean):
    total_intensity_squared_diff_sum = 0
    total_mz_squared_diff_sum = 0
    total_precursor_mass_squared_diff_sum = 0

    total_intensity_count = 0
    total_mz_count = 0
    total_precursor_mass_count = 0

    for file_path in file_paths:
        # Read the entire Parquet file
        df = pq.read_table(file_path).to_pandas()

        for index, row in df.iterrows():
            # Extract intensity and mz arrays
            intensities = row['intensity']
            mz = row['mz']

            # Get the indices of the top 100 intensity values
            if len(intensities) > 100:
                # Keep the top 100 intensity and corresponding mz values
                sorted_indices = sorted(range(len(intensities)), key=lambda i: intensities[i], reverse=True)[:100]
                top_intensities = [intensities[i] for i in sorted_indices]
                top_mz = [mz[i] for i in sorted_indices]
            else:
                top_intensities = intensities
                top_mz = mz

            # Update squared differences
            total_intensity_squared_diff_sum += np.sum((top_intensities - intensity_mean) ** 2)
            total_mz_squared_diff_sum += np.sum((top_mz - mz_mean) ** 2)
            total_precursor_mass_squared_diff_sum += (row['precursor_mz_spectra'] - precursor_mass_mean) ** 2

            total_intensity_count += len(top_intensities)
            total_mz_count += len(top_mz)
            total_precursor_mass_count += 1  # Each row contributes 1 to precursor mass count

    # Calculate standard deviations
    intensity_std = np.sqrt(total_intensity_squared_diff_sum / total_intensity_count) if total_intensity_count > 0 else 0
    mz_std = np.sqrt(total_mz_squared_diff_sum / total_mz_count) if total_mz_count > 0 else 0
    precursor_mass_std = np.sqrt(total_precursor_mass_squared_diff_sum / total_precursor_mass_count) if total_precursor_mass_count > 0 else 0

    return intensity_std, mz_std, precursor_mass_std


def get_max_peptide_length(file_paths, psm_tokens):

    peptide_lengths = []
    for file_path in file_paths:
        # Read the entire Parquet file
        df = pq.read_table(file_path).to_pandas()

        for index, row in df.iterrows():
            # Extract intensity and mz arrays

            # Extract target (Y) from the row
            peptide = row['peptide_string_spec2psm']  # Assuming 'peptide' is the target column
            # Convert peptide sequence to a suitable representation, integer encoded

            # Take the length of each tensor to get the maximum length peptide + the start and end token
            peptide_tensor_len = len(torch.tensor(tokenize_peptide(peptide, psm_tokens), dtype=torch.long))
            peptide_lengths.append(peptide_tensor_len)

    return max(peptide_lengths)

# TODO write code to check for maximum peptide length...
# TODO we dont want to truncate lets say a 35 len peptide to 30 and actually use it in training. Those values should be skipped...
# TODO num missed cleavages plays a role here...
# TODO for now just look for max len but try to be able to skip during batch processing