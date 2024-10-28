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

PAD_SYMBOL = "<pad>"
START_SYMBOL = "<start>"
EOS_SYMBOL = "<end>"
UNK_SYMBOL = "<unk>"
PEPTIDE_SYMBOLS_TO_TOKEN_MAP = {'Q': 1, 'W': 2, 'E': 3, 'R': 4, 'T': 5, 'Y': 6, 'I': 7, 'P': 8, 'A': 9, 'S': 10, 'D': 11,
                        'F': 12, 'G': 13, 'H': 14, 'K': 15, 'L': 16, 'C': 17, 'V': 18, 'N': 19, 'M': 20, 'nta': 21,
                        'Kt': 22, 'nt': 23, 'Cc': 24, 'Mo': 25, 'Sp': 26, 'Tp': 27, 'Yp': 28, 'ntcy': 29, 'Nd': 30,
                        'Qd': 31, 'nTpg': 32, 'ndm': 33, 'Kdm': 34, 'ndmc': 35, 'Kdmc': 36, 'na': 37, 'ncy': 38,
                        'npg': 39, 'Ku': 40, 'Rs': 41, 'Ks': 42, 'ndmcpg': 43, 'Cn': 44, 'Yt': 45, 'Sa': 46,
                                UNK_SYMBOL: 47, START_SYMBOL: 48, EOS_SYMBOL: 49, PAD_SYMBOL: 0}

PEPTIDE_SYMBOLS_TO_TOKEN_MAP_SMALL = {'Q': 1, 'W': 2, 'E': 3, 'R': 4, 'T': 5, 'Y': 6, 'I': 7, 'P': 8, 'A': 9, 'S': 10, 'D': 11,
                        'F': 12, 'G': 13, 'H': 14, 'K': 15, 'L': 16, 'C': 17, 'V': 18, 'N': 19, 'M': 20, 'na': 21,
                        'Cc': 22, 'Mo': 23, UNK_SYMBOL: 24, START_SYMBOL: 25, EOS_SYMBOL: 26, PAD_SYMBOL: 0}

DISALLOWED_TOKENS_MAP = {'nta': 21, 'Kt': 22, 'nt': 23, 'Sp': 26, 'Tp': 27, 'Yp': 28, 'ntcy': 29, 'Nd': 30,
                        'Qd': 31, 'nTpg': 32, 'ndm': 33, 'Kdm': 34, 'ndmc': 35, 'Kdmc': 36, 'ncy': 38,
                        'npg': 39, 'Ku': 40, 'Rs': 41, 'Ks': 42, 'ndmcpg': 43, 'Cn': 44, 'Yt': 45, 'Sa': 46}

DISALLOWED_TOKENS = list(DISALLOWED_TOKENS_MAP.keys())


AMINO_ACIDS = list(std_amino_acids)
MOD_LIST = list(UNIMOD_TO_SPEC2PSM.values())

TOKEN_TO_MASS = {
    1:128.058577540,
    2:186.079312980,
    3:129.042593135,
    4:156.101111050,
    5:101.047678505,
    6:163.063328575,
    7:113.084064015,
    8:97.052763875,
    9:71.037113805,
    10:87.032028435,
    11:115.026943065,
    12:147.068413945,
    13:57.021463735,
    14:137.058911875,
    15:128.094963050,
    16:113.084064015,
    17:103.009184505,
    18:99.068413945,
    19:114.042927470,
    20:131.040484645,
    21:42.011,
    22:57.021+103.009184505,
    23:15.995+131.040484645,
}

# TODO I think I want to consider having a Peptide class
# Each Peptide  can basically store the aa_mod_symbols and the maps so I dont have to pass that data around...

def tokenize_peptide(peptide, aa_mod_symbols):

    split_peptide = split_peptide_by_aa_and_mods(peptide_string=peptide, aa_mod_symbols=aa_mod_symbols)

    encoded_sequence = ([PEPTIDE_SYMBOLS_TO_TOKEN_MAP_SMALL['<start>']] + \
                        [PEPTIDE_SYMBOLS_TO_TOKEN_MAP_SMALL.get(aa, PEPTIDE_SYMBOLS_TO_TOKEN_MAP_SMALL['<unk>']) for aa in split_peptide] + \
                        [PEPTIDE_SYMBOLS_TO_TOKEN_MAP_SMALL['<end>']])

    return encoded_sequence

def pad_with_zeros(array, target_length=150):
    """Pad the array with zeros to the target length."""
    padded_array = np.zeros(target_length)
    padded_array[:len(array)] = array  # Only fill with existing values
    return padded_array

def get_spec2psm_aa_mod_symbols(amino_acids=None, mods=None):
    if mods is None:
        mods = MOD_LIST
    if amino_acids is None:
        amino_acids = AMINO_ACIDS
    return sorted(amino_acids + mods, key=len, reverse=True)


def split_peptide_by_aa_and_mods(peptide_string, aa_mod_symbols):
    # Sort the tokens by length in descending order
    sorted_aa_mod_symbols = sorted(aa_mod_symbols, key=len, reverse=True)
    # Create a regex pattern by joining the sorted tokens with '|'
    pattern = '|'.join(map(re.escape, sorted_aa_mod_symbols))
    # Use re.findall to split the input string by matching tokens
    return re.findall(pattern, peptide_string)

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