import re
import numpy as np
import pyarrow.parquet as pq

from pyteomics.parser import std_amino_acids


class Tokenizer:

    def __init__(self, peptide_manager, max_peptide_length=62, spectra_length=150):
        """
        Initializes the Tokenizer with amino acid and modification maps.
        """

        self.peptide_manager = peptide_manager
        self.mod_map = peptide_manager.mod_map
        self.token_map = peptide_manager.token_map
        self.max_peptide_length = max_peptide_length
        self.spectra_length = spectra_length
        self.AMINO_ACIDS = list(std_amino_acids)
        self.MOD_LIST = list(self.mod_map.values())
        self.aa_mod_symbols = self.get_aa_mod_symbols(amino_acids=self.AMINO_ACIDS, mods=self.MOD_LIST)
        self.eos_token = self.token_map[self.peptide_manager.EOS_SYMBOL]
        self.start_token = self.token_map[self.peptide_manager.START_SYMBOL]
        self.pad_token = self.token_map[self.peptide_manager.PAD_SYMBOL]
        self.unk_token = self.token_map[self.peptide_manager.UNK_SYMBOL]
        self.eos_symbol = self.peptide_manager.EOS_SYMBOL
        self.start_symbol = self.peptide_manager.START_SYMBOL
        self.pad_symbol = self.peptide_manager.PAD_SYMBOL
        self.unk_symbol = self.peptide_manager.UNK_SYMBOL
        self.vocab = self.get_vocab()
        self.vocab_size = self.get_vocab_size()

    def get_aa_mod_symbols(self, amino_acids=None, mods=None):
        if mods is None:
            mods = self.MOD_LIST
        if amino_acids is None:
            amino_acids = self.AMINO_ACIDS
        return sorted(amino_acids + mods, key=len, reverse=True)

    def tokenize_peptide(self, peptide):
        """
        Tokenizes a peptide sequence including its modifications.
        """
        split_peptide = self.split_peptide_by_aa_and_mods(peptide)
        return [self.start_token] + [self.token_map.get(aa, self.unk_token) for aa in split_peptide] + [self.eos_token]

    def split_peptide_by_aa_and_mods(self, peptide_string):
        """
        Splits a peptide string into its amino acids and modifications.
        """
        pattern = '|'.join(map(re.escape, self.aa_mod_symbols))
        return re.findall(pattern, peptide_string)

    def pad_sequence(self, sequence):
        """
        Pads the tokenized sequence to a fixed length.
        Assumes zero padding
        """
        padded_array = np.zeros(self.max_peptide_length, dtype=np.int64)
        sequence_length = min(len(sequence), self.max_peptide_length)
        padded_array[:sequence_length] = sequence[:sequence_length]
        return padded_array

    def get_max_peptide_length(self, file_paths, peptide_column='peptide_string_spec2psm'):
        """
        Finds the maximum peptide length in the given files.
        """
        max_length = 0
        for file_path in file_paths:
            df = pq.read_table(file_path).to_pandas()
            for peptide in df[peptide_column]:
                tokenized_length = len(self.tokenize_peptide(peptide))
                max_length = max(max_length, tokenized_length)
        return max_length

    def get_vocab(self):
        return self.token_map

    def get_vocab_size(self):
        return len(self.token_map)
