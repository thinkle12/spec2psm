import torch
import pyarrow.parquet as pq
import numpy as np
from torch.utils.data import Dataset
from spec2psm.preprocess import get_spec2psm_aa_mod_symbols, pad_with_zeros, tokenize_peptide
from spectrum_utils.spectrum import MsmsSpectrum

PSM_TOKENS = get_spec2psm_aa_mod_symbols()

class Spec2PSMDataset(Dataset):

    def __init__(self, file_paths, row_group_size=500, max_peptide_length=62, spectra_length=150, normalize=True):
        self.file_paths = file_paths
        self.row_group_size = row_group_size
        self.parquet_files = [pq.ParquetFile(file_path) for file_path in file_paths]

        self.row_group_sizes = []
        self.total_rows_per_file = []
        self.cumulative_rows_per_file = []  # Cumulative rows for easier indexing across files
        self.cumulative_row_groups = []  # Cumulative row groups across files

        # Iterate over each file and calculate total rows and row group sizes
        cumulative_rows = 0
        cumulative_groups = 0
        for file_path in self.file_paths:
            parquet_file = pq.ParquetFile(file_path)

            # Get the number of row groups and number of rows in each group
            num_row_groups = parquet_file.metadata.num_row_groups
            row_groups = []
            total_rows_in_file = 0

            # Collect row group sizes for this file
            for i in range(num_row_groups):
                row_group_size = parquet_file.metadata.row_group(i).num_rows
                row_groups.append(row_group_size)
                total_rows_in_file += row_group_size

            # Store the row group sizes and total rows for this file
            self.row_group_sizes.append(row_groups)
            self.total_rows_per_file.append(total_rows_in_file)

            # Precalculate cumulative row counts for this file
            cumulative_rows += total_rows_in_file
            self.cumulative_rows_per_file.append(cumulative_rows)

            # Precalculate cumulative row group counts
            cumulative_groups += num_row_groups
            self.cumulative_row_groups.append(cumulative_groups)

        self.normalize = normalize
        self.max_peptide_length = max_peptide_length
        self.psm_tokens = get_spec2psm_aa_mod_symbols()
        self.spectra_length = spectra_length

    def __len__(self):
        # Return the total number of rows across all files
        # return sum([self.row_group_size * x - self.row_group_size for x in self.row_group_sizes])
        total_rows = 0
        for file in self.parquet_files:
            for group_idx in range(file.metadata.num_row_groups):
                row_group = file.read_row_group(group_idx)
                total_rows += row_group.num_rows  # Use actual number of rows in the group
        return total_rows
        # return 1000

    def __getitem__(self, idx):
        # Determine which file and row group to read from
        file_idx, row_group_idx, row_idx_within_group = self._get_file_and_row_group(idx)

        # Calculate row index within the determined row group
        #row_idx_within_group = idx % self.row_group_size

        # Read only the specific row group
        row_group = self.parquet_files[file_idx].read_row_group(row_group_idx)

        # Convert to Pandas DataFrame to access specific row
        df = row_group.to_pandas()

        # Retrieve the specific row
        row = df.iloc[row_idx_within_group]

        # TODO it would be nice to maybe create a MSMSSpectra Object to handle all the padding, normalization, etc
        # TODO also maybe a Peptide object too to do the same for the peptide...
        # Extract relevant data from the row
        intensities = row['intensity']
        mz = row['mz']
        precursor_mz = row['precursor_mz_spectra']
        precursor_charge = row['charge_spectra']
        precursor_mass_spectra = row['precursor_mass_spectra']

        spectrum = MsmsSpectrum(
            "",
            precursor_mz,
            precursor_charge,
            mz.astype(np.float64),
            intensities.astype(np.float32),
        )

        # TODO Future - Set the mz range and precursor peak to configurable params
        spectrum.set_mz_range(50, 2500)
        spectrum.remove_precursor_peak(2, "Da")
        spectrum.filter_intensity(max_num_peaks = self.spectra_length)
        spectrum.scale_intensity("root", 1)

        mz_transformed = spectrum.mz
        intensities_transformed = spectrum.intensity

        if len(intensities_transformed) < self.spectra_length:
            intensities_transformed = np.concatenate((intensities_transformed, np.zeros(self.spectra_length - len(intensities_transformed))))
            mz_transformed = np.concatenate((mz_transformed, np.zeros(self.spectra_length - len(mz_transformed))))

        # Convert to tensors
        intensities_transformed = torch.tensor(intensities_transformed, dtype=torch.float32)
        mz_transformed = torch.tensor(mz_transformed, dtype=torch.float32)

        precursor_mass = torch.tensor(row['precursor_mz_spectra'], dtype=torch.float32)
        charge = torch.tensor(row['charge_spectra'], dtype=torch.float32)

        # Create a mask where intensity or mz is zero
        mask = (intensities_transformed == 0) | (mz_transformed == 0)  # Use OR for masking

        # Prepare precursor_mass and charge
        precursor_mass_padded = np.where(mask, 0, precursor_mass)
        charge_padded = np.where(mask, 0, charge)

        # Pad precursor mass and charge to ensure they have length 100
        # Only using this code if we are combining the precuror info with the sequence info
        # precursor_mass_padded = pad_with_zeros(precursor_mass_padded, target_length=self.spectra_length)
        # charge_padded = pad_with_zeros(charge_padded, target_length=self.spectra_length)
        # features = torch.stack([mz_transformed, intensities_transformed,
        #                         torch.tensor(precursor_mass_padded, dtype=torch.float32), torch.tensor(charge_padded, dtype=torch.float32)], dim=-1)

        features = torch.stack([mz_transformed, intensities_transformed], dim=-1)

        # Extract target (Y) from the row
        peptide = row['peptide_string_spec2psm']  # Assuming 'peptide_string_spec2psm' is the target column

        # Convert peptide sequence to a suitable representation, integer encoded
        target = torch.tensor(tokenize_peptide(peptide, aa_mod_symbols=self.psm_tokens), dtype=torch.long)
        # Pad the tokenized sequence to max peptide length
        padded_target = torch.tensor(pad_with_zeros(target, target_length=self.max_peptide_length))

        # Convert the tensors to the correct types...
        features_32 = features.clone().detach().float()
        padded_target_long = padded_target.clone().detach().long()

        return features_32, padded_target_long, precursor_mass, charge, precursor_mass_spectra

    # def _get_file_and_row_group(self, idx):
    #     # Determine which file contains the desired row group
    #     cumulative_idx = 0
    #     for i, cumulative_groups in enumerate(self.cumulative_row_groups):
    #         if idx < cumulative_groups * self.row_group_size:
    #             row_group_idx = (idx - cumulative_idx * self.row_group_size) // self.row_group_size
    #             return i, row_group_idx
    #         cumulative_idx = cumulative_groups

    def _get_file_and_row_group(self, idx):
        cumulative_start_idx = 0  # Keep track of the cumulative starting index for each file

        for file_idx, num_rows_in_file in enumerate(self.total_rows_per_file):
            idx_within_file = idx - cumulative_start_idx  # Index within this file

            if idx_within_file < num_rows_in_file:  # Check if the index is within this file
                # Compute the row group index
                row_group_idx = idx_within_file // self.row_group_size

                # Check if we're in the last row group
                if row_group_idx == len(self.row_group_sizes[file_idx]) - 1:
                    last_row_group_size = self.row_group_sizes[file_idx][-1]
                    # Ensure idx doesn't exceed the last row group
                    if idx_within_file >= (row_group_idx * self.row_group_size + last_row_group_size):
                        raise IndexError("Index exceeds the last row group size.")

                # Calculate row index within the row group
                row_idx_within_group = idx_within_file % self.row_group_size

                return file_idx, row_group_idx, row_idx_within_group

            # Move the cumulative index to the start of the next file
            cumulative_start_idx += num_rows_in_file

        raise IndexError("Index out of range.")

    def l2_normalize(self, features):
        # Compute the L2 norm for each row
        norm = np.linalg.norm(features, ord=2, axis=0, keepdims=True)  # Calculate the L2 norm along the rows
        # Normalize the features
        normalized_features = features / norm
        return normalized_features

    def normalize_peak_intensities(self, intensities):
        """
        Applies square-root transformation and normalizes the intensities.

        Parameters:
        intensities (np.ndarray): Array of peak intensities.

        Returns:
        np.ndarray: Normalized intensities.
        """
        # Apply square-root transformation
        sqrt_intensities = np.sqrt(intensities)

        # Normalize by dividing by the sum of square-root intensities
        normalized_intensities = sqrt_intensities / np.sum(sqrt_intensities)

        return normalized_intensities