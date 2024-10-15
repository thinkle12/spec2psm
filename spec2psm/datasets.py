# This file is used for creating a Pytorch Dataset clas to use in model / inference
# The input will be tab delimited with multiple columns
# 1. spectrum
# 2. peptide (with mods inserted) I think I should use [123] as mod insertion or (456)
# 3. peptide FDR (used to filter on)
# 4. decoy flag - whether the peptide is a target or a decoy. 1 for target, 0 for decoy
# 5. retention time
# 6. precursor mass
# 7. precursor charge
# Try to think of other features
# So basically the input is tab delimited


import torch
import pyarrow.parquet as pq
import numpy as np
from torch.utils.data import Dataset
from spec2psm.preprocess import get_spec2psm_tokens, pad_with_zeros, tokenize_peptide
from spectrum_utils.spectrum import MsmsSpectrum

PSM_TOKENS = get_spec2psm_tokens()

class Spec2PSMDataset(Dataset):

    def __init__(self, file_paths, row_group_size=100, max_peptide_length=30, normalize=True):
        self.file_paths = file_paths
        self.row_group_size = row_group_size
        self.parquet_files = [pq.ParquetFile(file_path) for file_path in file_paths]
        self.row_group_sizes = [file.metadata.num_row_groups for file in self.parquet_files]

        # Calculate cumulative number of row groups to determine file and group
        self.cumulative_row_groups = []
        total_groups = 0
        for num_groups in self.row_group_sizes:
            total_groups += num_groups
            self.cumulative_row_groups.append(total_groups)

        # Store normalization parameters (mean and std)
        # TODO if these are all None we need to calculate them...
        self.normalize = normalize
        self.max_peptide_length = max_peptide_length
        self.psm_tokens = get_spec2psm_tokens()

    def __len__(self):
        # Return the total number of rows across all files
        return sum([self.row_group_size * x - self.row_group_size for x in self.row_group_sizes])
        # return 1000

    def __getitem__(self, idx):
        # Determine which file and row group to read from
        file_idx, row_group_idx = self._get_file_and_row_group(idx)

        # Calculate row index within the determined row group
        row_idx_within_group = idx % self.row_group_size

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

        spectrum.set_mz_range(50, 2500)

        spectrum.remove_precursor_peak(2, "Da")

        spectrum.filter_intensity(max_num_peaks = 150)

        spectrum.scale_intensity("root", 1)

        mz_transformed = spectrum.mz
        intensities_transformed = spectrum.intensity


        # Pad mz and intensity to length 100
        if len(intensities_transformed) < 150:
            intensities_transformed = np.concatenate((intensities_transformed, np.zeros(150 - len(intensities_transformed))))
            mz_transformed = np.concatenate((mz_transformed, np.zeros(150 - len(mz_transformed))))

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
        precursor_mass_padded = pad_with_zeros(precursor_mass_padded)
        charge_padded = pad_with_zeros(charge_padded)

        # Stack the features to form a tensor of shape (100, 4)
        # features = torch.stack([mz, intensities, precursor_mass.repeat(100), charge.repeat(100)], dim=-1)
        try:
            # features = torch.stack([mz_transformed, intensities_transformed,
            #                         torch.tensor(precursor_mass_padded, dtype=torch.float32), torch.tensor(charge_padded, dtype=torch.float32)], dim=-1)
            features = torch.stack([mz_transformed, intensities_transformed], dim=-1)

        except:
            print("problem")

        # Extract target (Y) from the row
        peptide = row['peptide_string_spec2psm']  # Assuming 'peptide' is the target column
        # Convert peptide sequence to a suitable representation, integer encoded

        target = torch.tensor(tokenize_peptide(peptide, psm_tokens=self.psm_tokens), dtype=torch.long)

        padded_target = torch.tensor(pad_with_zeros(target, target_length=self.max_peptide_length))

        # Convert the tensors to the correct types...
        features_32 = features.clone().detach().float()
        padded_target_long = padded_target.clone().detach().long()

        # # Get random permutation of indices
        # shuffled_indices = torch.randperm(features_32.size(1))
        #
        # # Scramble the tensor
        # features_32 = features_32[shuffled_indices]

        return features_32, padded_target_long, precursor_mass, charge, precursor_mass_spectra

    def _get_file_and_row_group(self, idx):
        # Determine which file contains the desired row group
        cumulative_idx = 0
        for i, cumulative_groups in enumerate(self.cumulative_row_groups):
            if idx < cumulative_groups * self.row_group_size:
                row_group_idx = (idx - cumulative_idx * self.row_group_size) // self.row_group_size
                return i, row_group_idx
            cumulative_idx = cumulative_groups

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