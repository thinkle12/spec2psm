import re
import yaml
from pyopenms import AASequence
import logging

logger = logging.getLogger(__name__)

# TODO we need to write inference code...
# TODO we need to write entropy score calculation...
# TODO we need to write a full inference pipeline that returns back the best match (Greedy or beam search) AND an inference pipeline that returns the entropy score for the peptide found by a search algorithm...
# TODO the first is de novo mode, the second is post search rescoring mode... These would each be different classes (Methods probably) to perform bulk inference and return tabular file.
# TODO we will always probably return parquet or csv or mzTab - Give this as an option for inference only
# TODO we should also probably clean the repo up soon.. Including cleaning up the model code...
# TODO run black on everything
# TODO generate some tests...


class PeptideManager(object):

    UNIMOD_TO_SPEC2PSM = {
        '.(UniMod:737)(UniMod:1)': 'nta',  # TMT, Acetyl
        'K(UniMod:737)': 'Kt',  # TMT
        '.(UniMod:737)': 'nt',  # TMT
        'C(UniMod:4)': "Cc",  # Carbamidomethyl
        'K(UniMod:5)': 'Kcy',  # Carbamyl on K
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

    PAD_SYMBOL = "[PAD]"
    START_SYMBOL = "[CLS]"
    EOS_SYMBOL = "[SEP]"
    UNK_SYMBOL = "[UNK]"
    LEUCINE_TOKEN = 5
    ISOLEUCINE_TOKEN = 11

    TOKENS_TO_ADD = ['na', 'Cc', 'Mo', 'Kt', 'nt', 'Sp', 'Tp', 'Yp', 'Nd', 'Qd', 'ncy', 'Ku', 'Kcy']

    PEPTIDE_SYMBOLS_TO_TOKEN_MAP_SMALL = {
        'Q': 18,
        'W': 24,
        'E': 9,
        'R': 13,
        'T': 15,
        'Y': 20,
        'I': ISOLEUCINE_TOKEN,
        'P': 16,
        'A': 6,
        'S': 10,
        'D': 14,
        'Z': 28,
        'F': 19,
        'G': 7,
        'H': 22,
        'K': 12,
        'L': LEUCINE_TOKEN,
        'C': 23,
        'V': 8,
        'N': 17,
        'M': 21,
        'B': 27,
        'X': 25,
        'U': 26,
        'O': 29,
        'na': 30,
        'Cc': 31,
        'Mo': 32,
        UNK_SYMBOL: 1,
        START_SYMBOL: 2,
        EOS_SYMBOL: 3,
        PAD_SYMBOL: 0,
        '[MASK]': 4,
    }

    PEPTIDE_SYMBOLS_TO_TOKEN_MAP_MED = {
        'Q': 18,
        'W': 24,
        'E': 9,
        'R': 13,
        'T': 15,
        'Y': 20,
        'I': ISOLEUCINE_TOKEN,
        'P': 16,
        'A': 6,
        'S': 10,
        'D': 14,
        'Z': 28,
        'F': 19,
        'G': 7,
        'H': 22,
        'K': 12,
        'L': LEUCINE_TOKEN,
        'C': 23,
        'V': 8,
        'N': 17,
        'M': 21,
        'B': 27,
        'X': 25,
        'U': 26,
        'O': 29,
        'na': 30,
        'Cc': 31,
        'Mo': 32,
        'Kt': 33,
        'nt': 34,
        'Sp': 35,
        'Tp': 36,
        'Yp': 37,
        'Nd': 38,
        'Qd': 39,
        'ncy': 40,
        'Ku': 41,
        'Kcy': 42,
        UNK_SYMBOL: 1,
        START_SYMBOL: 2,
        EOS_SYMBOL: 3,
        PAD_SYMBOL: 0,
        '[MASK]': 4,
    }

    # TODO redefine the large mod map
    PEPTIDE_SYMBOLS_TO_TOKEN_MAP_LARGE = {
        'Q': 18,
        'W': 24,
        'E': 9,
        'R': 13,
        'T': 15,
        'Y': 20,
        'I': ISOLEUCINE_TOKEN,
        'P': 16,
        'A': 6,
        'S': 10,
        'D': 14,
        'Z': 28,
        'F': 19,
        'G': 7,
        'H': 22,
        'K': 12,
        'L': LEUCINE_TOKEN,
        'C': 23,
        'V': 8,
        'N': 17,
        'M': 21,
        'B': 27,
        'X': 25,
        'U': 26,
        'O': 29,
        'na': 30,
        'Cc': 31,
        'Mo': 32,
        'Kt': 33,
        'nt': 34,
        'Sp': 35,
        'Tp': 36,
        'Yp': 37,
        'Nd': 38,
        'Qd': 39,
        'ncy': 40,
        'Ku': 41,
        'Kcy': 42,
        'npg': 43,
        UNK_SYMBOL: 1,
        START_SYMBOL: 2,
        EOS_SYMBOL: 3,
        PAD_SYMBOL: 0,
        '[MASK]': 4,
    }

    # Might need to expand these
    MASSDIFF_UNIMOD_MAP = {
        "K[229]": "K(UniMod:737)",
        "n[229]": ".(UniMod:737)",
        "C[57]": "C(UniMod:4)",
        "M[16]": "M(UniMod:35)",
        "S[80]": "S(UniMod:21)",  # Check this
        "T[80]": "T(UniMod:21)",  # Check this
        "Y[80]": "Y(UniMod:21)",  # Check this
        "n[272]": ".(UniMod:737)(UniMod:5)",
        "N[1]": "N(UniMod:7)",
        "Q[1]": "Q(UniMod:7)",
        "n[212]": ".(UniMod:737)(UniMod:28)",
        "n[28]": ".(UniMod:36)",
        "K[28]": "K(UniMod:36)",
        "n[34]": ".(UniMod:510)",
        "K[34]": "K(UniMod:510)",
        "n[42]": ".(UniMod:1)",
        "n[43]": ".(UniMod:5)",
        "n[-17]": ".(UniMod:28)",
        "K[114]": "K(UniMod:121)",
        "R[10]": "R(UniMod:267)",
        "K[8]": "K(UniMod:259)",
        "n[17]": ".(UniMod:510)(UniMod:28)",
    }

    # Might need to expand these
    MASS_UNIMOD_MAP = {
        "K[357]": "K(UniMod:737)",  # TMT6Plex
        "n[230]": ".(UniMod:737)",  # TMT6Plex
        "C[160]": "C(UniMod:4)",  # Carbamidomethyl
        "M[147]": "M(UniMod:35)",  # Oxidation
        "Y[392]": "Y(UniMod:737)",  # TMT6Plex
        "S[167]": "S(UniMod:21)",  # Phosphorylation
        "T[181]": "T(UniMod:21)",  # Phosphorylation
        "Y[243]": "Y(UniMod:21)",  # Phosphorylation
        "N[116]": "N(UniMod:7)",  # Deamidation
        "Q[129]": "Q(UniMod:7)",  # Deamidation
        "n[42]": ".(UniMod:1)",  # N-terminal acetylation
    }

    def __init__(self, token_map_size="medium", config_path=None):
        self.token_map = {}
        self.mod_map = self.UNIMOD_TO_SPEC2PSM
        self.mass_unimod_map = self.MASS_UNIMOD_MAP
        self.massdiff_unimod_map = self.MASSDIFF_UNIMOD_MAP
        self.config = {}
        if config_path:
            self.config = self._load_config(config_path)
        self.token_map = self._get_token_map_by_size(size=token_map_size)
        if self.config:
            # Updating Token Maps from Configuration
            self.mod_map, self.token_map, self.mass_unimod_map, self.massdiff_unimod_map = (
                self.update_token_maps_from_config(
                    config=self.config,
                    unimod_to_spec2psm=self.mod_map,
                    token_map=self.token_map,
                    mass_map=self.mass_unimod_map,
                    massdiff_map=self.massdiff_unimod_map,
                )
            )

        self.UNIMOD_REPLACE_PATTERN = re.compile("|".join(re.escape(key) for key in self.mass_unimod_map.keys()))
        self.SPEC2PSM_REPLACE_PATTERN = re.compile("|".join(re.escape(key) for key in self.mod_map.keys()))

        self.reversed_token_map = self._reverse_token_map(self.token_map)
        self.reversed_mod_map = self._reverse_mod_map(self.mod_map)

    @classmethod
    def _extract_unimod_numbers(cls, value):
        # Find all occurrences of 'UniMod:some_number' and return the numbers as a list
        return re.findall(r'UniMod:(\d+)', value)

    def _load_config(self, config_path):
        """
        Expecting yaml config that looks like this:

        tokens:
          token1:
            unimod: "UniMod:737,UniMod:1"
            spec2psm: "nta_new"
            mass: 123.123
            massdiff: 45.45
            residue: n

          token2:
            unimod: "UniMod:5"
            spec2psm: "Kcy_new"
            mass: 150.150
            massdiff: 50.50
            residue: K

          token3:
            unimod: "UniMod:28"
            spec2psm: "npg_new"
            mass: 200.200
            massdiff: 60.60
            residue: n

        """
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def _get_token_map_by_size(self, size='small'):
        if size == 'small':
            return self.PEPTIDE_SYMBOLS_TO_TOKEN_MAP_SMALL
        elif size == 'medium':
            return self.PEPTIDE_SYMBOLS_TO_TOKEN_MAP_MED
        elif size == 'large':
            return self.PEPTIDE_SYMBOLS_TO_TOKEN_MAP_LARGE
        else:
            logger.info(f"Unknown token map size: {size}, using medium token map")
            return self.PEPTIDE_SYMBOLS_TO_TOKEN_MAP_MED

    def validate_and_update(self, new_entries, target_dict):
        """Validates and updates a target dictionary with new entries."""
        for key, value in new_entries.items():
            if key in target_dict:
                raise ValueError(f"Duplicate key detected: {key}")
            target_dict[key] = value

    def update_token_maps_from_config(self, config, unimod_to_spec2psm, token_map, mass_map, massdiff_map):

        # This gets the largest token in token map. We need to get this to create the next token
        largest_key = max(token_map, key=token_map.get)
        next_token = token_map[largest_key]

        new_mods = config.get('tokens', {})

        # Iterate over each token in the config
        for token_name, token_data in new_mods.items():

            # Initialize the next token
            next_token = next_token + 1

            # Get the attributes and transform them accordingly
            unimod = token_data.get('unimod', "")
            unimod_list = unimod.split(',') if unimod else []
            unimod_list = ["(" + x + ")" for x in unimod_list]
            unimod_string = "".join(unimod_list)
            spec2psm = token_data.get('spec2psm', "")
            residue = token_data.get('residue', "")
            mass = token_data.get('mass', "")
            massdiff = token_data.get('massdiff', "")

            # Different logic if residue is n or c term
            unimod_residue = residue
            if residue == 'n' or residue == 'c':
                unimod_residue = "."

            unimod_string = "{}{}".join(unimod_residue, unimod_string)

            # Check for uniqueness
            if unimod_string in unimod_to_spec2psm:
                raise ValueError(
                    f"Unimod {unimod} already exists in the mapping! Please remove this mod from your config file."
                )
            if spec2psm in token_map.values():
                raise ValueError(
                    f"Spec2PSM {spec2psm} already exists in the token map! Please remove this mod from your config file or select a different string."
                )

            # Add to the unimod_to_spec2psm map
            unimod_to_spec2psm[unimod] = spec2psm

            # Add to the token map
            token_map[spec2psm] = next_token

            # You can also update mass and massdiff if needed
            # If mass and massdiff are specified, ensure their validity (e.g., numeric checks)
            if residue and mass and massdiff:
                mass_rounded = int(round(float(mass['name'])))  # Round mass to integer
                mass_rounded_key = residue + "[" + str(mass_rounded) + "]"

                if mass_rounded_key in mass_map.keys():
                    raise ValueError(f"Mass {mass_rounded_key} already exists in the mapping! Mod already supported.")

                mass_map[mass_rounded_key] = unimod_string

                massdiff_rounded = int(round(float(massdiff['name'])))  # Round mass to integer
                massdiff_rounded_key = residue + "[" + str(massdiff_rounded) + "]"

                if massdiff_rounded_key in massdiff_map.keys():
                    raise ValueError(
                        f"Mass {massdiff_rounded_key} already exists in the mapping! Mod already supported."
                    )

                massdiff_map[massdiff_rounded_key] = unimod_string

        return unimod_to_spec2psm, token_map, mass_map, massdiff_map

    # Reverse the token_map to map integers back to the corresponding string
    def _reverse_token_map(self, token_map):
        return {v: k for k, v in token_map.items()}

    # Reverse the mod_map to map modification tokens back to UniMod strings
    def _reverse_mod_map(self, mod_map):
        # Create a reverse map from modification tokens to their UniMod representations
        reversed_mod_map = {}
        for mod_token, mod_name in mod_map.items():
            reversed_mod_map[mod_name] = mod_token
        return reversed_mod_map

    # Reconstruct the peptide sequence from the token list
    def reconstruct_peptide_sequence(self, tokenized_sequence):
        # Step 1: Reverse the tokenized sequence to get the string representation
        try:
            amino_acids = [self.reversed_token_map[token] for token in tokenized_sequence]

            # Step 2: Reconstruct peptide string including modifications
            reconstructed_peptide = []
            for aa in amino_acids:
                if aa in self.reversed_mod_map:
                    # If modification exists, add the modified version from the mod_map
                    reconstructed_peptide.append(self.reversed_mod_map[aa])
                else:
                    # If no modification, just add the amino acid
                    reconstructed_peptide.append(aa)

            joined = ''.join(reconstructed_peptide)
            fixed = joined.replace(self.START_SYMBOL, "")
            fixed = fixed.replace(self.EOS_SYMBOL, "")
            fixed = fixed.replace(self.PAD_SYMBOL, "")
            return fixed
        except:
            return ""

    # Calculate the theoretical peptide mass using PyOpenMS
    def calculate_peptide_mass(self, peptide_str, charge):
        # PyOpenMS AASequence.fromString will automatically handle the sequence and modifications
        try:
            aa_sequence = AASequence.fromString(peptide_str)
            return aa_sequence.getMZ(charge)
        except:
            return 0

    # Main function to process the output of your model and compute mass
    def detokenize_and_calculate_mass(self, tokenized_sequences, charges):
        peptide_masses = []

        for i in range(len(tokenized_sequences)):
            tokenized_sequence = tokenized_sequences[i]
            # Step 1: Reconstruct the peptide sequence
            peptide_str = self.reconstruct_peptide_sequence(tokenized_sequence)

            charge = charges[i]
            # Step 2: Calculate theoretical mass using PyOpenMS
            peptide_mass = self.calculate_peptide_mass(peptide_str, charge)

            peptide_masses.append(peptide_mass)

        return peptide_masses
