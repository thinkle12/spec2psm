import logging
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pyopenms as ms
from pyteomics import pepxml, mzid
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# set up our logger
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

PROTON_MASS = 1.007276


class BaseProteomicsFile(object):

    def __init__(self):
        pass


class IdXML(BaseProteomicsFile):

    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    def parse_ms2s(self):
        # TODO implement idXML parsing
        pass


class Mzml(BaseProteomicsFile):
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    def parse_ms2s(self):
        # Get list of precursor mass, charge, mz, intensity, scan number, scan ID, retention time

        logger.info("Parsing mzML file: {}".format(self.filepath))
        exp = ms.MSExperiment()
        ms.MzMLFile().load(self.filepath, exp)

        # List to hold extracted precursor information
        data = []
        # Iterate over all spectra in the experiment
        for spectrum in exp:
            if spectrum.getMSLevel() == 2:  # Look only at MS2 spectra
                precursors = spectrum.getPrecursors()
                for precursor in precursors:
                    precursor_mz = precursor.getMZ()  # Precursor m/z
                    precursor_charge = precursor.getCharge()  # Precursor charge
                    mz, intensity = spectrum.get_peaks()
                    spectrum_identifier = spectrum.getNativeID()  # Scan ID
                    retention_time = spectrum.getRT()  # Retention time

                    data.append(
                        {
                            "mz": mz,
                            "intensity": intensity,
                            "precursor_mass": (precursor_mz * precursor_charge) - (precursor_charge * PROTON_MASS),
                            "precursor_mz": precursor_mz,
                            "charge": precursor_charge,
                            "scan": spectrum_identifier,
                            "scan_id": spectrum_identifier.split("scan=")[-1],
                            "retention_time": retention_time,
                        }
                    )

        logger.info("finished parsing mzML file: {}".format(self.filepath))
        logger.info("Found {} total scans".format(len(data)))

        self.data = data

    def results_to_parquet(self, output_filename=None, directory=None):

        # Handle if both or neither inputs are provided
        if output_filename and directory:
            raise ValueError("Please only either input an output filename OR a directory but not both.")

        if not output_filename and not directory:
            directory = os.getcwd()

        if not output_filename:
            # Get the filename from the spectra object file
            output_filename = os.path.splitext(os.path.basename(self.data.filepath))[0]
            output_filename = str(Path(output_filename).with_suffix(".parquet"))

        if directory:
            output_filename = os.path.join(directory, output_filename)

        rename_columns = {
            "mz": "mz",
            "intensity": "intensity",
            "precursor_mass": "precursor_mass_spectra",
            "precursor_mz": "precursor_mz_spectra",
            "charge": "charge_spectra",
            "scan_id": "scan_id",  # This one remains the same
            "retention_time": "retention_time_spectra",
        }

        # Rename columns
        self.data = self.data.rename(columns=rename_columns)

        self.data.to_parquet(output_filename, row_group_size=500, engine="pyarrow")


class Mzid(BaseProteomicsFile):

    def __init__(self, filepath, tokenizer):
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.data = None
        self.modifications = None
        self.search_params = None
        self.unimod_replace_pattern = re.compile(
            "|".join(re.escape(key) for key in self.tokenizer.peptide_manager.mass_unimod_map.keys())
        )
        self.spec2psm_replace_pattern = re.compile(
            "|".join(re.escape(key) for key in self.tokenizer.peptide_manager.mod_map.keys())
        )

    def parse_results(self):
        # Get list of precursor mass, charge, peptide, modifications, scan number, scan ID, retention time
        logger.info("Parsing mzid file: {}".format(self.filepath))
        mzid = ms.MzIdentMLFile()
        protein_ids = []
        peptide_ids = []
        mzid.load(self.filepath, protein_ids, peptide_ids)

        # use protein_ids to get fixed_modifications, digestion_enzyme, variable_modifications
        # precursor_mass_tolerance, precursor_mass_tolerance_ppm, fragment_mass_tolerance, fragment_mass_tolerance_ppm
        search_parameters = protein_ids[0].getSearchParameters()
        digestion_enzyme = search_parameters.digestion_enzyme
        enzyme = digestion_enzyme.getName()
        fixed_modifications = search_parameters.fixed_modifications
        fragment_mass_tolerance = search_parameters.fragment_mass_tolerance
        fragment_mass_tolerance_ppm = search_parameters.fragment_mass_tolerance_ppm
        missed_cleavages = search_parameters.missed_cleavages
        precursor_mass_tolerance = search_parameters.precursor_mass_tolerance
        precursor_mass_tolerance_ppm = search_parameters.precursor_mass_tolerance_ppm
        mass_type = search_parameters.mass_type
        variable_modifications = search_parameters.variable_modifications
        search_engine = protein_ids[0].getSearchEngine()
        search_engine_version = protein_ids[0].getSearchEngineVersion()

        parameters = {
            "enzyme": enzyme,
            "fixed_modifications": fixed_modifications,
            "fragment_mass_tolerance": fragment_mass_tolerance,
            "fragment_mass_tolerance_ppm": fragment_mass_tolerance_ppm,
            "missed_cleavages": missed_cleavages,
            "precursor_mass_tolerance": precursor_mass_tolerance,
            "precursor_mass_tolerance_ppm": precursor_mass_tolerance_ppm,
            "mass_type": mass_type,
            "variable_modifications": variable_modifications,
            "search_engine": search_engine,
            "search_engine_version": search_engine_version,
        }

        # List to hold extracted peptide information
        peptide_info = []
        # Iterate over all peptide identifications
        for spectrum in peptide_ids:
            # Ident here is really a scan or spectrum
            scan_id = spectrum.getMetaValue("MS:1001115")
            spectrum_reference = spectrum.getMetaValue("spectrum_reference")
            if not scan_id:
                scan_id = spectrum_reference.split("scan=")[-1]
                scan_id = scan_id.split(" ")[0]
                try:
                    float(scan_id)
                except:
                    scan_id = spectrum_reference.split("index=")[-1]
                    scan_id = scan_id.split(" ")[0]
            precursor_mz = spectrum.getMZ()
            retention_time = spectrum.getRT()

            # Loop over each psm hit
            for hit in spectrum.getHits():
                rank = hit.getRank()
                if rank > 1:
                    # Skip if not rank 1
                    continue

                peptide_sequence = hit.getSequence()
                peptide_string = peptide_sequence.toBracketString()
                peptide_string_unimod = peptide_sequence.toUniModString()  # Peptide sequence
                charge = hit.getCharge()  # Peptide charge
                hit_mz = hit.getSequence().getMonoWeight(ms.Residue.ResidueType.Full, hit.getCharge()) / hit.getCharge()
                # hit_mz = peptide_sequence.getMZ(charge) # MZ
                # hit_mass = peptide_sequence.getMonoWeight() # Full Mass
                ppm = (abs(hit_mz - precursor_mz) / hit_mz) * 10**6

                score = hit.getScore()
                q_value = hit.getMetaValue("MS:1002054")

                spec2psm_string = self.spec2psm_replace_pattern.sub(
                    lambda x: self.tokenizer.peptide_manager.mod_map[x.group()], peptide_string_unimod
                )

                peptide_info.append(
                    {
                        "precursor_mz": precursor_mz,
                        "precursor_mass": (precursor_mz * charge) - (charge * PROTON_MASS),
                        "retention_time": retention_time,
                        "charge": charge,
                        "hit_mz": hit_mz,
                        "hit_mass": (hit_mz * charge) - (charge * PROTON_MASS),
                        "ppm": ppm,
                        "q_value": q_value,
                        "score": score,
                        "peptide_string": peptide_string,
                        "peptide_string_unimod": peptide_string_unimod,
                        "peptide_string_spec2psm": spec2psm_string,
                        "target_or_decoy": hit.getMetaValue("target_decoy"),
                        "scan": spectrum_reference,
                        "scan_id": scan_id,
                    }
                )

        logger.info("finished parsing mzid file: {}".format(self.filepath))
        logger.info("Found {} total PSMs".format(len(peptide_info)))
        self.search_params = parameters
        self.data = peptide_info


class PepXML(BaseProteomicsFile):

    def __init__(self, filepath, tokenizer):
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.data = None
        self.modifications = None
        self.search_params = None
        self.unimod_replace_pattern = re.compile(
            "|".join(re.escape(key) for key in self.tokenizer.peptide_manager.mass_unimod_map.keys())
        )
        self.spec2psm_replace_pattern = re.compile(
            "|".join(re.escape(key) for key in self.tokenizer.peptide_manager.mod_map.keys())
        )

    def parse_results(self):
        # Get list of precursor mass, charge, peptide, modifications, scan number, scan ID, retention time

        logger.info("Parsing pepXML file: {}".format(self.filepath))

        params = self.parse_parameters()

        # Convert string 1 to True and string 0 to False
        fmu = True if params["parameters"]["fragment_mass_units"] == "1" else False
        pmt = True if params["parameters"]["precursor_true_units"] == "1" else False

        filtered_parameters = {
            "enzyme": params["parameters"]["search_enzyme_name"],
            "fixed_modifications": [x for x in params["modifications"] if "variable" in x and x["variable"] == "N"],
            "fragment_mass_tolerance": params["parameters"]["fragment_mass_tolerance"],
            "fragment_mass_tolerance_ppm": fmu,
            "missed_cleavages": params["parameters"]["allowed_missed_cleavage_1"],
            "precursor_mass_tolerance": params["parameters"]["precursor_true_tolerance"],
            "precursor_mass_tolerance_ppm": pmt,
            "mass_type": "NA",
            "variable_modifications": [
                x for x in params["modifications"] if "variable" not in x or x["variable"] == "Y"
            ],
            "search_engine": params["summary_attributes"]["search_engine"],
            "search_engine_version": params["summary_attributes"]["search_engine_version"],
        }

        static_mods = [x for x in params["modifications"] if "variable" in x and x["variable"] == "N"]
        static_mod_dict = {attr['name']: int(float(attr['mass'])) for attr in static_mods}

        peptide_info = []
        with pepxml.read(self.filepath) as reader:
            for psm in reader:
                # Basic information about the PSM
                spectrum = psm.get('spectrum', 'N/A')
                spectrum_reference = psm.get('spectrumNativeID', 'N/A')
                scan_id = spectrum_reference.split("scan=")[-1]
                charge = psm.get('assumed_charge', 'N/A')
                precursor_mass = psm.get('precursor_neutral_mass', 'N/A')
                precursor_mz = (precursor_mass + (charge * PROTON_MASS)) / charge

                retention_time = psm.get('retention_time_sec', 'N/A')

                hit = psm["search_hit"][0]  # Get the first search hit
                score = hit.get('search_score', {}).get('hyperscore', 0)
                unmodified_peptide = hit["peptide"]
                modified_peptide = hit["modified_peptide"]
                hit_mass = hit["calc_neutral_pep_mass"]
                hit_mz = (hit_mass + (charge * PROTON_MASS)) / charge
                ppm = (abs(hit_mz - precursor_mz) / hit_mz) * 10**6
                modifications = hit["modifications"]
                # Loop through the static modifications and insert the mass at the specified position
                # That way we can convert the AA+static mod to a new symbol for the model
                # I.e. C isnt the same as C[57]
                # The masses are added automatically for variable and term mods but not static mods
                for aa, mass in static_mod_dict.items():
                    # modified_peptide = modified_peptide.replace(aa, f"{aa}[{mass}]")
                    # Regular expression to find any 'C' not followed by [160]
                    pattern = rf"(?<={aa})(?!\[{mass}\])"
                    # Replace matches with the modified version
                    modified_peptide = re.sub(pattern, f"[{mass}]", modified_peptide)

                # Create unimod string and spec2psm string
                unimod_string = self.unimod_replace_pattern.sub(
                    lambda x: self.tokenizer.peptide_manager.mass_unimod_map[x.group()], modified_peptide
                )

                spec2psm_string = self.spec2psm_replace_pattern.sub(
                    lambda x: self.tokenizer.peptide_manager.mod_map[x.group()], unimod_string
                )

                peptide_info.append(
                    {
                        "precursor_mz": precursor_mz,
                        "precursor_mass": precursor_mass,
                        "retention_time": retention_time,
                        "charge": charge,
                        "hit_mz": hit_mz,
                        "hit_mass": hit_mass,
                        "ppm": ppm,
                        "score": score,
                        "peptide_string": modified_peptide,
                        "peptide_string_unimod": unimod_string,
                        "peptide_string_spec2psm": spec2psm_string,
                        "scan": spectrum_reference,
                        "scan_id": scan_id,
                    }
                )

        logger.info("finished parsing pepXML file: {}".format(self.filepath))
        logger.info("Found {} total PSMs".format(len(peptide_info)))
        self.search_params = filtered_parameters
        self.data = peptide_info

    def parse_parameters(self):

        # Parse the pepXML file
        tree = ET.parse(self.filepath)
        root = tree.getroot()

        search_params = {}

        # Define the namespace
        ns = {'ns': 'http://regis-web.systemsbiology.net/pepXML'}

        # Find the search_summary element with namespace
        search_summary = root.find('.//ns:search_summary', ns)

        if search_summary is not None:
            # Extracting search summary attributes
            search_params['summary_attributes'] = {
                'base_name': search_summary.get('base_name', 'N/A'),
                'precursor_mass_type': search_summary.get('precursor_mass_type', 'N/A'),
                'search_engine': search_summary.get('search_engine', 'N/A'),
                'search_engine_version': search_summary.get('search_engine_version', 'N/A'),
                'fragment_mass_type': search_summary.get('fragment_mass_type', 'N/A'),
                'search_id': search_summary.get('search_id', 'N/A'),
            }

            # Extracting parameters
            parameters = search_summary.findall('ns:parameter', ns)
            search_params['parameters'] = {param.get('name'): param.get('value') for param in parameters}

            # Extracting modifications
            modifications = search_summary.findall('ns:aminoacid_modification', ns)
            terminal_modifications = search_summary.findall('ns:terminal_modification', ns)

            search_params['modifications'] = [
                {'name': mod.get('aminoacid'), 'mass': mod.get('mass'), 'variable': mod.get('variable')}
                for mod in modifications
            ]

            # Add terminal modifications to the list if needed
            for mod in terminal_modifications:
                search_params['modifications'].append(
                    {'name': 'Terminal Modification', 'mass': mod.get('mass'), 'terminus': mod.get('terminus')}
                )

        return search_params


class PercolatorOutput(object):
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    def parse_results(self, decoy_string="rev_"):
        # Get spectra + peptide + qvalue
        logger.info("Parsing pout file: {}".format(self.filepath))

        percolator_results = pd.read_csv(self.filepath, sep="\t")

        percolator_results["scan_id"] = percolator_results["PSMId"].apply(lambda x: x.split(".")[1])

        percolator_results.rename(columns={'q-value': 'q_value'}, inplace=True)

        percolator_results['target_or_decoy'] = percolator_results['proteinIds'].apply(
            lambda x: 'decoy' if x.startswith(decoy_string) else 'target'
        )

        self.data = percolator_results
        logger.info("finished parsing pout file: {}".format(self.filepath))
        logger.info("Found {} total PSMs".format(len(percolator_results)))


class SpectraToSearchMap(object):
    def __init__(self, spectra_object, search_object, percolator_object=None):
        self.spectra_object = spectra_object
        self.search_object = search_object
        self.percolator_object = percolator_object  # optional percolator object for inputting FDR
        self.manifest = self.search_object.search_params
        self.data = None

    def map_results(self, q_value_threshold=0.01):

        logger.info("Mapping spectra object and search object")

        spectra_df = pd.DataFrame(self.spectra_object.data)
        search_df = pd.DataFrame(self.search_object.data)

        mapped_results = pd.merge(spectra_df, search_df, on="scan_id", suffixes=('_spectra', '_search'))
        mapped_results['massdiff'] = mapped_results['precursor_mass_spectra'] - mapped_results['precursor_mass_search']
        mapped_results['chargediff'] = mapped_results['charge_spectra'] - mapped_results['charge_search']

        mass_check = np.all(np.isclose(mapped_results['massdiff'], 0, atol=0.01))
        charge_check = np.all(np.isclose(mapped_results['chargediff'], 0))

        if mass_check and charge_check:
            pass
        else:
            logger.info(
                "Mass Diff values between spectra file and search file are not within tolerance. Please examine."
            )
        # Save as data and then write to tsv

        if self.percolator_object:
            mapped_results = pd.merge(
                mapped_results, self.percolator_object.data, on="scan_id", suffixes=('_search', '_percolator')
            )
            mapped_results.rename(columns={'score_percolator': 'score'}, inplace=True)
            if q_value_threshold:
                logger.info("Number of results before Q Value filtering: {}".format(len(mapped_results)))
                mapped_results = mapped_results[mapped_results['q_value'] <= q_value_threshold]
                logger.info("Filtering by Q Value")
                logger.info("Number of results after Q Value filtering: {}".format(len(mapped_results)))
        else:
            if pd.api.types.is_float_dtype(mapped_results["q_value"]):
                mapped_results = mapped_results[mapped_results['q_value'] <= q_value_threshold]

        # Remove non-essential columns
        final_df = mapped_results[
            [
                'mz',
                'intensity',
                'precursor_mass_spectra',
                'precursor_mz_spectra',
                'charge_spectra',
                'scan_id',
                'retention_time_spectra',
                'ppm',
                'q_value',
                'score',
                'peptide_string',
                'peptide_string_unimod',
                'peptide_string_spec2psm',
                'target_or_decoy',
            ]
        ]

        # Also edit search params to include filenames...
        self.manifest["search_filename"] = self.search_object.filepath
        self.manifest["spectra_filename"] = self.spectra_object.filepath

        logger.info("finished mapping spectra object and search objects")
        logger.info("Successfully mapped {} total PSMs".format(len(final_df)))
        self.data = final_df

    def results_to_parquet(self, output_filename=None, directory=None):

        # Handle if both or neither inputs are provided
        if output_filename and directory:
            raise ValueError("Please only either input an output filename OR a directory but not both.")

        if not output_filename and not directory:
            directory = os.getcwd()

        if not output_filename:
            # Get the filename from the spectra object file
            output_filename = os.path.splitext(os.path.basename(self.spectra_object.filepath))[0]
            output_filename = str(Path(output_filename).with_suffix(".parquet"))

        if directory:
            output_filename = os.path.join(directory, output_filename)

        # Add the spec2psm filename to the manifest
        self.manifest["spec2psm_filename"] = output_filename
        # Very important, we need to set the row group size appropriately
        self.data.to_parquet(output_filename, row_group_size=500, engine="pyarrow")

    def manifest_to_tsv(self, output_filename, q_value=0.01):

        logger.info("Writing search parameters and filenames into manifest file {}".format(output_filename))
        new_spectra_filename = self.search_object.search_params["spectra_filename"]

        # Check if the column is floatable
        try:
            # Attempt to convert the column to float
            self.data["q_value"] = self.data["q_value"].astype(float)

            # Filter the dataset if conversion is successful
            filtered_df = self.data[self.data["q_value"] <= q_value]

            len_filtered = len(filtered_df)

        except ValueError:
            # Column is not floatable
            logger.info(f"Column q_value is not floatable.")
            len_filtered = len(self.data)

        self.search_object.search_params["num_psms"] = len(self.data)
        self.search_object.search_params["num_psms_filtered"] = len_filtered
        new_manifest = pd.DataFrame([self.search_object.search_params])

        if os.path.exists(output_filename):
            # If the file exists, read it into a DataFrame
            existing_manifest = pd.read_csv(output_filename, sep='\t')

            # Check if we already have that spectral filename in the manifest
            if new_spectra_filename in existing_manifest['spectra_filename'].values:
                # If we do, remove that row...
                logger.info(
                    "Found: {} in manifest file: {}. Replacing the mapping.".format(
                        new_spectra_filename, output_filename
                    )
                )
                existing_manifest = existing_manifest[existing_manifest['spectra_filename'] != new_spectra_filename]

            # Append the new DataFrame to the existing DataFrame
            # And then add the new manifest to the existing
            # This way we dont have duplicate mappings of spectra/search -> parquet
            # And we will always be up to date with the newest file mappings
            combined_manifest = pd.concat([existing_manifest, new_manifest], ignore_index=True)
        else:
            # If the file doesn't exist, just use the new DataFrame
            combined_manifest = new_manifest

        # Write the combined DataFrame back to the file
        combined_manifest.to_csv(output_filename, sep='\t', index=False)


class Parquet(object):

    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.train_filepath = self.create_filepath(filepath=self.filepath, tag="train")
        self.dev_filepath = self.create_filepath(filepath=self.filepath, tag="dev")
        self.test_filepath = self.create_filepath(filepath=self.filepath, tag="test")

    def create_filepath(self, filepath, tag):
        filename = filepath.split(".parquet")[0]
        new_filepath = filename + "_" + tag + ".parquet"
        return new_filepath

    def train_test_dev_split(self, train_percent=0.9, dev_percent=0.05, test_percent=0.05):

        total_percent = train_percent + dev_percent + test_percent
        if int(total_percent) != 1:
            raise ValueError("Percentages of Train + Dev + Test must sum to 1")

        parquet_df = pd.read_parquet(self.filepath)

        # First, split the data into 90% train and 10% temporary (dev + test)
        train_df, temp_df = train_test_split(parquet_df, test_size=1 - train_percent, random_state=42)

        # Then, split the temporary set into 5% dev and 5% test
        dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        # Check the sizes of the splits
        logger.info(f"Train set: {len(train_df)}, Dev set: {len(dev_df)}, Test set: {len(test_df)}")

        train_df.to_parquet(self.train_filepath, row_group_size=500, engine="pyarrow")
        dev_df.to_parquet(self.dev_filepath, row_group_size=500, engine="pyarrow")
        test_df.to_parquet(self.test_filepath, row_group_size=500, engine="pyarrow")
