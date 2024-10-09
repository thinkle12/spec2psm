# TODO
# First pass will be mzML + pepXML + Pout
# Or mzML + mzid if data exists

# Carbamidomethyl + Methionine Oxidation
# Just Methionine Oxidation (With no Carba) PDX010000
# Then add Carbamidomethyl + Methionine Oxidation + Deamidation of asparagine and glutamine
# Could maybe look for N-terminal lysine acetylation (Or n term acetyl in general)
# Then add Carbamidomethyl + Methionine Oxidation + Phosphorylation
# The look for TMT Global Proteomics!!!!

# TODO
# Spec2PSM
![logo](static/spec2PSM.png)

# High Priority
Download the carb + metox data and analyze it in Seqera
Track precursor + fragment tolerances, organism, modifications, path to data directory, path to manifest

# TODO

Get roughly 100k high quality spectra and then start training a model :)
Create a DataLoader module that takes as input a list of directories
These directories contain the parquet files
Then we need to transform the parquet files to tensors basically
The time steps will be the number of m/z and intensities
We also will include precursor mz, charge, retention time, ppm maybe
Need to figure out the shape of the input
Need to transform PEPT[123]IDE to PEPTtIDE
Might end up using PEPT(UniMod:4)IDE then mapping T(UniMod:4) -> "Tt" as its own token
Then need to group to P,E,P,Tt,I,D,E
Then need to transform that to tokens (This is for the decoder)
Need to figure out model architecture... start simple

Integrate spec2psm reader to create parquet file mapping peptide spectrum -> peptide sequence
Not sure what I meant here, maybe this means create a CLI script?



# Potential datasets - Go through these...
# https://www.ebi.ac.uk/pride/archive/projects/PXD054875 # Raw + Pepxml but theres a bunch of mods here # Orbitrap Eclipse
# https://www.ebi.ac.uk/pride/archive/projects/PXD054611 # mzml + mzid, no mods DDA but Q-exactive
# https://www.ebi.ac.uk/pride/archive/projects/PXD054199 # mzml + mzid, phos/acetyl Q-exactive
# https://www.ebi.ac.uk/pride/archive/projects/PXD019446 - Some potential here but its a tripleTOF instrument...
# https://www.ebi.ac.uk/pride/archive/projects/PXD007913 - Potential here but its complex
# https://www.ebi.ac.uk/pride/archive/projects/PXD013503 - Small but potential
# https://www.ebi.ac.uk/pride/archive/projects/PXD013445 - Again, small but potential
# https://www.ebi.ac.uk/pride/archive/projects/PXD006254 - Same
# https://www.ebi.ac.uk/pride/archive/projects/PXD024584 - Lots of data for mzid and mzml but potentially confusing because no metadata
# https://www.ebi.ac.uk/pride/archive/projects/PXD007913 - Potential
# https://www.ebi.ac.uk/pride/archive/projects/PXD035689 - Potential
# https://www.ebi.ac.uk/pride/archive/projects/PXD028152 - This might be good but theres only one mzml and mzid so not sure if this is all data combined???
# https://www.ebi.ac.uk/pride/archive/projects/PXD042148 - Maybe???
# https://www.ebi.ac.uk/pride/archive/projects/PXD033413 - mzxml and mzid ???
# https://www.ebi.ac.uk/pride/archive/projects/PXD041415 - Perhaps the best dataset thus far mzid and mzml !!!
# https://www.ebi.ac.uk/pride/archive/projects/PXD022287 - Interesting - Small but potentially good? !!!
# https://www.ebi.ac.uk/pride/archive/projects/PXD010000 - Very comprehensive set of data with many many organisms and mzml + mzid (Lacks human, mouse, etc though) !!!
# https://www.ebi.ac.uk/pride/archive/projects/PXD002967 - Comprehensive set of tissue proteomics for human... could be decent? Has mzid and mzML - Old data tho...
# https://www.ebi.ac.uk/pride/archive/projects/PXD051803 - mzid but has mzxml...
# https://www.ebi.ac.uk/pride/archive/projects/PXD027259 - One large run for mzid and mzml it seems???
# https://www.ebi.ac.uk/pride/archive/projects/PXD036996 - This looks pretty good tbh mzid and mzml
# https://www.ebi.ac.uk/pride/archive/projects/PXD037247 - 4 files, not sure on the file mapping???
# https://www.ebi.ac.uk/pride/archive/projects/PXD055735 - Good 8 mzml and mzid
# https://www.ebi.ac.uk/pride/archive/projects/PXD055391 - 3 more
# https://www.ebi.ac.uk/pride/archive/projects/PXD039817 - Q exactive... 9 good TMT files...
# https://www.ebi.ac.uk/pride/archive/projects/PXD023042 - potential tmt dataset
# https://www.ebi.ac.uk/pride/archive/projects/PXD055391 - 3 files TMT Use this
# https://www.ebi.ac.uk/pride/archive/projects/PXD044454 - 3 files TMT study
# https://www.ebi.ac.uk/pride/archive/projects/PXD041587 - Human TMT 90 or so files each. This mighte be good???
# https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?task=b7789710a31f488c9a74eb3e3b6f61eb - There is MGF data here from deepnovo but everything is in the mgf which is weird
# https://www.ebi.ac.uk/pride/archive/projects/PXD002179 - MGF + mzid
# https://www.ebi.ac.uk/pride/archive/projects/PXD055391 TMT dataset
# https://www.ebi.ac.uk/pride/archive/projects/PXD029894 one TMT plex
# https://www.ebi.ac.uk/pride/archive/projects/PXD009960 one TMT plex again
# https://www.ebi.ac.uk/pride/archive/projects/PXD000710 Lots of data but old, not TMT
# https://www.ebi.ac.uk/pride/archive/projects/PXD001428 Lots of data but not a 1-1 mapping
# https://www.ebi.ac.uk/pride/archive/projects/PXD051777 TMT but small
# https://www.ebi.ac.uk/pride/archive/projects/PXD051747 One file but large, not TMT
# https://www.ebi.ac.uk/pride/archive/projects/PXD051724 - 4 TMT files They dont download... fml
# https://www.ebi.ac.uk/pride/archive/projects/PXD033105 - TMT potential - THESE ACTUALLY DOWNLOAD

# Looks like search of mzml mzid is working..