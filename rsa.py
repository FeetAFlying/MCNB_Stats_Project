# python script performing representational similarity analysis for tactile mental imagery in primary somatosensory cortex
# fMRI data is taken from a study by Nierhaus et al. (2023)

# citation of study:    Nierhaus, T., Wesolek, S., Pach, D., Witt, C. M., Blankenburg, F., & Schmidt, T. T. (2023).
#                       Content Representation of Tactile Mental Imagery in Primary Somatosensory Cortex.
#                       Eneuro, 10(6), ENEURO.0408-22.2023. https://doi.org/10.1523/ENEURO.0408-22.2023

# SETUP
# import all functions from functions script (should be in the same directory as this script)
from rsa_functions import *
import rsatoolbox
import rsatoolbox.rdm as rsr


# VARIABLES
# script should be in directory /code/ and data in another directory /data/
datapath = "../data/"

# subjects (N = 21)
subjects = ["001"]
# "002", "003", "004", "005", "006", "007", "008", "009", "010",
# "011", "012", "013", "014", "015", "016", "017", "018", "019", "020",
# "021"

# conditions
all_conditions = ["stim_press", "stim_flutt",
                  "stim_vibro", "imag_press", "imag_flutt", "imag_vibro"]
selected_conditions = ["stim_press", "stim_flutt",
                       "stim_vibro", "imag_press", "imag_flutt", "imag_vibro"]

# ROIs
regions_of_interest = ["right_BA2"]
# "left_IPL", "SMA", "right_IFG", "left_IFG", "rPSC_1", "rPSC_2", "rPSC_3b", "rSII_TR50_right", "rSII_TR50_left"]

# DATA FORMATING
# initiate 5D array to fill with beta values of all subjects
formatted_data = np.empty(
    (79, 95, 79, len(selected_conditions), len(subjects)))
for index, subject in enumerate(subjects):
    formatted_data[:, :, :, :, index] = format_data_for_subject(
        subject, datapath, selected_conditions, all_conditions)

# formated_data is now a 5D array with the following dimensions:
#       1st dimension: 79 voxels
#       2nd dimension: 95 voxels
#       3rd dimension: 79 voxels
#       4th dimension: 6 conditions/stimulus types
#       5th dimension: 21 subjects

# CALCULATE RSA
# representational similarity analysis for each region of interest
# loop over region of interests and compute a RSA for each region separately
for region in regions_of_interest:
    # apply roi mask to data so only voxels of that roi are analyzed
    voxels_from_region = get_voxels_from_region_of_interest(
        region, formatted_data, datapath)
    conditions_key = 'conditions'
    # transform data into dataset object for using the RSAToolbox by Schütt et al., 2019
    region_datasets = create_rsa_datasets(voxels_from_region, len(subjects), conditions_key)

    # select a subset of the dataset
    # select data only from conditions 1:3 (stimulation) and 4:6 (imagery)
    stimulation_data = []
    imagery_data = []
    stimulation_conditions = [conditions_key + str(number)
                              for number in range(1, 4)]
    imagery_conditions = [conditions_key + str(number)
                          for number in range(4, 7)]

    print(stimulation_conditions)
    # loop over subjects in our data_roi structure
    for dataset in region_datasets:
        # add subject data to stimulation substructure
        stimulation_data.append(dataset.subset_obs(
            by=conditions_key, value=stimulation_conditions))
        # add subject data to imagery substructure
        imagery_data.append(dataset.subset_obs(
            by=conditions_key, value=imagery_conditions))

    # CALCULATE RDM
    # calculates a representational dissimilarity matrix for stimulation data and for imagery data
    stimulation_RDM_euclidean = rsr.calc_rdm(
        stimulation_data, method='euclidean', descriptor=conditions_key)
    imagery_RDM_euclidean = rsr.calc_rdm(
        imagery_data, method='euclidean', descriptor=conditions_key)
    all_RDM_euclidean = rsr.calc_rdm(
        region_datasets, method='euclidean', descriptor=conditions_key)  # euclidean distance

    # TODO: do the same thing but with MAHALANOBIS DISTANCE

    show_debug_for_rdm(stimulation_RDM_euclidean)
    show_debug_for_rdm(imagery_RDM_euclidean)
    show_debug_for_rdm(all_RDM_euclidean)
    input("Press Enter to continue...")

    # RSA: SIMILIARITY OF RDMs
    # compares both RDMs (stimulation and imagery) and calculates their similiarity
    cosine_similarity = rsatoolbox.rdm.compare(
        stimulation_RDM_euclidean, imagery_RDM_euclidean, method='cosine')    # cosine similiarity
    # other possible similiarity measures: (method=x)
    #               Pearson ('corr")
    #               whitened comparison methods ('corr_cov' or 'cosine_cov')
    #               Kendall's tau ('tau-a')
    #               Spearman's rho ('rho-a')

    print('The cosine similarity of stimulation and imagery RDMs in ' +
          str(region) + ' is: ' + str(np.mean(cosine_similarity)))
