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
datapath = "/Volumes/INTENSO/data/"
# path where to save the results of the analysis
resultpath = "../analysis/"

# subjects (N = 10)
subjects = ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010"]

# conditions
all_conditions = ["stim_press", "stim_flutt", "stim_vibro", 
                  "imag_press", "imag_flutt", "imag_vibro"]
selected_conditions = ["stim_press", "stim_flutt", "stim_vibro",
                       "imag_press", "imag_flutt", "imag_vibro"]

# ROIs
# five regions of interest as defined by the original paper (intersection of stimulation vs. baseline contrast and anatomic masks)
#       rPSC_1      :   contralateral (right) primary somatosensory cortex BA 1
#       rPSC_2      :   contralateral (right) primary somatosensory cortex BA 2
#       rPSC_3b     :   contralateral (right) primary somatosensory cortex BA 3b
#       rSII_right  :   contralateral (right) secondary somatosensory cortex
#       rSII_left   :   ipsilateral (left) secondary somatosensory cortex
regions_of_interest = ["rPSC_2"] 
# "rPSC_1", "rPSC_3b", "rSII_TR50_right", "rSII_TR50_left"


# DATA FORMATING
# initiate 5D array to fill with beta values of all subjects
formatted_data = np.empty(
    (79, 95, 79, len(selected_conditions), len(subjects)))
for index, subject in enumerate(subjects):
    formatted_data[:, :, :, :, index] = format_data_for_subject(
        subject, datapath, selected_conditions, all_conditions)
# average over subjects
averaged_data = average_over_subjects(formatted_data)

# averaged_data is now a 4D array with the following dimensions:
#       1st dimension: 79 voxels
#       2nd dimension: 95 voxels
#       3rd dimension: 79 voxels
#       4th dimension: 6 conditions/stimulus types

# CALCULATE RSA
# representational similarity analysis for each region of interest
# loop over region of interests and compute a RSA for each region separately
for region in regions_of_interest:
    # apply roi mask to data so only voxels of that roi are analyzed
    voxels_from_region = get_voxels_from_region_of_interest(
        region, datapath)
    # index those voxels in our main data array and rearrange dimensions of array to fit dataset object
    data_from_region = rearrange_array(voxels_from_region, averaged_data)

    # data_from_region is now a 2D array with the following dimensions
    #       1st dimension: 6 conditions/stimulus types
    #       2nd dimension: roi voxels

    conditions_key = 'conditions'
    # transform data into dataset object for using the RSAToolbox by Schütt et al., 2019
    region_dataset = create_rsa_datasets(data_from_region, len(subjects), conditions_key)

    # select a subset of the dataset
    # select data only from conditions 1:3 (stimulation) and 4:6 (imagery)
    stimulation_conditions = [conditions_key + str(number)
                              for number in range(1, 4)]
    imagery_conditions = [conditions_key + str(number)
                          for number in range(4, 7)]
    stimulation_data = region_dataset.subset_obs(by=conditions_key, value=stimulation_conditions)
    imagery_data = region_dataset.subset_obs(by=conditions_key, value=imagery_conditions)

    # CALCULATE RDM
    # calculates a representational dissimilarity matrix for stimulation data and for imagery data
    stimulation_RDM_euclidean = rsr.calc_rdm(
        stimulation_data, method='euclidean', descriptor=conditions_key)
    imagery_RDM_euclidean = rsr.calc_rdm(
        imagery_data, method='euclidean', descriptor=conditions_key)
    all_RDM_euclidean = rsr.calc_rdm(
        region_dataset, method='euclidean', descriptor=conditions_key)  # euclidean distance

    # TODO: do the same thing but with MAHALANOBIS DISTANCE

    show_debug_for_rdm(stimulation_RDM_euclidean)
    show_debug_for_rdm(imagery_RDM_euclidean)
    show_debug_for_rdm(all_RDM_euclidean)
    input("Press Enter to continue...")

    # RSA: SIMILIARITY OF RDMs
    # compares both RDMs (stimulation and imagery) and calculates their similiarity
    # cosine similiarity
    cosine_similarity = rsatoolbox.rdm.compare(
        stimulation_RDM_euclidean, imagery_RDM_euclidean, method='cosine')
    
    # other possible similiarity measures: (method=x)
    #               Pearson ('corr")
    #               whitened comparison methods ('corr_cov' or 'cosine_cov')
    #               Kendall's tau ('tau-a')
    #               Spearman's rho ('rho-a')

    print('The cosine similarity of stimulation and imagery RDMs in ' +
          str(region) + ' is: ' + str(np.mean(cosine_similarity)))
    
    # SAVE RESULTS
    # save representational dissimiliarity matrices and corresponding figures
    save_rdm_results(resultpath, region, 'stimulation', 'euclidean', stimulation_RDM_euclidean)
    save_rdm_results(resultpath, region, 'imagery', 'euclidean', imagery_RDM_euclidean)
    save_rdm_results(resultpath, region, 'all', 'euclidean', all_RDM_euclidean)

    # save representational similiarity analysis results
    save_rsa_results(resultpath, region, 'stim_imag', 'cosine', cosine_similarity)