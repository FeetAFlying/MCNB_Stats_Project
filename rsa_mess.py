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

# runs (6)
runs = ["01", "02", "03", "04", "05", "06"]
runs_count = len(runs)

# ROIs
# five regions of interest as defined by the original paper (intersection of stimulation vs. baseline contrast and anatomic masks)
#       rPSC_1      :   contralateral (right) primary somatosensory cortex BA 1
#       rPSC_2      :   contralateral (right) primary somatosensory cortex BA 2
#       rPSC_3b     :   contralateral (right) primary somatosensory cortex BA 3b
#       rSII_right  :   contralateral (right) secondary somatosensory cortex
#       rSII_left   :   ipsilateral (left) secondary somatosensory cortex
regions_of_interest = ["rPSC_2"] 
# "rPSC_1", "rPSC_3b", "rSII_TR50_right", "rSII_TR50_left"

# MODELS FOR RSA
# different theoretical models to be tested

# model_1 predicts that stimulation = imagery and stimulus types (press, flutt, vibro) are maximally separable
model_1 = np.array([[0, 1, 1, 0, 1, 1], [1, 0, 1, 1, 0, 1], [1, 1, 0, 1, 1, 0], 
                    [0, 1, 1, 0, 1, 1], [1, 0, 1, 1, 0, 1], [1, 1, 0, 1, 1, 0]])
# model_2 assumes a chance distribution of dissimiliarities (0.5)
model_2 = np.eye(6,6)
model_2[:,:] = 0.5
# model_3 assumes a random distribution of dissimilarities
model_3 = np.random.rand(6,6)
# model_4 assumes that the same stimulus types (e.g., press and press) between stimulation and imagery are less similiar
# than between stimulation and stimulation or imagery and imagery
model_4 = np.array([[0, 1, 1, 0.2, 1, 1], [1, 0, 1, 1, 0.2, 1], [1, 1, 0, 1, 1, 0.2], 
                    [0.2, 1, 1, 0, 1, 1], [1, 0.2, 1, 1, 0, 1], [1, 1, 0.2, 1, 1, 0]])
# model_5 is the same as model_4
# and stimulation or imagery conditions are more similiar (within condition) than stim-imag (between condition)
model_5 = np.array([[0, 0.8, 0.8, 0.2, 1, 1], [0.8, 0, 0.8, 1, 0.2, 1], [0.8, 0.8, 0, 1, 1, 0.2], 
                    [0.2, 1, 1, 0, 0.8, 0.8], [1, 0.2, 1, 0.8, 0, 0.8], [1, 1, 0.2, 0.8, 0.8, 0]])

# DATA FORMATING
# initiate 6D array to fill with beta values of all subjects
formatted_data = np.empty(
    (79, 95, 79, runs_count, len(selected_conditions), len(subjects)))
for index, subject in enumerate(subjects):
    formatted_data[:, :, :, :, :, index] = format_data_for_subject(
        subject, datapath, selected_conditions, all_conditions, runs_count)

# average over subjects
averaged_data_subjects = average_over_subjects(formatted_data)
# averaged_data_subjects is now a 5D array with the following dimensions:
#       1st dimension: 79 voxels
#       2nd dimension: 95 voxels
#       3rd dimension: 79 voxels
#       4th dimension: 6 runs
#       5th dimension: 6 conditions/stimulus types

# average over runs
averaged_data_runs = average_over_runs(averaged_data_subjects)
# averaged_data_runs is now a 4D array with the following dimensions:
#       1st dimension: 79 voxels
#       2nd dimension: 95 voxels
#       3rd dimension: 79 voxels
#       5th dimension: 6 conditions/stimulus types

# CALCULATE RSA
# representational similarity analysis for each region of interest
# loop over region of interests and compute a RSA for each region separately
for region in regions_of_interest:
    # apply roi mask to data so only voxels of that roi are analyzed
    voxels_from_region = get_voxels_from_region_of_interest(
        region, datapath)
    
    # index those voxels in our main data array and rearrange dimensions of array to fit dataset object
    # for data only averaged across subjects and NOT runs
    data_from_region_averaged_subjects = rearrange_array_averaged_subjects(voxels_from_region, averaged_data_subjects)

    # data_from_region_averaged_subjects is now a 3D array with the following dimensions
    #       1st dimension: 6 conditions/stimulus types
    #       2nd dimension: roi voxels
    #       3rd dimensions: 6 runs

    # for data averaged across subjects and runs
    data_from_region_averaged_runs = rearrange_array_averaged_runs(voxels_from_region, averaged_data_runs)

    # data_from_region_averaged_runs is now a 2D array with the following dimensions
    #       1st dimension: 6 conditions/stimulus types
    #       2nd dimension: roi voxels

    conditions_key = 'conditions'
    # transform data into dataset object for using the RSAToolbox by Schütt et al., 2019
    # only for data averaged across subjects
    region_dataset_averaged_subjects = create_rsa_datasets_3D(data_from_region_averaged_runs, len(subjects), conditions_key, runs_count)
    # for data averaged across subjects and runs
    region_dataset_averaged_runs = create_rsa_datasets_2D(data_from_region_averaged_runs, len(subjects), conditions_key)

    # select a subset of the dataset
    # select data only from conditions 1:3 (stimulation) and 4:6 (imagery)
    stimulation_conditions = [conditions_key + str(number)
                              for number in range(1, 4)]
    imagery_conditions = [conditions_key + str(number)
                          for number in range(4, 7)]
    stimulation_data_averaged_subjects = region_dataset_averaged_subjects.subset_obs(by=conditions_key, value=stimulation_conditions)
    stimulation_data_averaged_runs = region_dataset_averaged_runs.subset_obs(by=conditions_key, value=stimulation_conditions)
    imagery_data_averaged_subjects = region_dataset_averaged_subjects.subset_obs(by=conditions_key, value=imagery_conditions)
    imagery_data_averaged_runs = region_dataset_averaged_runs.subset_obs(by=conditions_key, value=imagery_conditions)

    # CALCULATE RDM
    # calculates a representational dissimilarity matrix for stimulation data and for imagery data
    # euclidean distance
    stimulation_RDM_euclidean = rsr.calc_rdm(
        stimulation_data_averaged_runs, method='euclidean', descriptor=conditions_key)
    imagery_RDM_euclidean = rsr.calc_rdm(
        imagery_data_averaged_runs, method='euclidean', descriptor=conditions_key)
    all_RDM_euclidean = rsr.calc_rdm(
        region_dataset_averaged_runs, method='euclidean', descriptor=conditions_key)

    # mahalanobis distance also takes noise into account
    # computing the diagonal noise covariance matrix from measurements 
    # (using the version of the data that is not averaged across runs)
    # TODO: Fix this (covariance matrix is 0.0 atm, maybe use precision matrix (inverse) but does not work because matrix = singular)
    # results in 0 mahalanobis distance - DOES NOT WORK
    stimulation_noise_covariance_matrix_diagonal = rsatoolbox.data.noise.cov_from_measurements(
        stimulation_data_averaged_subjects, obs_desc=conditions_key, dof=len(selected_conditions)/2, method='diag')
    stimulation_RDM_mahalanobis = rsatoolbox.rdm.calc_rdm(
        stimulation_data_averaged_subjects, descriptor=conditions_key, method='mahalanobis', noise=stimulation_noise_covariance_matrix_diagonal)
    imagery_noise_covariance_matrix_diagonal = rsatoolbox.data.noise.cov_from_measurements(
        imagery_data_averaged_subjects, obs_desc=conditions_key, dof=len(selected_conditions)/2, method='diag')
    imagery_RDM_mahalanobis = rsatoolbox.rdm.calc_rdm(
        imagery_data_averaged_subjects, descriptor=conditions_key, method='mahalanobis', noise=imagery_noise_covariance_matrix_diagonal)
    all_noise_covariance_matrix_diagonal = rsatoolbox.data.noise.cov_from_measurements(
        region_dataset_averaged_subjects, obs_desc=conditions_key, dof=len(selected_conditions), method='diag')
    all_RDM_mahalanobis = rsatoolbox.rdm.calc_rdm(
       region_dataset_averaged_subjects, descriptor=conditions_key, method='mahalanobis', noise=all_noise_covariance_matrix_diagonal)

    # check rdms by printing and plotting them
    show_debug_for_rdm(stimulation_RDM_euclidean)
    show_debug_for_rdm(imagery_RDM_euclidean)
    show_debug_for_rdm(all_RDM_euclidean)
    show_debug_for_rdm(stimulation_RDM_mahalanobis)
    show_debug_for_rdm(imagery_RDM_mahalanobis)
    show_debug_for_rdm(all_RDM_mahalanobis)
    input("Press Enter to continue...")

    # COMPARE RDMs
    # compares both RDMs (stimulation and imagery) and calculates their similiarity
    # cosine similiarity
    similarity_method = 'cosine'
    # other possible similiarity measures:
    #               Pearson ('corr")
    #               whitened comparison methods ('corr_cov' or 'cosine_cov')
    #               Kendall's tau ('tau-a')
    #               Spearman's rho ('rho-a')
    # compares the euclidean rdms
    similarity = rsatoolbox.rdm.compare(
        stimulation_RDM_euclidean, imagery_RDM_euclidean, method=similarity_method)
    
    print('The ' + similarity_method + ' similarity of stimulation and imagery RDMs in ' +
          str(region) + ' is: ' + str(np.mean(similarity)))
    
    # SAVE RDM RESULTS
    # save representational dissimiliarity matrices and corresponding figures
    save_rdm_results(resultpath, region, 'stimulation', 'euclidean', stimulation_RDM_euclidean)
    save_rdm_results(resultpath, region, 'imagery', 'euclidean', imagery_RDM_euclidean)
    save_rdm_results(resultpath, region, 'all', 'euclidean', all_RDM_euclidean)
    save_rdm_results(resultpath, region, 'stimulation', 'mahalanobis', stimulation_RDM_mahalanobis)
    save_rdm_results(resultpath, region, 'imagery', 'mahalanobis', imagery_RDM_mahalanobis)
    save_rdm_results(resultpath, region, 'all', 'mahalanobis', all_RDM_mahalanobis)

    # save rdm comparison results
    save_rdm_comparison_results(resultpath, region, 'stim_imag', similarity_method, similarity)

    # DEFINE MODEL RDMS
    # fixed models
    model = rsatoolbox.model.ModelFixed('test', stimulation_RDM_euclidean)
    pred = model.predict() # returns a numpy vectorized format
    pred_rdm = model.predict_rdm() # returns a RDMs object
    print(model)
    print(pred)
    print(pred_rdm)