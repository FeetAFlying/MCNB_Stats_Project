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
# initiate 5D array to fill with beta values of all subjects
formatted_data = np.empty(
    (79, 95, 79, len(selected_conditions), len(subjects)))
for index, subject in enumerate(subjects):
    formatted_data[:, :, :, :, index] = format_data_for_subject(
        subject, datapath, selected_conditions, all_conditions)
    

# formatted_data is now a 4D array with the following dimensions:
#       1st dimension: 79 voxels
#       2nd dimension: 95 voxels
#       3rd dimension: 79 voxels
#       4th dimension: 6 conditions/stimulus types
#       5th dimension: 10 participants

# CALCULATE RSA
# representational similarity analysis for each region of interest
# loop over region of interests and compute a RSA for each region separately
for region in regions_of_interest:
    # apply roi mask to data so only voxels of that roi are analyzed
    voxels_from_region = get_voxels_from_region_of_interest(
        region, datapath)
    # index those voxels in our main data array and rearrange dimensions of array to fit dataset object
    data_from_region = rearrange_array(voxels_from_region, formatted_data)

    # data_from_region is now a 3D array with the following dimensions
    #       1st dimension: 6 conditions/stimulus types
    #       2nd dimension: roi voxels
    #       3rd dimension: 10 participants

    conditions_key = 'conditions'
    # transform data into dataset object for using the RSAToolbox by Schütt et al., 2019
    region_datasets = create_rsa_datasets(data_from_region, len(subjects), conditions_key)

    # select a subset of the datasets
    # select data only from conditions 1:3 (stimulation) and 4:6 (imagery)
    stimulation_conditions = [conditions_key + str(number)
                              for number in range(1, 4)]
    imagery_conditions = [conditions_key + str(number)
                          for number in range(4, 7)]
    
    stimulation_data = []
    imagery_data = []
    for dataset in region_datasets:
        stimulation_data.append(dataset.subset_obs(by=conditions_key, value=stimulation_conditions))
        imagery_data.append(dataset.subset_obs(by=conditions_key, value=imagery_conditions))

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

    # DEFINE MODEL RDMS
    # add all models (2D arrays) together in a 3D array
    all_models = np.stack((model_1, model_2, model_3, model_4, model_5), axis=0)
    models_count = len(all_models[2])
    model_names = ["model_1", "model_2", "model_3", "model_4", "model_5"]
    # create a rdm dataset from those arrays
    model_rdms = rsatoolbox.rdm.RDMs(all_models,
                            rdm_descriptors={'models':model_names},
                            dissimilarity_measure='Euclidean'
                           )
    fig, ax, ret_val = rsatoolbox.vis.show_rdm(model_rdms, rdm_descriptor='models', figsize=(10,10))

    models = []
    for model in np.unique(model_names):
        rdm_model = model_rdms.subset('models', model)
        single_model = rsatoolbox.model.ModelFixed(model, rdm_model)
        models.append(single_model)

    print('created the following models:')
    for model in range(len(models)):
        print(models[model].name)

    # TODO: compare model rdms to data rdms
    # spearman or which other correlation? maybe simply not using the toolbox?
    # results don't get significant - is this because the models are simply shit? or is the data wrong?
    results_1 = rsatoolbox.inference.eval_fixed(models, all_RDM_euclidean, method='corr')
    rsatoolbox.vis.plot_model_comparison(results_1)
    print(results_1)
    input()

    # bootstrapping subjects
    results_2a = rsatoolbox.inference.eval_bootstrap_rdm(models, all_RDM_euclidean, method='corr')
    rsatoolbox.vis.plot_model_comparison(results_2a)
    print(results_2a)

    #results_1 = rsatoolbox.inference.eval_fixed(models, rdms_data, method='spearman')
    #rsatoolbox.vis.plot_model_comparison(results_1)

    #results_1 = rsatoolbox.inference.eval_fixed(models, rdms_data, method='tau-a')
    #rsatoolbox.vis.plot_model_comparison(results_1)

    #results_1 = rsatoolbox.inference.eval_fixed(models, rdms_data, method='rho-a')
    #rsatoolbox.vis.plot_model_comparison(results_1)


'''
    # fixed models
    fixed_model = rsatoolbox.model.ModelFixed('test', stimulation_RDM_euclidean)
    pred = fixed_model.predict() # returns a numpy vectorized format
    pred_rdm = fixed_model.predict_rdm() # returns a RDMs object
    print(fixed_model)
    print(pred)
    print(pred_rdm)

    # weighted models
    weighted_model = rsatoolbox.model.ModelWeighted('test', stimulation_RDM_euclidean)
'''