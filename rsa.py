# python script performing representational similarity analysis for tactile mental imagery in primary somatosensory cortex
# fMRI data is taken from a study by Nierhaus et al. (2023)

# citation of study:    Nierhaus, T., Wesolek, S., Pach, D., Witt, C. M., Blankenburg, F., & Schmidt, T. T. (2023).
#                       Content Representation of Tactile Mental Imagery in Primary Somatosensory Cortex.
#                       Eneuro, 10(6), ENEURO.0408-22.2023. https://doi.org/10.1523/ENEURO.0408-22.2023

# SETUP
from rsa_functions import *  # import all functions from functions script (should be in the same directory as this script)
import rsatoolbox
import rsatoolbox.rdm as rsr

datapath = "../data/"   # script should be in directory /code/ and data in another directory /data/

# subjects (N = 21)
subjs = ["001"]
# "002", "003", "004", "005", "006", "007", "008", "009", "010",
# "011", "012", "013", "014", "015", "016", "017", "018", "019", "020",
# "021"

# conditions
conds = ["stim_press", "stim_flutt", "stim_vibro", "imag_press", "imag_flutt", "imag_vibro"]

# ROIs
rois = ["right_BA2"]
# "left_IPL", "SMA", "right_IFG", "left_IFG", "rPSC_1", "rPSC_2", "rPSC_3b", "rSII_TR50_right", "rSII_TR50_left"]

# DATA FORMATION FOR ALL SUBJECTS
all_conds = np.empty((79, 95, 79, len(conds), len(subjs)))  # initiate 5D array to fill with beta values of all subjects
for i, sub in enumerate(subjs):
    betas_sub = remove_regressors(datapath, sub)    # remove regressors that we don't need (all except 6 conds)
    all_cond = separate_conds(conds, betas_sub)     # sort beta values into 6 conds
    all_conds[:,:,:,:,i] = separate_conds(conds, betas_sub) # add betas of subject to main 5D array

# RSA FOR ALL REGIONS OF INTEREST
# loop over region of interests and compute a RSA for each roi separately
for roi in rois:
    all_conds_roi = get_roi_voxels(roi, all_conds, datapath)    # apply roi mask to data so only voxels of that roi are analyzed
    data_roi = create_rsa_object(all_conds_roi, len(subjs))     # transform data into dataset object for using the RSAToolbox by Schütt et al., 2019

    # select a subset of the dataset
    # select data only from conditions 1:3 (stimulation) and 4:6 (imagery)
    stim_data = []
    imag_data = []
    # loop over subjects in our data_roi structure
    for sub_data in data_roi:
        stim_data.append(sub_data.subset_obs(by='conds', value=["cond_1", "cond_2", "cond_3"])) # add subject data to stimulation substructure
        imag_data.append(sub_data.subset_obs(by='conds', value=["cond_4", "cond_5", "cond_6"])) # add subject data to imagery substructure

    # STIMULATION RDM
    # calculates a representational dissimilarity matrix for stimulation data
    stim_RDM_euc = rsr.calc_rdm(stim_data, method='euclidean', descriptor='conds')  # euclidean distance
    # TO DO: MAHALANOBIS DISTANCE

    # IMAGERY RDM
    # calculates a representational dissimiliarity matrix for imagery data
    imag_RDM_euc = rsr.calc_rdm(imag_data, descriptor='conds')  # euclidean distance

    # print both RDMs to check
    print(stim_RDM_euc)
    print(imag_RDM_euc)

    # show both RDMS to check
    fig1, ax, ret_val = rsatoolbox.vis.show_rdm(stim_RDM_euc, show_colorbar='figure')
    fig2, ax, ret_val = rsatoolbox.vis.show_rdm(imag_RDM_euc, show_colorbar='figure')
    fig1.show()
    fig2.show()

    # RDM stim & imag
    RDM_euc = rsr.calc_rdm(data_roi, descriptor='conds')  # euclidean distance
    print(RDM_euc)
    fig3, ax, ret_val = rsatoolbox.vis.show_rdm(RDM_euc, show_colorbar='figure')
    fig3.show()

    # RSA: SIMILIARITY OF RDMs
    # compares both RDMs (stimulation and imagery) and calculates their similiarity
    sim_cosine = rsatoolbox.rdm.compare(stim_RDM_euc, imag_RDM_euc, method='cosine')    # cosine similiarity
    # other possible similiarity measures: (method=x)
    #               Pearson ('corr")
    #               whitened comparison methods ('corr_cov' or 'cosine_cov')
    #               Kendall's tau ('tau-a')
    #               Spearman's rho ('rho-a')

    print('The cosine similarity of stimulation and imagery RDMs in ' + str(roi) + ' is:')
    print(np.mean(sim_cosine))