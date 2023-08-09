# python script cross-validating the results of the rsa performed by rsa.py

# script trains classifier on the stimulation data and tests classifier on the imagery data
# if classifier performs good, neural representations of tactile stimulation and imagery are probably similiar

# OR: leave-one-out (subject or trial?)

# TODO: decide on cross-validation method & perform cross-validation

# SETUP
# imported formatted_data from rsa.py script with the following dimensions:
#       1st dimension: 79 voxels
#       2nd dimension: 95 voxels
#       3rd dimension: 79 voxels
#       4th dimension: 6 conditions/stimulus types
#       5th dimension: 10 participants
from rsa import formatted_data
from rsa_functions import *
import os
import numpy as np
import rsatoolbox
import rsatoolbox.data as rsd
import glob
import nibabel as nib

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
regions_of_interest = ["rPSC_2", "rPSC_1", "rPSC_3b", "rSII_TR50_right", "rSII_TR50_left"]

# CROSS-VALIDATION
# the first few paragraphs have just been copied from rsa.py as they are the same
# ----------------------------------------------------------------------------------------start
# loop over region of interests and compute a cross-validation for each region separately
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
        stimulation_sub_dataset = dataset.subset_obs(by=conditions_key, value=stimulation_conditions)
        imagery_sub_dataset = dataset.subset_obs(by=conditions_key, value=imagery_conditions)
        stimulation_data += [stimulation_sub_dataset]
        imagery_data += [imagery_sub_dataset]
    # ----------------------------------------------------------------------------------------end


'''
# main RSAToolbox function for cross-validation
train_set, test_set, ceil_set = rsatoolbox.inference.sets_leave_one_out_rdm(rdms_data)
results_cv = rsatoolbox.inference.crossval(models, rdms_data, train_set, test_set, ceil_set=ceil_set, method='corr')
'''