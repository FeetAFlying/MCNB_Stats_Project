# script with functions for the main RSA script (DO NOT RUN THIS SCRIPT)
# both scripts need to be in the same directory in order to be able to run the rsa.py script
import os
import numpy as np
import rsatoolbox
import rsatoolbox.data as rsd
import glob
import nibabel as nib
import matplotlib.pyplot as plt


## FUNCTIONS FOR FORMATTING AND AVERAGING

# our data is in the following format:
# 6 runs * 11 regressors + 6 constants = 72 beta files per participant
# regressors:       1: Stim Press
#                   2: Stim Flutt
#                   3: Stim Vibro
#                   4: Imag Press (only "successful" trials)
#                   5: Imag Flutt (only "successful" trials)
#                   6: Imag Vibro (only "successful" trials)
#                   7: Null 1
#                   8: Null 2
#                   9: pre Cue
#                   10: button press
#                   11: all the remaining (bad) Imag trials


# format_data_for_subject() takes a subject number, a datapath, the selected conditions and all conditions as input,
# removes unnecessary regressors, loads data and sorts beta values into conditions
# returns formatted data for the subject
def format_data_for_subject(subject: str,
                            datapath: str,
                            selected_conditions: list[str],
                            all_conditions: list[str],
                            runs_count: int) -> np.ndarray:
    # remove regressors that we don't need (all except 6 conditions)
    filtered_beta_files = remove_regressors(datapath, subject)
    # load only relevant data
    betas_sub = load_data(filtered_beta_files, datapath, subject)
    # sort beta values into 6 conditions
    return sort_data_into_conditions(selected_conditions, all_conditions, betas_sub, runs_count)


# remove_regressors() takes a datapath and a subject no as input 
# returns a list with the beta files that we need (only regressors 1-6, see above)
def remove_regressors(datapath: str,
                      subject: str) -> list[str]:
    folder_path = os.path.join(
        datapath, f"sub-{subject}", "1st_level_good_bad_Imag")
    # lists all beta files for the subject, sorted by name
    beta_files_subject = sorted(glob.glob('beta*.nii', root_dir=folder_path))
    # only get the relevant beta files
    # (6 files per condition, skipping 5 unused files, 6 times in total)
    filtered_beta_files = []
    for index in range(0, 56, 11):
        filtered_beta_files += beta_files_subject[index:index+6]

    if len(filtered_beta_files) != 36:
        raise ValueError("Number of beta files is not 36!")
    
    return filtered_beta_files


# load_data() takes a list with the filtered beta files, a datapath and a subject number as input
# returns a list with the loaded betas
def load_data(filtered_beta_files: list[str],
              datapath: str,
              subject: str) -> list[int]:
    folder_path = os.path.join(
    datapath, f"sub-{subject}", "1st_level_good_bad_Imag")
    betas_subject = []
    for beta_file in filtered_beta_files:
        # load nifti files
        file_path = os.path.join(folder_path, beta_file)
        beta = nib.load(file_path)
        # get data of nifti files
        beta_data = beta.get_fdata()
        # add data of file to subject list
        betas_subject.append(beta_data)
    # return list with relevant beta files for subject
    return betas_subject


# sort_data_into_conditions() takes the selected conditions, all conditions and a list of unsorted beta files as input
# returns a 5D array with the following dimensions:
#       1st dimension: 79 voxels
#       2nd dimension: 95 voxels
#       3rd dimension: 79 voxels
#       4th dimension: 6 runs/repetitions
#       5th dimension: 6 conditions/stimulus types
def sort_data_into_conditions(selected_conditions: list[str],
                        all_conditions: list[str],
                        betas_unsorted: list[int],
                        runs_count: int) -> np.ndarray:
    # initiate empty 5D array to fill
    separated_conditions = np.empty((79, 95, 79, runs_count, len(selected_conditions)))
    for index, condition in enumerate(selected_conditions):
        # get all betas for the condition
        # get index for condition (from all_conditions)
        num = all_conditions.index(condition)
        # add all six runs of stimulus together
        stimulus = np.stack(betas_unsorted[num::6], axis=-1)
        # add to main array
        separated_conditions[:, :, :, :, index] = stimulus
    return separated_conditions


# average_over_subjects() takes the already formatted data (6D array) as input
# and returns a 5D array of the data averaged over subjects
def average_over_subjects(formatted_data: np.ndarray) -> np.ndarray:
    averaged_data_subjects = np.mean(formatted_data, axis=5)
    return averaged_data_subjects


# average_over_runs() takes the already formatted and averaged data (5D array) as input
# and returns a 4D array of the data averaged over subjects and runs
def average_over_runs(averaged_data_subjects: np.ndarray) -> np.ndarray:
    averaged_data_runs = np.mean(averaged_data_subjects, axis=3)
    return averaged_data_runs


# get_voxels_from_region_of_interest() takes a region and a datapath as input
# loads a nifti file which defines that region and returns it as np.ndarray
def get_voxels_from_region_of_interest(region_of_interest: str,
                                       datapath: str) -> np.ndarray:
    folder_path = os.path.join(
        datapath, "rois", "*" + region_of_interest + "*.nii")
    all_files_path = glob.glob(folder_path)
    # check if there is only one file
    if len(all_files_path) != 1:
        raise ValueError("There is not exactly one region file!")
    file_path = all_files_path[0]
    # get data of nifti file
    region_data = nib.load(file_path).get_fdata()
    return region_data
    

# rearrange_array_averaged_subjects() takes an np.ndarray that defines a region of interest and
# a np.ndarray with the data averaged across participants (and NOT runs!) as inputs
# indexs the data with our region mask to only extract the voxels of that region
# combines the x, y, z voxel coordinates (first three dimensions) into a flat vector 
# and rearranges array dimensions
# 1 (conditions) -> 0, 0 (voxels) -> 1
# returns a 2D data array with the dimensions:
#       1st dimension: 6 conditions/stimulus types
#       2nd dimension: roi voxels
#       3rd dimension: 6 runs
def rearrange_array_averaged_subjects(region_data: np.ndarray,
                    averaged_data_subjects: np.ndarray) -> np.ndarray:
    # get all the indices of region data which are non-zero (that define the region)
    # and convert indices to flat index
    region_indices_flat = np.ravel_multi_index(np.nonzero(region_data),
                                               averaged_data_subjects.shape[:3])
    # access only the indexed voxel of our big conditions array
    voxels_of_selected_conditions = averaged_data_subjects.reshape(
        -1, *averaged_data_subjects.shape[3:])[region_indices_flat]
    # rearrange array so it fits the toolbox data structure
    # 1 (conditions) -> 0, 0 (voxels) -> 1, 2 (runs) -> 2
    selected_conditions_region_only_averaged_subjects = np.transpose(
        voxels_of_selected_conditions, (1, 0, 2))
    return selected_conditions_region_only_averaged_subjects

# rearrange_array() takes an np.ndarray that defines a region of interest and
# a np.ndarray with the data averaged across participants and runs as inputs
# indexs the data with our region mask to only extract the voxels of that region
# combines the x, y, z voxel coordinates (first three dimensions) into a flat vector 
# and rearranges array dimensions
# 1 (conditions) -> 0, 0 (voxels) -> 1
# returns a 2D data array with the dimensions:
#       1st dimension: 6 conditions/stimulus types
#       2nd dimension: roi voxels
def rearrange_array_averaged_runs(region_data: np.ndarray,
                    averaged_data_runs: np.ndarray) -> np.ndarray:
    # get all the indices of region data which are non-zero (that define the region)
    # and convert indices to flat index
    region_indices_flat = np.ravel_multi_index(np.nonzero(region_data),
                                               averaged_data_runs.shape[:3])
    # access only the indexed voxel of our big conditions array
    voxels_of_selected_conditions = averaged_data_runs.reshape(
        -1, *averaged_data_runs.shape[3:])[region_indices_flat]
    # rearrange array so it fits the toolbox data structure
    # 1 (conditions) -> 0, 0 (voxels) -> 1
    selected_conditions_region_only_averaged_runs = np.transpose(
        voxels_of_selected_conditions, (1, 0))
    return selected_conditions_region_only_averaged_runs


## FUNCTIONS FOR RSA
# create_rsa_dataset_3D() takes the data from a region, the number of runs, and a condition key as input
# returns a RSAToolbox object using the RSAToolbox by Schütt et al., 2019 with the following attributes:
#       data.measurements: 634 voxel values for 6 conditions
#       data.descriptors: subj no
#       data.obs_descriptors: cond no
#       data.channel_descriptors: vox no
def create_rsa_datasets_3D(data_from_region_3D: np.ndarray,
                        subject_count: int,
                        condition_key: str,
                        runs_count: int) -> rsd.Dataset:
    # rearrange 3D data so that the runs are not on a separate dimension but all added after one another on the second dimension
    data_from_region_2D = np.reshape(data_from_region_3D, (data_from_region_3D.shape[0], -1))
    voxel_count = data_from_region_2D.shape[1]
    condition_description = {condition_key: np.array([condition_key + str(c + 1)
                                                    for c in np.arange(data_from_region_2D.shape[0])])}
    voxel_description = {'voxels': np.array(['voxel_' + str(x + 1)
                                             for x in np.arange(voxel_count)])}
    descriptors = {'subjects': subject_count}
    rsa_data_averaged_subjects = rsd.Dataset(measurements=data_from_region_2D,
                        descriptors=descriptors,
                        obs_descriptors=condition_description,
                        channel_descriptors=voxel_description)
    return rsa_data_averaged_subjects

# create_rsa_dataset_2D() takes the data from a region, the number of subjects and a condition key as input
# returns a RSAToolbox object using the RSAToolbox by Schütt et al., 2019 with the following attributes:
#       data.measurements: 634 voxel values for 6 conditions
#       data.descriptors: subj no
#       data.obs_descriptors: cond no
#       data.channel_descriptors: vox no
def create_rsa_datasets_2D(data_from_region: np.ndarray,
                        subject_count: int,
                        condition_key: str) -> rsd.Dataset:
    voxel_count = data_from_region.shape[1]
    condition_description = {condition_key: np.array([condition_key + str(c + 1)
                                                    for c in np.arange(data_from_region.shape[0])])}
    voxel_description = {'voxels': np.array(['voxel_' + str(x + 1)
                                             for x in np.arange(voxel_count)])}
    descriptors = {'subjects': subject_count}
    rsa_data_averaged_runs = rsd.Dataset(measurements=data_from_region,
                        descriptors=descriptors,
                        obs_descriptors=condition_description,
                        channel_descriptors=voxel_description)
    return rsa_data_averaged_runs


# show_debug_for_rdm() takes the data of a representational dissimiliarity matrix as input
# prints it and plots a figure to check if everything went alright
def show_debug_for_rdm(rdm_data: rsatoolbox.rdm.RDMs):
    print(rdm_data)
    figure, axes, return_value = rsatoolbox.vis.show_rdm(
        rdm_data, show_colorbar='figure')
    figure.show()


# save_rdm_results() takes a resultpath, a region, a condition, a method
# and the data of a representational dissimiliarity matrix as input
# saves the matrix and corresponding figure to the specified directory
def save_rdm_results(resultpath: str,
                     region: str,
                     condition: str,
                     method: str,
                     rdm_data: rsatoolbox.rdm.RDMs):
    # make a new directory for the region and rdm
    rdm_path = os.path.join(
        resultpath, region, "rdm")
    if os.path.exists(rdm_path) == False:
        os.makedirs(rdm_path)

    # access rdm data as a matrix
    rdm_matrix = rdm_data.get_matrices()[0,:,:]

    # save matrix as text file
    matrix_filename = os.path.join(rdm_path, condition + "_rdm_" + method + ".txt")
    np.savetxt(matrix_filename, rdm_matrix, delimiter=',')

    # plot figure
    figure, axes, return_value = rsatoolbox.vis.show_rdm(
        rdm_data, show_colorbar='figure')
    # save figure as jpg file
    figure_filename = os.path.join(rdm_path, condition + "_rdm_" + method + ".jpg")
    figure.savefig(figure_filename)


# save_rdm_comparison_results() takes a resultpath, a region, a condition, a method
# and the data of a representational similiarity analysis as input
# saves the comparison measure to the specified directory
def save_rdm_comparison_results(resultpath: str,
                     region: str,
                     condition: str,
                     method: str,
                     comparison_data: np.ndarray):
    # make a new directory for the region and rdm_comparison
    comparison_path = os.path.join(
        resultpath, region, "rdm_comparison")
    if os.path.exists(comparison_path) == False:
        os.makedirs(comparison_path)

    # save comparison measure as text file
    filename = os.path.join(comparison_path, condition + "_rdm_comparison_" + method + ".txt")
    np.savetxt(filename, comparison_data, delimiter=',')