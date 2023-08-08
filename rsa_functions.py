import os
import numpy as np
import rsatoolbox
import rsatoolbox.data as rsd
import glob
import nibabel as nib

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


def format_data_for_subject(subject: str, datapath: str,
                            selected_conditions: list[str],
                            all_conditions: list[str]) -> np.ndarray:
    # remove regressors that we don't need (all except 6 conds)
    betas_sub = load_data(datapath, subject)
    # sort beta values into 6 conds
    return separate_conditions(selected_conditions, all_conditions, betas_sub)


def load_data(datapath: str, subject: str) -> list[int]:
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

    betas_subject = []
    for beta_file in beta_files_subject:
        # load nifti files
        file_path = os.path.join(folder_path, beta_file)
        beta = nib.load(file_path)
        # get data of nifti files
        beta_data = beta.get_fdata()
        # add data of file to subject list
        betas_subject.append(beta_data)
    # return list with relevant beta files for subject
    return betas_subject

# function sorts the beta files in separate conditions and returns a 4D array with the following dimensions:
#       1st dimension: 79 voxels
#       2nd dimension: 95 voxels
#       3rd dimension: 79 voxels
#       4th dimension: 6 conditions/stimulus types
# takes a list of conditions and a list of unsorted beta values as input
# returns 4D numpy array
# TODO: rename this function


def separate_conditions(selected_conditions: list[str],
                        all_conditions: list[str],
                        betas_unsorted: list[int]) -> np.ndarray:
    # initiate empty 4D array to fill
    separated_conditions = np.empty((79, 95, 79, len(selected_conditions)))
    for index, condition in enumerate(selected_conditions):
        # get all betas for the condition
        # get index for condition (from all_conditions)
        num = all_conditions.index(condition)
        # add all six runs of stimulus together
        stimulus = np.stack(betas_unsorted[num::6], axis=-1)
        # calculate the average over all repetitions
        stimulus_mean = np.mean(stimulus, axis=-1)
        # add to main array
        separated_conditions[:, :, :, index] = stimulus_mean
    return separated_conditions

# function takes a roi mask and our
# 5D numpy array (with participants as 5th dimension) and returns only voxels of that roi
# 5D numpy input array:
#       1st dimension: 79 voxels
#       2nd dimension: 95 voxels
#       3rd dimension: 79 voxels
#       4th dimension: 6 conditions/stimulus types
#       5th dimension: 21 subjects
# returns a 3D numpy array
#       1st dimension: 6 conditions/stimulus types
#       2nd dimension: roi voxels
#       3rd dimension: 21 participants
# we combine the first three dimensions to one dimension, and then move some dimensions
# 1 (conditions) -> 0, 0 (voxels) -> 1, 2 (participants) -> 2

# TODO: split function into three
# one function that gets the roi voxels
# one function that rearranges the array
# one function that averages over participants (or do this as part of data formatting earlier?)

def get_voxels_from_region_of_interest(region_of_interest: str,
                                       selected_conditions_data: np.ndarray,
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
    # get all the indices which are non-zero (that define the region)
    # and convert indices to flat index
    region_indices_flat = np.ravel_multi_index(np.nonzero(region_data),
                                               selected_conditions_data.shape[:3])
    # access only the indexed voxel of our big conditions array
    voxels_of_selected_conditions = selected_conditions_data.reshape(
        -1, *selected_conditions_data.shape[3:])[region_indices_flat]
    # rearrange array so it fits the toolbox data structure
    # 1 (conditions) -> 0, 0 (voxels) -> 1, 2 (participants) -> 2
    selected_conditions_region_only = np.transpose(
        voxels_of_selected_conditions, (1, 0, 2))
    return selected_conditions_region_only

# function creates a dataset object using the RSAToolbox by Schütt et al., 2019
# the output is a RSAToolbox object with the following attributes:
#       data.measurements: 634 voxel values for 6 conditions
#       data.descriptors: subj no
#       data.obs_descriptors: cond no
#       data.channel_descriptors: vox no


def create_rsa_datasets(voxels_of_region: np.ndarray, subject_count: int, condition_key: str) -> list[rsd.Dataset]:
    voxel_count = voxels_of_region.shape[1]
    condition_description = {condition_key: np.array([condition_key + str(c + 1)
                                                    for c in np.arange(voxels_of_region.shape[0])])}
    voxel_description = {'voxels': np.array(['voxel_' + str(x + 1)
                                             for x in np.arange(voxel_count)])}
    rsa_data = []  # list of dataset objects
    for index in np.arange(subject_count):
        descriptors = {'subject': index + 1}
        # append the dataset object to the data list
        rsa_data.append(rsd.Dataset(measurements=voxels_of_region[:, :, index],
                                    descriptors=descriptors,
                                    obs_descriptors=condition_description,
                                    channel_descriptors=voxel_description
                                    )
                        )
    return rsa_data


def show_debug_for_rdm(rdm_data: rsatoolbox.rdm.RDMs):
    print(rdm_data)
    figure, axes, return_value = rsatoolbox.vis.show_rdm(
        rdm_data, show_colorbar='figure')
    figure.show()
