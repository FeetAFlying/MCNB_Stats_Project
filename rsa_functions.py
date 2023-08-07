import os
import numpy as np
import rsatoolbox.data as rsd   # abbreviation to deal with dataset
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

# function reads in all beta files and removes all of the regressors we don't need (only need 6 stimulus conditions) for a specific subject
def remove_regressors(datapath, sub):
        subj = f"sub-{sub}"
        filepath = os.path.join(datapath, subj + "/1st_level_good_bad_Imag/")
        beta_files_subj = sorted(glob.glob('beta*.nii', root_dir = filepath))   # lists all beta files for the subject, sorted
        del beta_files_subj[66:]        # deletes the last 6 beta values (constants) from list
        del beta_files_subj[6:11]       # deletes regressors 7-11 for 1st run from list
        del beta_files_subj[12:17]      # deletes regressors 7-11 for 2nd run from list
        del beta_files_subj[18:23]      # deletes regressors 7-11 for 3rd run from list
        del beta_files_subj[24:29]      # deletes regressors 7-11 for 4th run from list
        del beta_files_subj[30:35]      # deletes regressors 7-11 for 5th run from list
        del beta_files_subj[36:41]      # deletes regressors 7-11 for 6th run from list
        betas_sub = []
        for beta_file in beta_files_subj:
                beta = nib.load(os.path.join(filepath, beta_file))      # load nifti files
                beta_data = beta.get_fdata()    # get data of nifti files
                betas_sub += [beta_data]        # add data of file to subject list
        return betas_sub        # return list with relevant beta files for subject

# function sorts the beta files in separate conditions and returns a 4D array with the following dimensions:
#       1st dimension: 79 voxels
#       2nd dimension: 95 voxels
#       3rd dimension: 79 voxels
#       4th dimension: 6 conditions/stimulus types
# takes a list of conditions and a list of unsorted beta values as input
# returns 4D numpy array
def separate_conds(conds, betas_unsorted):
        all_cond = np.empty((79, 95, 79, len(conds)))   # initiate empty 4D array to fill
        for cond in conds:
                if cond == "stim_press":
                        stim_press_6 = np.stack(betas_unsorted[0::6], axis = -1)        # add all six runs of stimulus together
                        stim_press_mean = np.mean(stim_press_6, axis = -1)      # calculate the average over all repetitions
                        all_cond[:,:,:,0] = stim_press_mean     # add to main array
                elif cond == "stim_flutt":
                        stim_flutt_6 = np.stack(betas_unsorted[1::6], axis = -1)        # add all six runs of stimulus together
                        stim_flutt_mean = np.mean(stim_flutt_6, axis = -1)      # calculate the average over all repetitions
                        all_cond[:,:,:,1] = stim_flutt_mean     # add to main array
                elif cond == "stim_vibro":
                        stim_vibro_6 = np.stack(betas_unsorted[2::6], axis = -1)     # add all six runs of stimulus together
                        stim_vibro_mean = np.mean(stim_vibro_6, axis = -1)      # calculate the average over all repetitions
                        all_cond[:,:,:,2] = stim_vibro_mean     # add to main array
                elif cond == "imag_press":
                        imag_press_6 = np.stack(betas_unsorted[3::6], axis = -1)        # add all six runs of stimulus together
                        imag_press_mean = np.mean(imag_press_6, axis = -1)      # calculate the average over all repetitions
                        all_cond[:,:,:,3] = imag_press_mean      # add to main array
                elif cond == "imag_flutt":
                        imag_flutt_6 = np.stack(betas_unsorted[4::6], axis = -1)     # add all six runs of stimulus together
                        imag_flutt_mean = np.mean(imag_flutt_6, axis = -1)  # calculate the average over all repetitions
                        all_cond[:,:,:,4] = imag_flutt_mean      # add to main array
                elif cond == "imag_vibro":
                        imag_vibro_6 = np.stack(betas_unsorted[5::6], axis = -1)        # add all six runs of stimulus together
                        imag_vibro_mean = np.mean(imag_vibro_6, axis = -1)      # calculate the average over all repetitions
                        all_cond[:,:,:,5] = imag_vibro_mean     # add to main array
        return all_cond

# function takes a roi mask and our 5D numpy array (with participants as 5th dimension) and returns only voxels of that roi
# returns a 3D numpy array
#       1st dimension: 6 conditions/stimulus types
#       2nd dimension: roi voxels
#       3rd dimension: 21 participants
def get_roi_voxels(roi, all_cond, datapath):
        roipath = os.path.join(datapath, "rois", "*" + roi + "*.nii")
        roi_files = glob.glob(roipath)
        first_file = roi_files[0]
        roi = nib.load(first_file)  # load nifti file with roi
        roi_data = roi.get_fdata()  # get data
        roi_ind = np.nonzero(roi_data)  # get all the indices which are non-zero (that define the roi)
        roi_ind_flat = np.ravel_multi_index(roi_ind, all_cond.shape[:3])  # convert indices to flat index
        all_cond_roi = all_cond.reshape(-1, *all_cond.shape[3:])[roi_ind_flat]  # access only the indexed voxel of our big cond array
        all_cond_roi_final = np.transpose(all_cond_roi, (1, 0, 2))  # rearange array so it fits toolbox data structure
        return all_cond_roi_final

# function creates a dataset object using the RSAToolbox by Schütt et al., 2019
# the output is a RSAToolbox object with the following attributes:
#       data.measurements: 634 voxel values for 6 conditions
#       data.descriptors: subj no
#       data.obs_descriptors: cond no
#       data.channel_descriptors: vox no
def create_rsa_object(all_conds_roi_final, nSubj):
        nVox = all_conds_roi_final.shape[1]
        cond_des = {'conds': np.array(['cond_' + str(c+1) for c in np.arange(all_conds_roi_final.shape[0])])}
        vox_des = {'voxels': np.array(['voxel_' + str(x+1) for x in np.arange(nVox)])}
        des = {'subj': 1}
        data = [] # list of dataset objects
        for i in np.arange(nSubj):
                des = {'subj': i+1}
                # append the dataset object to the data list
                data.append(rsd.Dataset(measurements=all_conds_roi_final[:,:,i],
                                descriptors=des,
                                obs_descriptors=cond_des,
                                channel_descriptors=vox_des
                                )
                                )
        return data