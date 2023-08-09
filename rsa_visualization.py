# python script visualizing the results of the rsa performed by rsa.py

## SETUP
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# path where to the results of the analysis are (and where to save the plots as well)
resultpath = "../analysis/"

# visualization steps
steps = ["plot_rdms"]
# "plot_rdm_comparison"

# subjects (N = 10)
subjects = ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010"]

# conditions
all_conditions = ["stim_press", "stim_flutt", "stim_vibro", 
                  "imag_press", "imag_flutt", "imag_vibro"]
stimulation_conditions = ["stim_press", "stim_flutt", "stim_vibro"]
imagery_conditions = ["imag_press", "imag_flutt", "imag_vibro"]

# ROIs
# five regions of interest as defined by the original paper (intersection of stimulation vs. baseline contrast and anatomic masks)
#       rPSC_1      :   contralateral (right) primary somatosensory cortex BA 1
#       rPSC_2      :   contralateral (right) primary somatosensory cortex BA 2
#       rPSC_3b     :   contralateral (right) primary somatosensory cortex BA 3b
#       rSII_right  :   contralateral (right) secondary somatosensory cortex
#       rSII_left   :   ipsilateral (left) secondary somatosensory cortex
regions_of_interest = ["rPSC_2", "rPSC_1", "rPSC_3b", "rSII_TR50_right", "rSII_TR50_left"]

## PLOTTING
for step in steps:
    if step == "plot_rdms":
        for region in regions_of_interest:
            rdm_path = os.path.join(resultpath, region, 'rdm')
            for index, subject in enumerate(subjects):
                # read in rdm as numpy array
                stimulation_rdm = np.genfromtxt(rdm_path + "/stimulation_rdm_euclidean_" + subject + ".txt", delimiter = ',')
                imagery_rdm = np.genfromtxt(rdm_path + "/imagery_rdm_euclidean_" + subject + ".txt", delimiter = ',')
                
                # plot heatmaps
                stimulation_figure, stimulation_axes = plt.subplots()
                stimulation_axes = sns.heatmap(stimulation_rdm,
                                        xticklabels=stimulation_conditions, 
                                        yticklabels=stimulation_conditions,
                                        cbar_kws = {'label':'Dissimiliarity (Euclidean)'},
                                        cmap='viridis')
                stimulation_axes.xaxis.tick_top()
                stimulation_axes.set_title('subject ' + subject)

                # plt.show()
                # save figure as jpg file
                stimulation_figure_filename = os.path.join(rdm_path, "stimulation_rdm_euclidean_" + subject + ".jpg")
                stimulation_figure.savefig(stimulation_figure_filename)
                plt.close()

                imagery_figure, imagery_axes = plt.subplots()
                imagery_axes = sns.heatmap(imagery_rdm,
                                    xticklabels=imagery_conditions,
                                    yticklabels=imagery_conditions,
                                    cbar_kws = {'label':'Dissimiliarity (Euclidean)'},
                                    cmap='viridis')
                imagery_axes.xaxis.tick_top()
                imagery_axes.set_title('subject ' + subject)

                # plt.show()
                # save figure as jpg file
                imagery_figure_filename = os.path.join(rdm_path, "imagery_rdm_euclidean" + subject + ".jpg")
                imagery_figure.savefig(imagery_figure_filename)
                plt.close()
    
    elif step == "plot_rdm_comparison":
        # TODO: fill out
        None