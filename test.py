# imports
import os
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import rsatoolbox
# matplotlib inline

# load model rdms from mat file
datapath = '/Users/Lucy/Documents/Berlin/FU/MCNB/2Semester/Stats II/Group Project/code/rsatoolbox-main/demos/'
matlab_data = io.matlab.loadmat(os.path.join(datapath + 'rdms_inferring/modelRDMs_A2020.mat'))
matlab_data = matlab_data['modelRDMs']
n_models = len(matlab_data[0])
model_names = [matlab_data[0][i][0][0] for i in range(n_models)]
measurement_model = [matlab_data[0][i][1][0] for i in range(n_models)]
rdms_array = np.array([matlab_data[0][i][3][0] for i in range(n_models)])

# store model rdms as rsatoolbox object
model_rdms = rsatoolbox.rdm.RDMs(rdms_array,
                            rdm_descriptors={'brain_computational_model':model_names,
                                             'measurement_model':measurement_model},
                            dissimilarity_measure='Euclidean'
                           )

# show model rdms from alexnet layer conv1
conv1_rdms = model_rdms.subset('brain_computational_model','conv1')
fig, ax, ret_val = rsatoolbox.vis.show_rdm(conv1_rdms, rdm_descriptor='measurement_model', figsize=(10,10))
fig.savefig('test_name.png', bbox_inches='tight', dpi=300)
fig.show()
input()

# print information about a set of rdms
conv1_rdms = model_rdms.subset('brain_computational_model','conv1')
print(conv1_rdms)