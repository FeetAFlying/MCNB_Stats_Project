import numpy as np

measurements = np.random.randn(10,10)
n_rep = 3
m_noisy = np.repeat(measurements, n_rep, axis=0)
m_noisy += np.random.randn(*m_noisy.shape)

data_from_region_3D = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
data_from_region_2D = np.reshape(data_from_region_3D, (data_from_region_3D.shape[0], -1))
models = np.array(np.array([[0, 1, 1, 0, 1, 1], [1, 0, 1, 1, 0, 1], [1, 1, 0, 1, 1, 0] ]))

model_2 = np.random.rand(6,6)
model_3 = np.eye(6,6)
model_3[:,:] = 0.5
model_3 = np.repeat(np.repeat(np.array([0.5]), 6, axis=0), 6, axis=1)
input()