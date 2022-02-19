import os
import numpy as np
from scipy.io import loadmat
import glob
import argparse

parser = argparse.ArgumentParser(description='netface_setting')
parser.add_argument('--param_folder', type=str, default='/content/FACIAL/video_preprocess/train1_deep3Dface')

opt = parser.parse_args()

param_folder = opt.param_folder

mat_path_list = sorted(glob.glob(os.path.join(param_folder, '*.mat')))
len_mat = len(mat_path_list)

id_params = np.zeros((len_mat, 80), float)
tex_params = np.zeros((len_mat, 80), float)
gamma_params = np.zeros((len_mat, 27), float)
exp_params = np.zeros((len_mat, 64), float)

for i in range(1, len_mat + 1):
    item = loadmat(os.path.join(param_folder, f'{i:06}.mat'))
    id_params[i - 1, :] = item['id']
    tex_params[i - 1, :] = item['tex']
    gamma_params[i - 1, :] = item['gamma']
    exp_params[i - 1, :] = item['exp']

id_params_path = os.path.join(param_folder, 'id_train1.npz')
np.savez(id_params_path, face=id_params)

tex_params_path = os.path.join(param_folder, 'tex_train1.npz')
np.savez(tex_params_path, face=tex_params)

gamma_params_path = os.path.join(param_folder, 'gamma_train1.npz')
np.savez(gamma_params_path, face=gamma_params)

exp_params_path = os.path.join(param_folder, 'exp_train1.npz')
np.savez(exp_params_path, face=exp_params)
