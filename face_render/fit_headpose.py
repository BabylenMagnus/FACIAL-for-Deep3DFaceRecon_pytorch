import numpy as np
from face3d import mesh
from face3d.morphable_model.fit import fit_points
from load_data import BFM
import pandas as pd
from scipy.signal import savgol_filter
import argparse
import os
import glob
from scipy.io import loadmat


parser = argparse.ArgumentParser(description='fit_headpose_setting')
parser.add_argument('--csv_path', type=str,
                    default='/content/FACIAL/video_preprocess/train1_openface/train1_512_audio.csv')
parser.add_argument('--save_path', type=str, default='/content/FACIAL/video_preprocess/train1_posenew.npz')

opt = parser.parse_args()

# --- 1. load model
face_model = BFM()
n_exp_para = face_model.exBase.shape[1]

kpt_ind = face_model.key_points
triangles = face_model.tri

csv_path = opt.csv_path
csv_info = pd.read_csv(csv_path)
num_image = len(csv_info)
base = int(csv_info.iloc[0]['frame']) - 1
save_path = opt.save_path

param_folder = '/content/FACIAL/video_preprocess/train1_deep3Dface/'
mat_path_list = sorted(glob.glob(os.path.join(param_folder, '*.mat')))
len_mat = len(mat_path_list)

item = loadmat(os.path.join(param_folder, f'{1:06}.mat'))
id_params = item['id']
tex_params = item['tex']
gamma_params = item['gamma']

exp_params = np.zeros((len_mat, 64), float)
for i in range(1, len_mat + 1):
    item = loadmat(os.path.join(param_folder, f'{i:06}.mat'))
    exp_params[i - 1, :] = item['exp']


h = 512
w = 512

headpose = np.zeros((num_image, 258), dtype=np.float32)
# --- 2. fit head pose for each frame
for frame_count in range(1, num_image + 1):
    if frame_count % 1000 == 0:
        print(frame_count)
    sub_csv_info = csv_info[csv_info['frame'] == frame_count + base]
    x = np.zeros((68, 2), dtype=np.float32)
    for i in range(68):
        x[i, 0] = sub_csv_info.iloc[0][' x_' + str(i)] - w / 2
        x[i, 1] = (h - sub_csv_info.iloc[0][' y_' + str(i)]) - h / 2 - 1
    X_ind = kpt_ind

    fitted_sp, fitted_ep, fitted_s, fitted_R, fitted_t = fit_points(
        x, X_ind, face_model, np.expand_dims(id_params, 0), n_ep=face_model.exBase.shape[1], max_iter=10
    )

    fitted_angles = mesh.transform.matrix2angle(fitted_R)
    fitted_angles = np.array([fitted_angles])

    chi_prev = np.concatenate((fitted_angles[0, :], [fitted_s], fitted_t, exp_params[frame_count - 1]), axis=0)
    params = np.concatenate((chi_prev, id_params, tex_params, gamma_params), axis=0)
    headpose[frame_count - 1, :] = params

# additional smooth
headpose = savgol_filter(headpose, 5, 3, axis=0)

np.savez(save_path, face=headpose)
