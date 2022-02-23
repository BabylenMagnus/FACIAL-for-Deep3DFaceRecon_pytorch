import os
import numpy as np
from skimage import io
from face3d import mesh
from load_data import BFM
import argparse

# input:
#   video_preprocess/train1_posenew.npz
# output:
#   video_preprocess/train_A

parser = argparse.ArgumentParser(description='Render_setting')
parser.add_argument('--real_params_path', type=str, default='../video_preprocess/train1_posenew.npz')
parser.add_argument('--outpath', type=str, default='../video_preprocess/train_A/')

opt = parser.parse_args()


def compute_norm(face_shape, face_model):
    face_id = face_model.tri  # vertex index for each triangle face, with shape [F,3], F is number of faces
    point_id = face_model.point_buf  # adjacent face index for each vertex, with shape [N,8], N is number of vertex
    shape = face_shape
    face_id = (face_id - 1).astype(np.int32)
    point_id = (point_id - 1).astype(np.int32)
    v1 = shape[:, face_id[:, 0], :]
    v2 = shape[:, face_id[:, 1], :]
    v3 = shape[:, face_id[:, 2], :]
    e1 = v1 - v2
    e2 = v2 - v3
    face_norm = np.cross(e1, e2)  # compute normal for each face
    face_norm = np.concatenate([face_norm, np.zeros([1, 1, 3])],
                               axis=1)  # concat face_normal with a zero vector at the end
    v_norm = np.sum(face_norm[:, point_id, :], axis=2)  # compute vertex normal using one-ring neighborhood
    v_norm = v_norm / np.expand_dims(np.linalg.norm(v_norm, axis=2), 2)  # normalize normal vectors

    return v_norm


def illumination_layer(face_texture, norm, gamma):
    num_vertex = np.shape(face_texture)[1]

    init_lit = np.array([0.8, 0, 0, 0, 0, 0, 0, 0, 0])
    gamma = np.reshape(gamma, [-1, 3, 9])
    gamma = gamma + np.reshape(init_lit, [1, 1, 9])

    # parameter of 9 SH function
    a0 = np.pi
    a1 = 2 * np.pi / np.sqrt(3.0)
    a2 = 2 * np.pi / np.sqrt(8.0)
    c0 = 1 / np.sqrt(4 * np.pi)
    c1 = np.sqrt(3.0) / np.sqrt(4 * np.pi)
    c2 = 3 * np.sqrt(5.0) / np.sqrt(12 * np.pi)

    y0 = np.tile(np.reshape(a0 * c0, [1, 1, 1]), [1, num_vertex, 1])
    y1 = np.reshape(-a1 * c1 * norm[:, :, 1], [1, num_vertex, 1])
    y2 = np.reshape(a1 * c1 * norm[:, :, 2], [1, num_vertex, 1])
    y3 = np.reshape(-a1 * c1 * norm[:, :, 0], [1, num_vertex, 1])
    y4 = np.reshape(a2 * c2 * norm[:, :, 0] * norm[:, :, 1], [1, num_vertex, 1])
    y5 = np.reshape(-a2 * c2 * norm[:, :, 1] * norm[:, :, 2], [1, num_vertex, 1])
    y6 = np.reshape(a2 * c2 * 0.5 / np.sqrt(3.0) * (3 * np.square(norm[:, :, 2]) - 1), [1, num_vertex, 1])
    y7 = np.reshape(-a2 * c2 * norm[:, :, 0] * norm[:, :, 2], [1, num_vertex, 1])
    y8 = np.reshape(a2 * c2 * 0.5 * (np.square(norm[:, :, 0]) - np.square(norm[:, :, 1])), [1, num_vertex, 1])

    y = np.concatenate([y0, y1, y2, y3, y4, y5, y6, y7, y8], axis=2)

    # Y shape:[batch,N,9].

    lit_r = np.squeeze(np.matmul(y, np.expand_dims(gamma[:, 0, :], 2)), 2)  # [batch,N,9] * [batch,9,1] = [batch,N]
    lit_g = np.squeeze(np.matmul(y, np.expand_dims(gamma[:, 1, :], 2)), 2)
    lit_b = np.squeeze(np.matmul(y, np.expand_dims(gamma[:, 2, :], 2)), 2)

    # shape:[batch,N,3]
    face_color = np.stack([lit_r * face_texture[:, :, 0], lit_g * face_texture[:, :, 1], lit_b * face_texture[:, :, 2]],
                          axis=2)
    lighting = np.stack([lit_r, lit_g, lit_b], axis=2) * 128

    return face_color, lighting


def render(face_model, chi):
    fitted_r = mesh.transform.angle2matrix(chi[0:3])

    fitted_s = chi[3]
    fitted_t = chi[4:7].copy()
    fitted_t[2] = 1.0
    fitted_ep = np.expand_dims(chi[7:71], 1)
    fitted_sp = np.expand_dims(face_model.sp, 1)
    tex_coeff = np.expand_dims(face_model.tex, 1)
    expression1 = face_model.exBase.dot(fitted_ep)

    gamma = np.expand_dims(face_model.gamma, 0)

    vertices = face_model.mean_shape.T + face_model.idBase.dot(fitted_sp) + expression1
    vertices = np.reshape(vertices, [int(3), int(len(vertices) / 3)], 'F').T

    face_norm = compute_norm(np.expand_dims(vertices, 0), face_model)
    face_norm_r = np.matmul(face_norm, np.expand_dims(fitted_r, 0))

    colors = face_model.mean_tex.T + face_model.texBase.dot(tex_coeff)
    colors = np.reshape(colors, [int(3), int(len(colors) / 3)], 'F').T
    face_color, lighting = illumination_layer(np.expand_dims(colors, 0), face_norm_r, gamma)
    colors = face_color[0, :]
    colors = np.minimum(np.maximum(colors, 0), 255)
    transformed_vertices = mesh.transform.similarity_transform(vertices, fitted_s, fitted_r, fitted_t)
    projected_vertices = transformed_vertices.copy()  # using stantard camera & orth projection

    h = 512
    w = 512
    colors[mask3, :] = 255.0
    image_vertices = mesh.transform.to_image(projected_vertices, h, w)
    image = mesh.render.render_colors(image_vertices, triangles - 1, colors, h, w)

    return image


def gen_gaosi_filter(r, sigma):
    gauss_temp = np.ones(r * 2 - 1)
    for i in range(0, r * 2 - 1):
        gauss_temp[i] = np.exp(-(i - r) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * 3.1415926))
    return gauss_temp


mask3 = np.load('eyemask.npy')
# gaosifilter
r = 3
sigma = 1
gaosifilter = gen_gaosi_filter(r, sigma)
sumgaosi = np.sum(gaosifilter)
gaosifilter = gaosifilter / sumgaosi

trainnum = 'test3'

real_params = opt.real_params_path
realparams = np.load(open(real_params, 'rb'))
realparams = realparams['face']

idparams = realparams[0, 71:151]
texparams = realparams[0, 151:231]
gammaparams = realparams[0, 231:]

# --- 1. load model
facemodel = BFM()
nver = facemodel.idBase.shape[0] / 3
ntri = facemodel.tri.shape[0]
n_shape_para = facemodel.idBase.shape[1]
n_exp_para = facemodel.exBase.shape[1]
n_tex_para = facemodel.texBase.shape[1]

kpt_ind = facemodel.key_points
triangles = facemodel.tri

facemodel.sp = idparams
facemodel.tex = texparams
facemodel.gamma = gammaparams

for i in range(1, realparams.shape[0] + 1):
    if i % 1000 == 0:
        print(i)
    chi_next = realparams[i - 1, :71].copy()
    if 3 < i < realparams.shape[0] - 2:
        for j in range(6):
            chi_next[j] = np.sum([realparams[i - 3, j], realparams[i - 2, j], realparams[i - 1, j], realparams[i, j],
                                  realparams[i + 1, j]] * gaosifilter)
    image = render(facemodel, chi_next).astype(np.uint8)

    save_folder = opt.outpath
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    io.imsave(os.path.join(save_folder, str("%06d" % i) + '.jpg'), image)
