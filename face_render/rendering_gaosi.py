import os
import argparse
import numpy as np
from skimage import io
from face3d.mesh import transform, render as mesh_render
from load_data import BFM

parser = argparse.ArgumentParser(description='Render_setting')
parser.add_argument('--train_params_path', type=str, default='../audio2face/data/train3.npz')
parser.add_argument('--net_params_path', type=str, default='../examples/test-result/obama2.npz')
parser.add_argument('--BFM_model_path', type=str, default='/content/FACIAL/face_render/BFM/')
parser.add_argument('--outpath', type=str, default='../examples/rendering/')

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

    lit_r = np.squeeze(np.matmul(y, np.expand_dims(gamma[:, 0, :], 2)), 2)
    lit_g = np.squeeze(np.matmul(y, np.expand_dims(gamma[:, 1, :], 2)), 2)
    lit_b = np.squeeze(np.matmul(y, np.expand_dims(gamma[:, 2, :], 2)), 2)

    face_color = np.stack([lit_r * face_texture[:, :, 0], lit_g * face_texture[:, :, 1], lit_b * face_texture[:, :, 2]],
                          axis=2)
    lighting = np.stack([lit_r, lit_g, lit_b], axis=2) * 128

    return face_color, lighting


def render(face_model, chi):
    fitted_r = transform.angle2matrix(chi[0:3])

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
    transformed_vertices = transform.similarity_transform(vertices, fitted_s, fitted_r, fitted_t)
    projected_vertices = transformed_vertices.copy()  # using stantard camera & orth projection

    h = 512
    w = 512

    colors[mask3, :] = 255.0
    image_vertices = transform.to_image(projected_vertices, h, w)
    image = mesh_render.render_colors(image_vertices, triangles - 1, colors, h, w)
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

real_params = opt.train_params_path
realparams = np.load(open(real_params, 'rb'))
realparams = realparams['face']

std1 = np.std(realparams, axis=0)
mean1 = np.mean(realparams, axis=0)

net_params = opt.net_params_path

netparams = np.load(open(net_params, 'rb'))
netparams = netparams['face']
std2 = np.std(netparams, axis=0)
mean2 = np.mean(netparams, axis=0)

for i in range(6):
    netparams[:, i] = netparams[:, i] * std1[i] + mean1[i]

from scipy.signal import savgol_filter

netparams = savgol_filter(netparams, 7, 3, axis=0)

idparams = realparams[0, 71:151]
texparams = realparams[0, 151:231]
gamma_params = realparams[0, 231:]

# --- 1. load model
face_model = BFM(opt.BFM_model_path)
nver = face_model.idBase.shape[0] / 3
ntri = face_model.tri.shape[0]
n_shape_para = face_model.idBase.shape[1]
n_exp_para = face_model.exBase.shape[1]
n_tex_para = face_model.texBase.shape[1]

kpt_ind = face_model.key_points
triangles = face_model.tri

face_model.sp = idparams
face_model.tex = texparams
face_model.gamma = gamma_params

example_name = net_params.split('/')[-1].replace('.npz', '')
save_folder = opt.outpath + example_name
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

n = netparams.shape[0] // 10

for i in range(1, netparams.shape[0] + 1):
    if not i % n:
        print(i)

    chi_next = netparams[i - 1, :].copy()
    if 3 < i < netparams.shape[0] - 2:
        for j in range(6):
            chi_next[j] = np.sum([netparams[i - 3, j], netparams[i - 2, j], netparams[i - 1, j], netparams[i, j],
                                  netparams[i + 1, j]] * gaosifilter)
    image = render(face_model, chi_next).astype(np.uint8)
    io.imsave(os.path.join(save_folder, str("%06d" % i) + '.jpg'), image)
