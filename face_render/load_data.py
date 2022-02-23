import numpy as np
from PIL import Image
from scipy.io import loadmat, savemat
from array import array
import os


# define face_model for reconstruction
class BFM:
    def __init__(self, model_path='/content/FACIAL/face_render/BFM/'):
        model = loadmat(os.path.join(model_path, 'BFM_model_front.mat'))
        self.expEV = np.loadtxt(os.path.join(model_path, 'std_exp.txt'))
        self.mean_shape = model['meanshape']  # mean face shape
        self.idBase = model['idBase']  # identity basis
        self.exBase = model['exBase']  # expression basis
        self.mean_tex = model['meantex']  # mean face texture
        self.texBase = model['texBase']  # texture basis
        self.point_buf = model[
            'point_buf']  # adjacent face index for each vertex, starts from 1 (only used for calculating face normal)
        self.tri = model['tri']  # vertex index for each triangle face, starts from 1
        self.key_points = np.squeeze(model['keypoints']).astype(np.int32) - 1  # 68 face landmark index, starts from 0


# load expression basis
def load_exp_basis():
    n_vertex = 53215
    exp_bin = open('BFM/Exp_Pca.bin', 'rb')
    exp_dim = array('i')
    exp_dim.fromfile(exp_bin, 1)
    exp_mu = array('f')
    exp_pc = array('f')
    exp_mu.fromfile(exp_bin, 3 * n_vertex)
    exp_pc.fromfile(exp_bin, 3 * exp_dim[0] * n_vertex)

    exp_pc = np.array(exp_pc)
    exp_pc = np.reshape(exp_pc, [exp_dim[0], -1])
    exp_pc = np.transpose(exp_pc)

    exp_ev = np.loadtxt('BFM/std_exp.txt')

    return exp_pc, exp_ev


# transfer original BFM09 to our face model
def transfer_bfm09():
    original_bfm = loadmat('BFM/01_MorphableModel.mat')
    shape_pc = original_bfm['shapePC']  # shape basis
    shape_ev = original_bfm['shapeEV']  # corresponding eigen value
    shape_mu = original_bfm['shapeMU']  # mean face
    tex_pc = original_bfm['texPC']  # texture basis
    tex_ev = original_bfm['texEV']  # eigen value
    tex_mu = original_bfm['texMU']  # mean texture

    exp_pc, exp_ev = load_exp_basis()

    # transfer BFM09 to our face model

    id_base = shape_pc * np.reshape(shape_ev, [-1, 199])
    id_base = id_base / 1e5  # unify the scale to decimeter
    id_base = id_base[:, :80]  # use only first 80 basis

    ex_base = exp_pc * np.reshape(exp_ev, [-1, 79])
    ex_base = ex_base / 1e5  # unify the scale to decimeter
    ex_base = ex_base[:, :64]  # use only first 64 basis

    tex_base = tex_pc * np.reshape(tex_ev, [-1, 199])
    tex_base = tex_base[:, :80]  # use only first 80 basis

    # our face model is cropped align face landmarks which contains only 35709 vertex.
    # original BFM09 contains 53490 vertex, and expression basis provided by JuYong contains 53215 vertex.
    # thus, we select corresponding vertex to get our face model.

    index_exp = loadmat('BFM/BFM_front_idx.mat')
    index_exp = index_exp['idx'].astype(np.int32) - 1  # starts from 0 (to 53215)

    index_shape = loadmat('BFM/BFM_exp_idx.mat')
    index_shape = index_shape['trimIndex'].astype(np.int32) - 1  # starts from 0 (to 53490)
    index_shape = index_shape[index_exp]

    id_base = np.reshape(id_base, [-1, 3, 80])
    id_base = id_base[index_shape, :, :]
    id_base = np.reshape(id_base, [-1, 80])

    tex_base = np.reshape(tex_base, [-1, 3, 80])
    tex_base = tex_base[index_shape, :, :]
    tex_base = np.reshape(tex_base, [-1, 80])

    ex_base = np.reshape(ex_base, [-1, 3, 64])
    ex_base = ex_base[index_exp, :, :]
    ex_base = np.reshape(ex_base, [-1, 64])

    mean_shape = np.reshape(shape_mu, [-1, 3]) / 1e5
    mean_shape = mean_shape[index_shape, :]
    mean_shape = np.reshape(mean_shape, [1, -1])

    meantex = np.reshape(tex_mu, [-1, 3])
    meantex = meantex[index_shape, :]
    meantex = np.reshape(meantex, [1, -1])

    # other info contains triangles, region used for computing photometric loss,
    # region used for skin texture regularization, and 68 landmarks index etc.
    other_info = loadmat('BFM/facemodel_info.mat')
    front_mask2_idx = other_info['frontmask2_idx']
    skin_mask = other_info['skinmask']
    key_points = other_info['keypoints']
    point_buf = other_info['point_buf']
    tri = other_info['tri']
    tri_mask2 = other_info['tri_mask2']

    # save our face model
    savemat(
        'BFM/BFM_model_front.mat',
        {
            'meanshape': mean_shape, 'meantex': meantex, 'idBase': id_base, 'exBase': ex_base, 'texBase': tex_base,
            'tri': tri, 'point_buf': point_buf, 'tri_mask2': tri_mask2, 'keypoints': key_points,
            'frontmask2_idx': front_mask2_idx, 'skinmask': skin_mask
        }
    )


# load landmarks for standard face, which is used for image preprocessing
def load_lm3d():
    lm3_d = loadmat('./BFM/similarity_Lm3D_all.mat')
    lm3_d = lm3_d['lm']

    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm3_d = np.stack([lm3_d[lm_idx[0], :], np.mean(lm3_d[lm_idx[[1, 2]], :], 0), np.mean(lm3_d[lm_idx[[3, 4]], :], 0),
                     lm3_d[lm_idx[5], :], lm3_d[lm_idx[6], :]], axis=0)
    lm3_d = lm3_d[[1, 2, 0, 3, 4], :]

    return lm3_d


# load input images and corresponding 5 landmarks
def load_img(img_path, lm_path):
    image = Image.open(img_path)
    lm = np.loadtxt(lm_path)

    return image, lm


# save 3D face to obj file
def save_obj(path, v, f, c):
    with open(path, 'w') as file:
        for i in range(len(v)):
            file.write('v %f %f %f %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2], c[i, 0], c[i, 1], c[i, 2]))

        file.write('\n')

        for i in range(len(f)):
            file.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))

    file.close()
