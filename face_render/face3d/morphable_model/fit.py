"""
Estimating parameters about vertices: shape para, exp para, pose para(s, R, t)
"""
import numpy as np
from .. import mesh

''' TODO: a clear document. 
Given: image_points, 3D Model, Camera Matrix(s, R, t2d)
Estimate: shape parameters, expression parameters

Inference: 

    projected_vertices = s*P*R(mu + shape + exp) + t2d  --> image_points
    s*P*R*shape + s*P*R(mu + exp) + t2d --> image_poitns

    # Define:
    X = vertices
    x_hat = projected_vertices
    x = image_points
    A = s*P*R
    b = s*P*R(mu + exp) + t2d
    ==>
    x_hat = A*shape + b  (2 x n)

    A*shape (2 x n)
    shape = reshape(shapePC * sp) (3 x n)
    shapePC*sp : (3n x 1)

    * flatten:
    x_hat_flatten = A*shape + b_flatten  (2n x 1)
    A*shape (2n x 1)
    --> A*shapePC (2n x 199)  sp: 199 x 1
    
    # Define:
    pc_2d = A* reshape(shapePC)
    pc_2d_flatten = flatten(pc_2d) (2n x 199)

    =====>
    x_hat_flatten = pc_2d_flatten * sp + b_flatten ---> x_flatten (2n x 1)

    Goals:
    (ignore flatten, pc_2d-->pc)
    min E = || x_hat - x || + lambda*sum(sp/sigma)^2
          = || pc * sp + b - x || + lambda*sum(sp/sigma)^2

    Solve:
    d(E)/d(sp) = 0
    2 * pc' * (pc * sp + b - x) + 2 * lambda * sp / (sigma' * sigma) = 0

    Get:
    (pc' * pc + lambda / (sigma'* sigma)) * sp  = pc' * (x - b)

'''


def estimate_shape(x, shape_mu, shape_pc, expression, s, r, t2d, lamb=3000):
    """
    Args:
        x: (2, n). image points (to be fitted)
        shape_mu: (3n, 1)
        shape_pc: (3n, n_sp)
        shapeEV: (n_sp, 1)
        expression: (3, n)
        s: scale
        r: (3, 3). rotation matrix
        t2d: (2,). 2d translation
        lamb: regulation coefficient

    Returns:
        shape_para: (n_sp, 1) shape parameters(coefficients)
    """
    x = x.copy()
    assert (shape_mu.shape[0] == shape_pc.shape[0])
    assert (shape_mu.shape[0] == x.shape[1] * 3)

    dof = shape_pc.shape[1]

    n = x.shape[1]
    # sigma = shapeEV
    sigma = np.ones((shape_pc.shape[1], 1), dtype=np.float32)
    t2d = np.array(t2d)
    P = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    A = s * P.dot(r)

    # --- calc pc
    pc_3d = np.resize(shape_pc.T, [dof, n, 3])  # 199 x n x 3
    pc_3d = np.reshape(pc_3d, [dof * n, 3])
    pc_2d = pc_3d.dot(A.T.copy())  # 199 x n x 2

    pc = np.reshape(pc_2d, [dof, -1]).T  # 2n x 199

    # --- calc b
    # shapeMU
    mu_3d = np.resize(shape_mu, [n, 3]).T  # 3 x n
    # expression
    exp_3d = expression
    # 
    b = A.dot(mu_3d + exp_3d) + np.tile(t2d[:, np.newaxis], [1, n])  # 2 x n
    b = np.reshape(b.T, [-1, 1])  # 2n x 1

    # --- solve
    equation_left = np.dot(pc.T, pc) + lamb * np.diagflat(1 / sigma ** 2)
    x = np.reshape(x.T, [-1, 1])
    equation_right = np.dot(pc.T, x - b)

    shape_para = np.dot(np.linalg.inv(equation_left), equation_right)

    return shape_para


def estimate_expression(x, shapeMU, expPC, shape, s, R, t2d, lamb=2000):
    """
    Args:
        x: (2, n). image points (to be fitted)
        shapeMU: (3n, 1)
        expPC: (3n, n_ep)
        expEV: (n_ep, 1)
        shape: (3, n)
        s: scale
        R: (3, 3). rotation matrix
        t2d: (2,). 2d translation
        lambda: regulation coefficient

    Returns:
        exp_para: (n_ep, 1) shape parameters(coefficients)
    """
    x = x.copy()
    assert (shapeMU.shape[0] == expPC.shape[0])
    assert (shapeMU.shape[0] == x.shape[1] * 3)

    dof = expPC.shape[1]

    n = x.shape[1]
    # sigma = expEV
    sigma = np.ones((expPC.shape[1], 1), dtype=np.float32)
    t2d = np.array(t2d)
    P = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    A = s * P.dot(R)

    # --- calc pc
    pc_3d = np.resize(expPC.T, [dof, n, 3])
    pc_3d = np.reshape(pc_3d, [dof * n, 3])
    pc_2d = pc_3d.dot(A.T)
    pc = np.reshape(pc_2d, [dof, -1]).T  # 2n x 29

    # --- calc b
    # shapeMU
    mu_3d = np.resize(shapeMU, [n, 3]).T  # 3 x n
    # expression
    shape_3d = shape
    # 
    b = A.dot(mu_3d + shape_3d) + np.tile(t2d[:, np.newaxis], [1, n])  # 2 x n
    b = np.reshape(b.T, [-1, 1])  # 2n x 1

    # --- solve
    equation_left = np.dot(pc.T, pc) + lamb * np.diagflat(1 / sigma ** 2)
    x = np.reshape(x.T, [-1, 1])
    equation_right = np.dot(pc.T, x - b)

    exp_para = np.dot(np.linalg.inv(equation_left), equation_right)

    return exp_para


def fit_points(x, X_ind, model, id_coeff, n_ep, max_iter=4):
    """
    Args:
        id_coeff:
        x: (n, 2) image points
        X_ind: (n,) corresponding Model vertex indices
        model: 3DMM
        max_iter: iteration
    Returns:
        sp: (n_sp, 1). shape parameters
        ep: (n_ep, 1). exp parameters
        s, R, t
    """
    x = x.copy().T

    # -- init
    # sp = np.zeros((n_sp, 1), dtype = np.float32)
    sp = id_coeff.T
    n_sp = sp.shape[0]
    ep = np.zeros((n_ep, 1), dtype=np.float32)

    # -------------------- estimate
    X_ind_all = np.tile(X_ind[np.newaxis, :], [3, 1]) * 3
    X_ind_all[1, :] += 1
    X_ind_all[2, :] += 2
    valid_ind = X_ind_all.flatten('F')

    shapeMU = model.mean_shape[:, valid_ind].T
    shapePC = model.idBase[valid_ind, :n_sp]
    expPC = model.exBase[valid_ind, :n_ep]

    for i in range(max_iter):
        X = shapeMU + shapePC.dot(sp) + expPC.dot(ep)
        X = np.reshape(X, [int(len(X) / 3), 3]).T

        # ----- estimate pose
        P = mesh.transform.estimate_affine_matrix_3d22d(X.T, x.T)
        s, R, t = mesh.transform.P2sRt(P)
        rx, ry, rz = mesh.transform.matrix2angle(R)
        # print('Iter:{}; estimated pose: s {}, rx {}, ry {}, rz {}, t1 {}, t2 {}'.format(i, s, rx, ry, rz, t[0], t[1]))

        # ----- estimate shape
        # expression
        shape = shapePC.dot(sp)
        shape = np.reshape(shape, [int(len(shape) / 3), 3]).T
        ep = estimate_expression(x, shapeMU, expPC, shape, s, R, t[:2], lamb=1)

        # #shape
        # expression = expPC.dot(ep)
        # expression = np.reshape(expression, [int(len(expression)/3), 3]).T
        # sp = estimate_shape(x, shapeMU, shapePC, expression, s, R, t[:2], lamb = 1)

    return sp, ep, s, R, t


# ---------------- fit
def fit_points1(x, X_ind, model, n_sp, n_ep, max_iter=4):
    """
    Args:
        x: (n, 2) image points
        X_ind: (n,) corresponding Model vertex indices
        model: 3DMM
        max_iter: iteration
    Returns:
        sp: (n_sp, 1). shape parameters
        ep: (n_ep, 1). exp parameters
        s, R, t
    """
    x = x.copy().T

    # -- init
    sp = np.zeros((n_sp, 1), dtype=np.float32)
    ep = np.zeros((n_ep, 1), dtype=np.float32)

    # -------------------- estimate
    X_ind_all = np.tile(X_ind[np.newaxis, :], [3, 1]) * 3
    X_ind_all[1, :] += 1
    X_ind_all[2, :] += 2
    valid_ind = X_ind_all.flatten('F')

    shapeMU = model['shapeMU'][valid_ind, :]
    shapePC = model['shapePC'][valid_ind, :n_sp]
    expPC = model['expPC'][valid_ind, :n_ep]

    for i in range(max_iter):
        X = shapeMU + shapePC.dot(sp) + expPC.dot(ep)
        X = np.reshape(X, [int(len(X) / 3), 3]).T

        # ----- estimate pose
        P = mesh.transform.estimate_affine_matrix_3d22d(X.T, x.T)
        s, R, t = mesh.transform.P2sRt(P)
        rx, ry, rz = mesh.transform.matrix2angle(R)
        # print('Iter:{}; estimated pose: s {}, rx {}, ry {}, rz {}, t1 {}, t2 {}'.format(i, s, rx, ry, rz, t[0], t[1]))

        # ----- estimate shape
        # expression
        shape = shapePC.dot(sp)
        shape = np.reshape(shape, [int(len(shape) / 3), 3]).T
        ep = estimate_expression(x, shapeMU, expPC, model['expEV'][:n_ep, :], shape, s, R, t[:2], lamb=20)

        # shape
        # expression = expPC.dot(ep)
        # expression = np.reshape(expression, [int(len(expression)/3), 3]).T
        # sp = estimate_shape(x, shapeMU, shapePC, model['shapeEV'][:n_sp,:], expression, s, R, t[:2], lamb = 40)

    return sp, ep, s, R, t


# ---------------- fitting process
def fit_points_for_show(x, X_ind, model, id_coeff, n_ep, max_iter=4):
    """
    Args:
        x: (n, 2) image points
        X_ind: (n,) corresponding Model vertex indices
        model: 3DMM
        max_iter: iteration
    Returns:
        sp: (n_sp, 1). shape parameters
        ep: (n_ep, 1). exp parameters
        s, R, t
    """
    x = x.copy().T

    # -- init
    sp = id_coeff.T
    n_sp = sp.shape[0]
    ep = np.zeros((n_ep, 1), dtype=np.float32)

    # -------------------- estimate
    X_ind_all = np.tile(X_ind[np.newaxis, :], [3, 1]) * 3
    X_ind_all[1, :] += 1
    X_ind_all[2, :] += 2
    valid_ind = X_ind_all.flatten('F')

    shapeMU = model.mean_shape[:, valid_ind].T
    shapePC = model.idBase[valid_ind, :n_sp]
    expPC = model.exBase[valid_ind, :n_ep]

    s = 50
    r = mesh.transform.angle2matrix([0, 0, 0])
    t = [0, 0, 0]
    lsp = []
    lep = []
    ls = []
    lR = []
    lt = []
    for i in range(max_iter):
        X = shapeMU + shapePC.dot(sp) + expPC.dot(ep)
        X = np.reshape(X, [int(len(X) / 3), 3]).T
        lsp.append(sp)
        lep.append(ep)
        ls.append(s), lR.append(r), lt.append(t)

        # ----- estimate pose
        p = mesh.transform.estimate_affine_matrix_3d22d(X.T, x.T)
        s, r, t = mesh.transform.P2sRt(p)
        lsp.append(sp)
        lep.append(ep)
        ls.append(s), lR.append(r), lt.append(t)

        # ----- estimate shape
        # expression
        shape = shapePC.dot(sp)
        shape = np.reshape(shape, [int(len(shape) / 3), 3]).T
        ep = estimate_expression(x, shapeMU, expPC, shape, s, r, t[:2], lamb=1)
        lsp.append(sp)
        lep.append(ep)
        ls.append(s), lR.append(r), lt.append(t)

        # shape
        # expression = expPC.dot(ep)
        # expression = np.reshape(expression, [int(len(expression)/3), 3]).T
        # sp = estimate_shape(x, shapeMU, shapePC,  expression, s, R, t[:2], lamb = 1)

    # print('ls', ls)
    # print('lR', lR)
    return np.array(lsp), np.array(lep), np.array(ls), np.array(lR), np.array(lt)
