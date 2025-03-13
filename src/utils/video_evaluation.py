import numpy as np
import pinocchio as pin

def get_average_rot_errors_dt(est_poses, gt_poses, dts, sym_axis=None, N_symmetries=101):
    errors = []
    s = np.sum(dts)
    for dt in dts:
        e = get_rot_errors(est_poses, gt_poses, dt, sym_axis=sym_axis, N_symmetries=N_symmetries)
        e = np.mean(e) / dt
        errors.append(e)
    return np.mean(errors)


def get_average_depth_errors_dt(est_poses, gt_poses, est_scale, gt_scale, dts, est_pts=None):
    est_poses_new = align_object_origins(est_poses, gt_poses, est_scale, est_pts=est_pts)

    errors = []
    for dt in dts:
        e = get_translation_errors_depth(est_poses_new, gt_poses, est_scale, gt_scale, dt)
        e = np.mean(e) / dt
        errors.append(e)
    return np.mean(errors)


def get_average_proj_errors_dt(est_poses, gt_poses, est_scale, gt_scale, dts, w, h, K=None, est_pts=None):
    diag = np.sqrt(w**2 + h**2)
    est_poses_new = align_object_origins(est_poses, gt_poses, est_scale, est_pts=est_pts)
    
    errors = []
    for dt in dts:
        e = get_translation_errors_proj(est_poses_new, gt_poses, dt=dt, w=w, h=h, K=K)
        e = np.mean(e) / dt
        errors.append(e)
    return np.mean(errors)/diag*100


def get_rot_errors(est_poses, gt_poses, dt, sym_axis=None, N_symmetries=101):
    errors = []
    N = len(est_poses)

    for t1 in range(N-dt):
        t2 = t1 + dt
        
        R1_est = est_poses[t1].rotation
        R2_est = est_poses[t2].rotation
        R1_gt = gt_poses[t1].rotation
        R2_gt = gt_poses[t2].rotation

        if sym_axis is not None:
            syms = [pin.exp(sym_axis*a) for a in np.linspace(-np.pi, np.pi, N_symmetries)]
        else:
            syms = [np.eye(3)]

        e = min(rot_error_in_cframe(R1_est, R2_est, R1_gt, R2_gt@ S) for S in syms)
        errors.append(e)

    return errors


def rot_error_in_cframe(R1_est, R2_est, R1_gt, R2_gt):
    a = pin.log(R2_est @ R1_est.T) # spatial velocity
    b = pin.log(R2_gt @ R1_gt.T)  # spatial velocity
    return np.linalg.norm(a - b)


def get_translation_errors_depth(est_poses, gt_poses, est_scale, gt_scale, dt):
    errors = []
    N = len(est_poses)

    for t1 in range(N-dt):
        t2 = t1 + dt

        v_est = np.linalg.norm(est_poses[t1].translation) / est_scale - np.linalg.norm(est_poses[t2].translation) / est_scale
        v_gt = np.linalg.norm(gt_poses[t1].translation) / gt_scale - np.linalg.norm(gt_poses[t2].translation) / gt_scale
        
        errors.append(np.linalg.norm(v_est - v_gt))
    return errors


def get_translation_errors_proj(est_poses, gt_poses, dt, w, h, K=None):
    errors = []
    N = len(est_poses)

    for t1 in range(N-dt):
        t2 = t1 + dt

        p1_est = project(est_poses[t1].translation, K=K, w=w, h=h)
        p2_est = project(est_poses[t2].translation, K=K, w=w, h=h)
        v_est = p2_est - p1_est
        
        p1_gt = project(gt_poses[t1].translation, K=K, w=w, h=h)
        p2_gt = project(gt_poses[t2].translation, K=K, w=w, h=h)
        v_gt = p2_gt - p1_gt
        
        errors.append(np.linalg.norm(v_est - v_gt))

    return errors


def project(x, w, h, K=None):
    if K is None:
        f = np.sqrt(w**2 + h**2)
        K = np.array(
            [[f, 0, w/2.0],
             [0, f, h/2.0],
             [0, 0,   1.0]])
    u = K @ x
    u = u[:2] / u[2]
    return u


def align_object_origins(poses1, poses2, scale, ref_frame_idxs=None, est_pts=None):
    if ref_frame_idxs is None:
        ref_frame_idxs = range(len(poses1))

    origins_in_o1 = []
    for ref_idx in ref_frame_idxs:
        o1 = poses1[ref_idx].translation
        o2 = poses2[ref_idx].translation
        
        x = o2/np.linalg.norm(o2)*np.linalg.norm(o1)
        origin_in_o1 = poses1[ref_idx].actInv(x)
        if np.linalg.norm(origin_in_o1) < scale:
            origins_in_o1.append(origin_in_o1)
        del origin_in_o1

    if len(origins_in_o1) == 0:
        return poses1

    origin_in_o1 = np.mean(origins_in_o1, axis=0)

    # limit the change of the origin using the scale of the object:
    norm = np.linalg.norm(origin_in_o1)
    max_change = scale / 2.0
    if norm > max_change:
        origin_in_o1 = origin_in_o1 / norm * max_change

    #est_pts_new = est_pts - x_in_o1
    poses1_new = change_object_origin(poses1, origin_in_o1)
    return poses1_new


def svd_pointcloud_align(pointcloud):
    assert pointcloud.shape[1] == 3

    X = pointcloud - np.mean(pointcloud, axis=0)
    _, _, V = np.linalg.svd(X.T@X)
    return pointcloud@V.T


def change_object_origin(poses, new_origin):
    T = pin.SE3(np.eye(3), new_origin)
    poses_new = [p*T for p in poses]
    #poses_new = [pin.SE3(p.rotation, p.rotation @ new_origin + p.translation) for p in poses] 
    return poses_new
