from lib.config import cfg
import numpy as np
import os
import torch
from PIL import Image
from scipy import spatial
import math
import cv2

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 10000*1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    # return np.array([x, y, z])
    return np.rad2deg([x, y, z])

def pnp(points_3d, points_2d, camera_matrix, method=cv2.SOLVEPNP_ITERATIVE):
    try:
        dist_coeffs = pnp.dist_coeffs
    except:
        dist_coeffs = np.zeros(shape=[8, 1], dtype='float64')

    assert points_3d.shape[0] == points_2d.shape[0], 'points 3D and points 2D must have same number of vertices'
    if method == cv2.SOLVEPNP_EPNP:
        points_3d = np.expand_dims(points_3d, 0)
        points_2d = np.expand_dims(points_2d, 0)

    points_2d = np.ascontiguousarray(points_2d.astype(np.float64))
    points_3d = np.ascontiguousarray(points_3d.astype(np.float64))
    camera_matrix = camera_matrix.astype(np.float64)
    _, R_exp, t = cv2.solvePnP(points_3d,
                               points_2d,
                               camera_matrix,
                               dist_coeffs,
                               flags=method)
    # , None, None, False, cv2.SOLVEPNP_UPNP)

    # R_exp, t, _ = cv2.solvePnPRansac(points_3D,
    #                            points_2D,
    #                            cameraMatrix,
    #                            distCoeffs,
    #                            reprojectionError=12.0)

    R, _ = cv2.Rodrigues(R_exp)
    # trans_3d=np.matmul(points_3d,R.transpose())+t.transpose()
    # if np.max(trans_3d[:,2]<0):
    #     R=-R
    #     t=-t

    return np.concatenate([R, t], axis=-1)


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy


class Evaluator:

    def __init__(self, result_dir):
        self.result_dir = result_dir
        self.diameter = np.array([54])

        self.proj2d = []
        self.add = []
        self.icp_add = []
        self.cmd5 = []
        self.mask_ap = []
        self.add_dist = []
        self.trans_error = []
        self.rot_error = []
        self.adds = []
        self.adds_dist = []


    def trans_rot_error(self, pose_pred, pose_targets):
        gt_pose_rot = pose_targets[:3,:3]
        gt_pose_trans = pose_targets[:3,-1]
        pred_pose_rot = pose_pred[:3,:3]
        pred_pose_trans = pose_pred[:3,-1]
        try:
            b_rot_angle = rotationMatrixToEulerAngles(gt_pose_rot)
            reproject_b_rot_angle = rotationMatrixToEulerAngles(pred_pose_rot)
        except:
            print("rotation is not orthogonal")
        trans_error = np.linalg.norm(gt_pose_trans-pred_pose_trans)
        rot_error = np.absolute(b_rot_angle-reproject_b_rot_angle)
        self.trans_error.append(trans_error)
        self.rot_error.append(rot_error)

    def projection_2d(self, pose_pred, pose_targets, K, threshold=5):
        model_2d_pred = project(self.model, K, pose_pred)
        model_2d_targets = project(self.model, K, pose_targets)
        proj_mean_diff = np.mean(np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1))

        self.proj2d.append(proj_mean_diff < threshold)

    def add_metric(self, pose_pred, pose_targets, icp=False, syn=False, percentage=0.1):
        diameter = self.diameter * percentage
        model_pred = np.dot(self.model, pose_pred[:, :3].T) + pose_pred[:, 3]
        model_targets = np.dot(self.model, pose_targets[:, :3].T) + pose_targets[:, 3]

        if syn:
            mean_dist_index = spatial.cKDTree(model_pred)
            mean_dist, _ = mean_dist_index.query(model_targets, k=1)
            mean_dist = np.mean(mean_dist)
        else:
            mean_dist = np.mean(np.linalg.norm(model_pred - model_targets, axis=-1))

        add_error = np.mean(np.linalg.norm(model_pred - model_targets, axis=-1))

        adds_error_index = spatial.cKDTree(model_pred)
        adds_error, _ = adds_error_index.query(model_targets, k=1)
        adds_error = np.mean(adds_error)

        self.add_dist.append(add_error)
        self.add.append(add_error < diameter)
        self.adds_dist.append(adds_error)
        self.adds.append(adds_error < diameter)

    def cm_degree_5_metric(self, pose_pred, pose_targets):
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_targets[:, 3])
        rotation_diff = np.dot(pose_pred[:, :3], pose_targets[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
        self.cmd5.append(translation_distance < 5 and angular_distance < 5)

    def mm_degree_5_metric(self, pose_pred, pose_targets):
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_targets[:, 3])
        rotation_diff = np.dot(pose_pred[:, :3], pose_targets[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
        self.cmd5.append(translation_distance < 5 and angular_distance < 5)

    def mask_iou(self, output, batch):
        mask_pred = torch.argmax(output['seg'], dim=1)[0].detach().cpu().numpy()
        mask_gt = batch['mask'][0].detach().cpu().numpy()
        iou = (mask_pred & mask_gt).sum() / (mask_pred | mask_gt).sum()
        self.mask_ap.append(iou > 0.7)

    def evaluate(self, pose_gt, pose_pred):
        self.projection_2d(pose_pred, pose_gt, K)
        if cfg.cls_type in ['eggbox', 'glue','shaft']:
            self.add_metric(pose_pred, pose_gt, syn=True)
        else:
            self.add_metric(pose_pred, pose_gt)
        self.cm_degree_5_metric(pose_pred, pose_gt)
        self.trans_rot_error(pose_pred, pose_gt)
        # self.mask_iou(output, batch)

    def summarize(self,save_path=None):
        proj2d = np.mean(self.proj2d)
        add = np.mean(self.add)
        add_dist = np.mean(self.add_dist)
        adds = np.mean(self.adds)
        adds_dist = np.mean(self.adds_dist)
        cmd5 = np.mean(self.cmd5)
        ap = np.mean(self.mask_ap)
        trans_error = np.mean(self.trans_error)
        rot_error = np.mean(self.rot_error)
        print('2d projections metric: {}'.format(proj2d))
        print('ADD metric: {}'.format(add))
        print('ADD mean distance', add_dist)
        print('ADD-S metric: {}'.format(adds))
        print('ADD-S mean distance', adds_dist)
        print('5 cm 5 degree metric: {}'.format(cmd5))
        print('mask ap70: {}'.format(ap))
        print('trans_error: {}'.format(trans_error))
        print('rot_error: {}'.format(rot_error))
        
        self.proj2d = []
        self.add = []
        self.cmd5 = []
        self.mask_ap = []
        self.icp_add = []
        self.trans_error = []
        self.rot_error = []
        results = {'proj2d': proj2d, 'add': add,'add-s': adds, 'ADD-distance': add_dist, 'ADDS-distance': adds_dist, 'cmd5': cmd5, 'trans_error': trans_error, 'rot_error':rot_error}
        if save_path is not None:
            np.save(save_path+'/results.npy',results)
        return {'proj2d': proj2d, 'add': add, 'add-s': adds, 'ADD-distance': add_dist, 'ADDS-distance': adds_dist, 'cmd5': cmd5, 'trans_error': trans_error, 'rot_error':rot_error}
