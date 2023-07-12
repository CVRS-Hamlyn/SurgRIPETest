from lib.datasets.dataset_catalog import DatasetCatalog
from lib.config import cfg
import pycocotools.coco as coco
import numpy as np
from lib.utils.pvnet import pvnet_pose_utils, pvnet_data_utils
import os
from lib.utils.linemod import linemod_config
import torch
if cfg.test.icp:
    from lib.utils import icp_utils
from PIL import Image
from lib.utils.img_utils import read_depth
from scipy import spatial
import math

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


class Evaluator:

    def __init__(self, result_dir):
        self.result_dir = result_dir
        args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args['ann_file']
        # self.ann_file = 'data/custom_PG/all.json'
        # self.ann_file = 'data/custom_LND/all.json'
        # self.ann_file ='data/custom_PG/all_test.json'
        print("!!!evalutor ann file", self.ann_file)
        self.coco = coco.COCO(self.ann_file)

        data_root = args['data_root']
        # model_path = 'data/custom/model.ply'
        # self.model = pvnet_data_utils.get_ply_model(model_path)
        # model_path = 'data/custom/convert_Tube45mm_53mm.npy'
        # model_path = 'data/custom_LND/LND_cut_notip.npy'
        model_path = 'data/custom_PG/PG_cut_notip.npy'
        self.model = np.load(model_path)
        # self.diameter = np.loadtxt('data/custom/diameter.txt').item()
        self.diameter = np.array([54])
        self.icp_render = icp_utils.SynRenderer(cfg.cls_type) if cfg.test.icp else None

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
        model_2d_pred = pvnet_pose_utils.project(self.model, K, pose_pred)
        model_2d_targets = pvnet_pose_utils.project(self.model, K, pose_targets)
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


        if icp:
            self.icp_add.append(mean_dist < diameter)
        else:
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

    def icp_refine(self, pose_pred, anno, output, K):
        depth = read_depth(anno['depth_path'])
        mask = torch.argmax(output['seg'], dim=1)[0].detach().cpu().numpy()
        if pose_pred[2, 3] <= 0 or np.sum(mask) < 20:
            return pose_pred
        depth[mask != 1] = 0
        pose_pred_tmp = pose_pred.copy()
        pose_pred_tmp[:3, 3] = pose_pred_tmp[:3, 3] * 1000
        R_refined, t_refined = icp_utils.icp_refinement(depth, self.icp_render, pose_pred_tmp[:3, :3], pose_pred_tmp[:3, 3], K.copy(), (depth.shape[1], depth.shape[0]), depth_only=True,            max_mean_dist_factor=5.0)
        R_refined, _ = icp_utils.icp_refinement(depth, self.icp_render, R_refined, t_refined, K.copy(), (depth.shape[1], depth.shape[0]), no_depth=True)
        pose_pred = np.hstack((R_refined, t_refined.reshape((3, 1)) / 1000))
        return pose_pred

    def evaluate(self, output, batch):
        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()

        img_id = int(batch['img_id'][0])
        # print("anns, img_id",img_id)
        # print(self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id)))
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        kpt_3d = np.concatenate([anno['fps_3d'], [anno['center_3d']]], axis=0)
        K = np.array(anno['K'])

        pose_gt = np.array(anno['pose'])
        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)
        if self.icp_render is not None:
            pose_pred_icp = self.icp_refine(pose_pred.copy(), anno, output, K)
            self.add_metric(pose_pred_icp, pose_gt, icp=True)
        self.projection_2d(pose_pred, pose_gt, K)
        if cfg.cls_type in ['eggbox', 'glue','shaft']:
            self.add_metric(pose_pred, pose_gt, syn=True)
        else:
            self.add_metric(pose_pred, pose_gt)
        self.cm_degree_5_metric(pose_pred, pose_gt)
        self.trans_rot_error(pose_pred, pose_gt)
        self.mask_iou(output, batch)
    
    def save(self,save_path):
        proj2d = np.array(self.proj2d)
        add = np.array(self.add)
        add_dist = np.array(self.add_dist)
        cmd5 = np.array(self.cmd5)
        ap = np.array(self.mask_ap)
        trans_error = np.array(self.trans_error)
        rot_error = np.array(self.rot_error)
        adds_dist = np.array(self.adds_dist)
        np.save(save_path+'/trans_error.npy',trans_error)
        np.save(save_path+'/rot_error.npy',rot_error)
        np.save(save_path+'/add_dist.npy',add_dist)
        np.save(save_path+'/adds_dist.npy',adds_dist)
        print('record saved!')

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
