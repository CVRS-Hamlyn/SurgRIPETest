import numpy as np
import cv2
import os
import yaml

def show_mask(im, rvecs, tvecs, cam_matrix, dist_coeff, pts_3d):
    imgpts, jac = cv2.projectPoints(pts_3d, rvecs, tvecs, cam_matrix, dist_coeff)

    image_mask = np.zeros(im.shape[:2],dtype = np.uint8)
    radius = 1
    thickness = 1

    height,width,_ = im.shape

    for i, pt in enumerate(imgpts):
        pt_x = int(pt[0,0])
        pt_y = int(pt[0,1])

        if pt_x<width and pt_x>-1 and pt_y<height and pt_y>-1:
            # use the BGR format to match the original image type
            cv2.circle(image_mask,(pt_x, pt_y), radius, 255, thickness)

    thresh = cv2.threshold(image_mask, 30, 255, cv2.THRESH_BINARY)[1]

    # contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    contours = contours[0]
    if contours is None:
        return None
    if len(contours)==0:
        return None

    cnt = max(contours, key=cv2.contourArea)

    image_mask = np.zeros(im.shape[:2],dtype = np.uint8)
    cv2.drawContours(image_mask, [cnt], -1, 255, -1)

    return image_mask


def show_axis(im, rvecs, tvecs, cam_matrix, dist_coeff, length, is_show=False):
    axis = np.float32([[0, 0, 0], [length,0,0], [0,length,0], [0,0,length]]).reshape(-1,3)
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, cam_matrix, dist_coeff)
    imgpts = np.rint(imgpts).astype(int)
    frame_centre = tuple(imgpts[0].ravel())

    thickness = 2
    im = cv2.line(im, frame_centre, tuple(imgpts[3].ravel()), (255,0,0), thickness, cv2.LINE_AA)#B 3 Z
    im = cv2.line(im, frame_centre, tuple(imgpts[2].ravel()), (0,255,0), thickness, cv2.LINE_AA)#G 2 Y
    im = cv2.line(im, frame_centre, tuple(imgpts[1].ravel()), (0,0,255), thickness, cv2.LINE_AA)#R 1 X

    if is_show:
        cv2.imshow("image", im)
        cv2.waitKey(0)

    return im


def show_blend_mask(im1, im2, alpha=0.5):
    if len(im2.shape)<3:
        im2 = cv2.cvtColor(im2,cv2.COLOR_GRAY2RGB)
    dst = cv2.addWeighted(im1, 1, im2, alpha, 0)

    return dst


def vis(data_path,config,camera_matrix,dist_coefs,project_3d=False,img_format='.png'):
    pts_3d = np.load(data_path+config['3d_model'])

    imgs_path = data_path+'image/'
    img_paths = os.listdir(imgs_path)

    img_idxs = []
    for img_path in img_paths:
        img_idx = img_path[:-4]
        img_idxs.append(int(img_idx))
    if not os.path.exists(data_path+'mask_rgb/'):
        os.makedirs(data_path+'mask_rgb/')

    for img_idx in img_idxs:

        img_path = data_path+'image/{}{}'.format(img_idx,img_format)
        dist_coefs = None
        im = cv2.imread(img_path)

        pose_path = data_path+'pose/{}.npy'.format(img_idx)
        pose = np.load(pose_path)

        r = pose[:,:3][:3]
        t = pose[:,-1][:3]

        ### only for demonstration
        if project_3d:  ### If project 3D model to 2D image
            mask = show_mask(im, r, t, camera_matrix, dist_coefs, pts_3d)
        else:           ### If load 2D mask
            mask_path = data_path+'mask/{}{}'.format(img_idx,img_format)
            mask = cv2.imread(mask_path)

        im = show_axis(im, r, t, camera_matrix, dist_coefs, 6,False)      


        blend = show_blend_mask(im, mask, 0.5)

        cv2.imshow('Estimated Pose', im)
        cv2.waitKey(0)  
        cv2.imshow('Estimated Pose', blend)
        cv2.waitKey(0)  


        
if __name__ == '__main__':
    subset_name = 'LND'

    data_path = './{}/TRAIN/'.format(subset_name)
    # data_path = './{}/sample_TEST/'.format(subset_name)
    skip_path = data_path+'config.yaml'
    print(skip_path)

    with open(skip_path) as f_tmp:
        config =  yaml.load(f_tmp, Loader=yaml.FullLoader)
    camera_matrix = np.array(config['cam']['camera_matrix']['data']).reshape((3,3))
    if config['cam']['dist_coeff'] is not None:
        dist_coefs = np.array(config['cam']['dist_coeff']['data'])
    else:
        dist_coefs = None

    vis(data_path,config['dataset'],camera_matrix,dist_coefs,project_3d=False)
