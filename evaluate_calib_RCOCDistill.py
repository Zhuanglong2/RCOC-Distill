# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------

# Modified Author: Xudong Lv
# based on github.com/cattaneod/CMRNet/blob/master/main_visibility_CALIB.py

import cv2
import math
import os
import random
import time

# import apex
import mathutils
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn as nn

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from DatasetRadarLidarCamera import DatasetLidarCameraKittiOdometry
from losses import DistancePoints3D, GeometricLoss, L1Loss, ProposedLoss, CombinedLoss

from models.RCOCDistill import TestNet


from quaternion_distances import quaternion_distance

from tensorboardX import SummaryWriter
from utils import (mat2xyzrpy, merge_inputs, overlay_imgs, quat2mat,
                   quaternion_from_matrix, rotate_back, rotate_forward,
                   tvector2mat)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

ex = Experiment("RCOC-Distill")
ex.captured_out_filter = apply_backspaces_and_linefeeds

# noinspection PyUnusedLocal
@ex.config
def config():
    checkpoints = '/home/long/PycharmProjects/RCOC-Distill/checkpoint'
    dataset = 'NTU4DRadLM'  # NTU4DRadLM     Dual_Radar
    data_folder =  '/home/long/Datasets/NTU4DRadLM-Calib' #NTU4DRadLM-Calib   Dual_Radar-Calib
    use_reflectance = False
    val_sequence = 0     # NTU4DRadLM: 6 7 8    Dual_Radar: 0
    val_sequence = f"{val_sequence:02d}"
    show = False
    save_image = False
    BASE_LEARNING_RATE = 3e-4  # 1e-4  3e-4
    loss = 'combined'
    max_t = 0.2
    max_r = 10.0
    iterative_method = 'single'
    output = '/home/long/PycharmProjects/RCOC-Distill/output'
    output = os.path.join(output, iterative_method)
    num_worker = 3   # 6
    network = 'Res_f1'
    optimizer = 'adam'
    resume = False   # True
    weights = '/home/long/PycharmProjects/RCOC-Distill/DualRadar-RCOCDistill/val_seq_00/pointnet/checkpoint_r10.00_t0.20_e690_0.010.tar'
    rescale_rot = 1.0
    rescale_transl = 2.0  # 2
    weight_point_cloud = 0.5
    dropout = 0.0
    max_depth = 80.
    out_fig_lg = 'EN'


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


EPOCH = 1
def _init_fn(worker_id, seed):
    seed = seed + worker_id + EPOCH * 100
    print(f"Init worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_2D_lidar_projection(pcl, cam_intrinsic):
    pcl_xyz = cam_intrinsic @ pcl.T
    pcl_xyz = pcl_xyz.T
    pcl_z = pcl_xyz[:, 2]
    pcl_xyz = pcl_xyz / (pcl_xyz[:, 2, None] + 1e-10)
    pcl_uv = pcl_xyz[:, :2]

    return pcl_uv, pcl_z


def lidar_project_depth(pc_rotated, cam_calib, img_shape):
    pc_rotated = pc_rotated[:3, :].detach().cpu().numpy()
    cam_intrinsic = cam_calib.numpy()
    pcl_uv, pcl_z = get_2D_lidar_projection(pc_rotated.T, cam_intrinsic)
    mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (pcl_uv[:, 1] > 0) & (
            pcl_uv[:, 1] < img_shape[0]) & (pcl_z > 0)
    pcl_uv = pcl_uv[mask]
    pcl_z = pcl_z[mask]
    pcl_uv = pcl_uv.astype(np.uint32)
    pcl_z = pcl_z.reshape(-1, 1)
    depth_img = np.zeros((img_shape[0], img_shape[1], 1))
    depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z
    depth_img = torch.from_numpy(depth_img.astype(np.float32))
    depth_img = depth_img.cuda()
    depth_img = depth_img.permute(2, 0, 1)

    return depth_img, pcl_uv

# CNN test
@ex.capture
def val(model, rgb_img, refl_img, refl_img2, lidar_input_gt,target_transl, target_rot, loss_fn, point_clouds, loss):
    model.eval()

    # Run model
    with torch.no_grad():
        transl_err, rot_err, features2, features2_distillation = model(rgb_img, refl_img, refl_img2, test=False)

    if loss == 'points_distance' or loss == 'combined':
        losses = loss_fn(point_clouds, target_transl, target_rot, transl_err, rot_err, features2, features2_distillation, lidar_input_gt)
    else:
        losses = loss_fn(target_transl, target_rot, transl_err, rot_err)

    total_trasl_error = torch.tensor(0.0).to(target_rot.device)
    total_rot_error = quaternion_distance(target_rot, rot_err, target_rot.device)
    total_rot_error = total_rot_error * 180. / math.pi
    for j in range(rgb_img.shape[0]):
        total_trasl_error += torch.norm(target_transl[j] - transl_err[j]) * 100.

    return losses, total_trasl_error.item(), total_rot_error.sum().item(), rot_err, transl_err,features2, features2_distillation


@ex.automain
def main(_config, _run, seed):
    global EPOCH
    print('Loss Function Choice: {}'.format(_config['loss']))

    if _config['val_sequence'] is None:
        raise TypeError('val_sequences cannot be None')
    else:
        # _config['val_sequence'] = f"{_config['val_sequence']:02d}"
        print("Val Sequence: ", _config['val_sequence'])
        dataset_class = DatasetLidarCameraKittiOdometry

    if _config['dataset'] == 'NTU4DRadLM':
        img_shape = (480, 640)  # NTU
        input_size = (256, 384)
    elif _config['dataset'] == 'Dual_Radar':
        img_shape = (600, 960)  # dual radar
        input_size = (256, 384)
    else:
        raise TypeError('dataset is wrong!')

    show = _config['show']
    rgb_path = os.path.join(_config['output'], 'rgb')
    if not os.path.exists(rgb_path):
        os.makedirs(rgb_path)
    depth_path = os.path.join(_config['output'], 'depth')
    if not os.path.exists(depth_path):
        os.makedirs(depth_path)

    true_depth_path = os.path.join(_config['output'], 'true_depth')
    if not os.path.exists(true_depth_path):
        os.makedirs(true_depth_path)

    input_path = os.path.join(_config['output'], 'input')
    if not os.path.exists(input_path):
        os.makedirs(input_path)

    gt_path = os.path.join(_config['output'], 'gt')
    if not os.path.exists(gt_path):
        os.makedirs(gt_path)

    if _config['out_fig_lg'] == 'EN':
        results_path = os.path.join(_config['output'], 'results_en')
    elif _config['out_fig_lg'] == 'CN':
        results_path = os.path.join(_config['output'], 'results_cn')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    pred_path = os.path.join(_config['output'], 'pred')

    if not os.path.exists(os.path.join(pred_path, 'iteration_'+str(1))):
        os.makedirs(os.path.join(pred_path, 'iteration_'+str(1)))

    # save pointcloud to the output path
    pc_input_path = os.path.join(_config['output'], 'pointcloud', 'input')
    if not os.path.exists(pc_input_path):
        os.makedirs(pc_input_path)
    pc_pred_path = os.path.join(_config['output'], 'pointcloud', 'pred')
    if not os.path.exists(pc_pred_path):
        os.makedirs(pc_pred_path)

    dataset_val = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'], dataset=_config['dataset'],
                                split='val', use_reflectance=_config['use_reflectance'],
                                val_sequence=_config['val_sequence'])
    #######################
    # dataset_val = dataset_train
    #######################
    model_savepath = os.path.join(_config['checkpoints'], 'val_seq_' + _config['val_sequence'], 'pointnet')
    if not os.path.exists(model_savepath):
        os.makedirs(model_savepath)
    log_savepath = os.path.join(_config['checkpoints'], 'val_seq_' + _config['val_sequence'], 'log')
    if not os.path.exists(log_savepath):
        os.makedirs(log_savepath)


    np.random.seed(seed)
    torch.random.manual_seed(seed)

    def init_fn(x): return _init_fn(x, seed)

    val_dataset_size = len(dataset_val)
    print('Number of the val dataset: {}'.format(val_dataset_size))

    # Training and validation set creation
    num_worker = _config['num_worker']

    ValImgLoader = torch.utils.data.DataLoader(dataset=dataset_val,
                                                shuffle=False,
                                                batch_size=8,
                                                num_workers=num_worker,
                                                worker_init_fn=init_fn,
                                                collate_fn=merge_inputs,
                                                drop_last=False,
                                                pin_memory=True)



    # loss function choice
    if _config['loss'] == 'simple':
        loss_fn = ProposedLoss(_config['rescale_transl'], _config['rescale_rot'])
    elif _config['loss'] == 'geometric':
        loss_fn = GeometricLoss()
        loss_fn = loss_fn.cuda()
    elif _config['loss'] == 'points_distance':
        loss_fn = DistancePoints3D()
    elif _config['loss'] == 'L1':
        loss_fn = L1Loss(_config['rescale_transl'], _config['rescale_rot'])
    elif _config['loss'] == 'combined':
        loss_fn = CombinedLoss(_config['rescale_transl'], _config['rescale_rot'], _config['weight_point_cloud'])
    else:
        raise ValueError("Unknown Loss Function")

    # network choice and settings
    if _config['network'].startswith('Res'):
        feat = 1
        md = 4
        split = _config['network'].split('_')
        for item in split[1:]:
            if item.startswith('f'):
                feat = int(item[-1])
            elif item.startswith('md'):
                md = int(item[2:])
        assert 0 < feat < 7, "Feature Number from PWC have to be between 1 and 6"
        assert 0 < md, "md must be positive"

        model = TestNet(input_size, use_feat_from=feat, md=md,
                       use_reflectance=_config['use_reflectance'], dropout=_config['dropout'],
                       Action_Func='leakyrelu', attention=False, res_num=18)

    else:
        raise TypeError("Network unknown")
    if _config['weights'] is not None:
        print(f"Loading weights from {_config['weights']}")
        checkpoint = torch.load(_config['weights'], map_location='cpu')
        saved_state_dict = checkpoint['state_dict']
        model.load_state_dict(saved_state_dict)

    # model = model.to(device)
    model = nn.DataParallel(model)
    model = model.cuda()

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    if _config['loss'] == 'geometric':
        parameters += list(loss_fn.parameters())
    if _config['optimizer'] == 'adam':
        optimizer = optim.Adam(parameters, lr=_config['BASE_LEARNING_RATE'], weight_decay=5e-6)
        # Probably this scheduler is not used
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 70], gamma=0.5)
    else:
        optimizer = optim.SGD(parameters, lr=_config['BASE_LEARNING_RATE'], momentum=0.9,
                              weight_decay=5e-6, nesterov=True)


    if _config['weights'] is not None and _config['resume']:
        checkpoint = torch.load(_config['weights'], map_location='cpu')
        opt_state_dict = checkpoint['optimizer']
        optimizer.load_state_dict(opt_state_dict)

    val_iter = 0
    ## Validation ##
    total_val_loss = 0.
    total_val_t = 0.
    total_val_r = 0.

    local_loss = 0.0
    for batch_idx, sample in enumerate(ValImgLoader):
        #print(f'batch {batch_idx+1}/{len(TrainImgLoader)}', end='\r')
        start_time = time.time()
        lidar_input = []
        lidar_input2 = []
        rgb_input = []
        lidar_gt = []
        lidar_gt2 = []
        shape_pad_input = []
        real_shape_input = []
        pc_rotated_input = []

        # gt pose
        sample['tr_error'] = sample['tr_error'].cuda()
        sample['rot_error'] = sample['rot_error'].cuda()

        for idx in range(len(sample['rgb'])):
            # ProjectPointCloud in RT-pose
            real_shape = [sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2], sample['rgb'][idx].shape[0]]

            sample['point_cloud'][idx] = sample['point_cloud'][idx].cuda()
            pc_lidar = sample['point_cloud'][idx].clone()
            sample['point_cloud2'][idx] = sample['point_cloud2'][idx].cuda()

            pc_rotated2 = sample['point_cloud2'][idx]
            if _config['max_depth'] < 80.:
                pc_lidar = pc_lidar[:, pc_lidar[0, :] < _config['max_depth']].clone()
                pc_lidar2 = pc_lidar2[:, pc_lidar2[0, :] < _config['max_depth']].clone()

            depth_gt, uv = lidar_project_depth(pc_lidar, sample['calib'][idx], real_shape) # image_shape
            depth_gt /= _config['max_depth']
            depth_gt2, uv2  = lidar_project_depth(pc_rotated2, sample['calib2'][idx], real_shape)  # image_shape
            depth_gt2 /= _config['max_depth']

            R = mathutils.Quaternion(sample['rot_error'][idx]).to_matrix()
            R.resize_4x4()
            T = mathutils.Matrix.Translation(sample['tr_error'][idx])
            RT = torch.mm(torch.from_numpy(np.asarray(T)), torch.from_numpy(np.asarray(R))).to(sample['point_cloud'][idx].device).float()
            # RT = T * R

            pc_rotated = rotate_back(sample['point_cloud'][idx], RT)  # Pc` = RT * Pc
            pc_rotated2 = rotate_back(sample['point_cloud2'][idx], RT)  # Pc` = RT * Pc

            if _config['max_depth'] < 80.:
                pc_rotated = pc_rotated[:, pc_rotated[0, :] < _config['max_depth']].clone()
                pc_rotated2 = pc_rotated2[:, pc_rotated2[0, :] < _config['max_depth']].clone()

            depth_img, uv = lidar_project_depth(pc_rotated, sample['calib'][idx], real_shape) # image_shape
            depth_img /= _config['max_depth']
            depth_img2, uv2 = lidar_project_depth(pc_rotated2, sample['calib2'][idx], real_shape)  # image_shape
            depth_img2 /= _config['max_depth']

            # PAD ONLY ON RIGHT AND BOTTOM SIDE
            rgb = sample['rgb'][idx].cuda()
            shape_pad = [0, 0, 0, 0]

            shape_pad[3] = (img_shape[0] - rgb.shape[1])  # // 2
            shape_pad[1] = (img_shape[1] - rgb.shape[2])  # // 2 + 1

            rgb = F.pad(rgb, shape_pad)
            depth_img = F.pad(depth_img, shape_pad)
            depth_img2 = F.pad(depth_img2, shape_pad)
            depth_gt = F.pad(depth_gt, shape_pad)
            depth_gt2 = F.pad(depth_gt2, shape_pad)

            rgb_input.append(rgb)
            lidar_input.append(depth_img)
            lidar_input2.append(depth_img2)
            lidar_gt.append(depth_gt)
            lidar_gt2.append(depth_gt2)
            real_shape_input.append(real_shape)
            shape_pad_input.append(shape_pad)
            pc_rotated_input.append(pc_rotated)

        lidar_input_gt = torch.stack(lidar_gt)
        lidar_input = torch.stack(lidar_input)
        lidar_input2 = torch.stack(lidar_input2)
        rgb_input = torch.stack(rgb_input)
        rgb_resize = F.interpolate(rgb_input, size=input_size, mode="bilinear", align_corners=True)
        lidar_resize = F.interpolate(lidar_input, size=input_size, mode="bilinear", align_corners=True)
        lidar_resize2 = F.interpolate(lidar_input2, size=input_size, mode="bilinear", align_corners=True)
        lidar_input_gt = F.interpolate(lidar_input_gt, size=input_size, mode="bilinear", align_corners=True)

        if _config['save_image']:
            for i in range(rgb_input.shape[0]):
                out0 = overlay_imgs(rgb_input[i], lidar_input[i].unsqueeze(0))
                out0 = out0[:img_shape[0], :img_shape[1], :]
                cv2.imwrite(os.path.join(input_path, sample['rgb_name'][i]) + '.png', out0[:, :, [2, 1, 0]] * 255)

                out1 = overlay_imgs(rgb_input[i], lidar_gt[i].unsqueeze(0))
                out1 = out1[:img_shape[0], :img_shape[1], :]
                cv2.imwrite(os.path.join(gt_path, sample['rgb_name'][i])+'.png', out1[:, :, [2, 1, 0]]*255)

                depth_img = lidar_input[i].detach().cpu().numpy()
                depth_img = (depth_img / np.max(depth_img)) * 255
                cv2.imwrite(os.path.join(depth_path, sample['rgb_name'][i])+'.png', depth_img[0, :img_shape[0], :img_shape[1]])

                depth_gt = lidar_gt[i].detach().cpu().numpy()
                depth_gt = (depth_gt / np.max(depth_gt)) * 255
                cv2.imwrite(os.path.join(true_depth_path, sample['rgb_name'][i])+'.png', depth_gt[0, :img_shape[0], :img_shape[1]])

        if show:
            out0 = overlay_imgs(rgb_input[0], lidar_input[0].unsqueeze(0))
            out1 = overlay_imgs(rgb_input[0], lidar_gt[0].unsqueeze(0))
            out2 = overlay_imgs(rgb_input[0], lidar_gt2[0].unsqueeze(0))
            cv2.imshow("INPUT", out0[:, :, [2, 1, 0]])
            cv2.imshow("GT", out1[:, :, [2, 1, 0]])
            cv2.imshow("GT-lidar", out2[:, :, [2, 1, 0]])
            cv2.waitKey(1)

        loss, trasl_e, rot_e, R_predicted,  T_predicted, features2, features2_distillation = val(model, rgb_resize, lidar_resize, lidar_resize2, lidar_input_gt,
                                                              sample['tr_error'], sample['rot_error'],
                                                              loss_fn, sample['point_cloud'], _config['loss'])
        total_val_t += trasl_e
        total_val_r += rot_e
        local_loss += loss['total_loss'].item()

        if _config['save_image']:
            for i in range(rgb_input.shape[0]):
                R_predict = quat2mat(R_predicted[i])
                T_predict = tvector2mat(T_predicted[i])
                RT_predict = torch.mm(T_predict, R_predict)

                rotated_point_cloud = pc_rotated_input[i]
                rotated_point_cloud = rotate_forward(rotated_point_cloud, RT_predict)
                depth_img_pred, uv_pred = lidar_project_depth(rotated_point_cloud, sample['calib'][i], real_shape)
                depth_img_pred /= _config['max_depth']
                depth_pred = F.pad(depth_img_pred, shape_pad_input[i])
                lidar = depth_pred.unsqueeze(0)

                out2 = overlay_imgs(rgb_input[i], lidar)
                out2 = out2[:img_shape[0], :img_shape[1], :]
                cv2.imwrite(os.path.join(os.path.join(pred_path, 'iteration_' + str(1)), sample['rgb_name'][i]+'.png'), out2[:, :, [2, 1, 0]] * 255)

        val_iter += 1

    print("------------------------------------")
    print('total val loss = %.3f' % (total_val_loss / len(dataset_val)))
    print(f'total traslation error: {total_val_t / len(dataset_val)} cm')
    print(f'total rotation error: {total_val_r / len(dataset_val)} °')
    print("------------------------------------")
