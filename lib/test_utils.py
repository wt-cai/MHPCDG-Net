import torch
import numpy as np
from torchsparse import SparseTensor
from torchsparse.utils import sparse_collate_fn, sparse_quantize
from plyfile import PlyData, PlyElement
import os

def init_image_coor(height, width, u0=None, v0=None):
    u0 = width / 2.0 if u0 is None else u0
    v0 = height / 2.0 if v0 is None else v0

    x_row = np.arange(0, width)
    x = np.tile(x_row, (height, 1))
    x = x.astype(np.float32)
    u_u0 = x - u0

    y_col = np.arange(0, height)
    y = np.tile(y_col, (width, 1)).T
    y = y.astype(np.float32)
    v_v0 = y - v0
    return u_u0, v_v0

def depth_to_pcd(depth, u_u0, v_v0, f, invalid_value=0):
    mask_invalid = depth <= invalid_value
    depth[mask_invalid] = 0.0
    x = u_u0 / f * depth
    y = v_v0 / f * depth
    z = depth
    pcd = np.stack([x, y, z], axis=2)
    return pcd, ~mask_invalid

def pcd_to_sparsetensor(pcd, mask_valid, voxel_size=0.01, num_points=100000):
    pcd_valid = pcd[mask_valid]
    block_ = pcd_valid
    block = np.zeros_like(block_)
    block[:, :3] = block_[:, :3]

    pc_ = np.round(block_[:, :3] / voxel_size)
    pc_ -= pc_.min(0, keepdims=1)
    feat_ = block

    # transfer point cloud to voxels
    inds = sparse_quantize(pc_,
                           feat_,
                           return_index=True,
                           return_invs=False)
    if len(inds) > num_points:
        inds = np.random.choice(inds, num_points, replace=False)

    pc = pc_[inds]
    feat = feat_[inds]
    lidar = SparseTensor(feat, pc)
    feed_dict = [{'lidar': lidar}]
    inputs = sparse_collate_fn(feed_dict)
    return inputs

def pcd_uv_to_sparsetensor(pcd, u_u0, v_v0, mask_valid, f= 500.0, voxel_size=0.01, mask_side=None, num_points=100000):
    if mask_side is not None:
        mask_valid = mask_valid & mask_side
    pcd_valid = pcd[mask_valid]
    u_u0_valid = u_u0[mask_valid][:, np.newaxis] / f
    v_v0_valid = v_v0[mask_valid][:, np.newaxis] / f

    block_ = np.concatenate([pcd_valid, u_u0_valid, v_v0_valid], axis=1)
    block = np.zeros_like(block_)
    block[:, :] = block_[:, :]


    pc_ = np.round(block_[:, :3] / voxel_size)
    pc_ -= pc_.min(0, keepdims=1)
    feat_ = block

    # transfer point cloud to voxels
    inds = sparse_quantize(pc_,
                           feat_,
                           return_index=True,
                           return_invs=False)
    if len(inds) > num_points:
        inds = np.random.choice(inds, num_points, replace=False)

    pc = pc_[inds]
    feat = feat_[inds]
    lidar = SparseTensor(feat, pc)
    feed_dict = [{'lidar': lidar}]
    inputs = sparse_collate_fn(feed_dict)
    return inputs


def refine_focal_one_step(depth, focal, model, u0, v0):
    # reconstruct PCD from depth
    u_u0, v_v0 = init_image_coor(depth.shape[0], depth.shape[1], u0=u0, v0=v0)
    pcd, mask_valid = depth_to_pcd(depth, u_u0, v_v0, f=focal, invalid_value=0)
    # input for the voxelnet
    feed_dict = pcd_uv_to_sparsetensor(pcd, u_u0, v_v0, mask_valid, f=focal, voxel_size=0.005, mask_side=None)
    inputs = feed_dict['lidar'].cuda()

    outputs = model(inputs)
    return outputs

def refine_shift_one_step(depth_wshift, model, focal, u0, v0):
    # reconstruct PCD from depth
    u_u0, v_v0 = init_image_coor(depth_wshift.shape[0], depth_wshift.shape[1], u0=u0, v0=v0)
    pcd_wshift, mask_valid = depth_to_pcd(depth_wshift, u_u0, v_v0, f=focal, invalid_value=0)
    # input for the voxelnet
    feed_dict = pcd_to_sparsetensor(pcd_wshift, mask_valid, voxel_size=0.01)
    inputs = feed_dict['lidar'].cuda()

    outputs = model(inputs)
    return outputs

def refine_focal(depth, focal, model, u0, v0):
    last_scale = 1
    focal_tmp = np.copy(focal)
    for i in range(1):
        scale = refine_focal_one_step(depth, focal_tmp, model, u0, v0)
        focal_tmp = focal_tmp / scale.item()
        last_scale = last_scale * scale
    return torch.tensor([[last_scale]])

def refine_shift(depth_wshift, model, focal, u0, v0):
    depth_wshift_tmp = np.copy(depth_wshift)
    last_shift = 0
    for i in range(1):
        shift = refine_shift_one_step(depth_wshift_tmp, model, focal, u0, v0)
        shift = shift if shift.item() < 0.7 else torch.tensor([[0.7]])
        depth_wshift_tmp -= shift.item()
        last_shift += shift.item()
    return torch.tensor([[last_shift]])

def reconstruct_3D(depth, f):
    """
    Reconstruct depth to 3D pointcloud with the provided focal length.
    Return:
        pcd: N X 3 array, point cloud
    """
    cu = depth.shape[1] / 2
    cv = depth.shape[0] / 2
    width = depth.shape[1]
    height = depth.shape[0]
    row = np.arange(0, width, 1)
    u = np.array([row for i in np.arange(height)])
    col = np.arange(0, height, 1)
    v = np.array([col for i in np.arange(width)])
    v = v.transpose(1, 0)

    if f > 1e5:
        print('Infinit focal length!!!')
        x = u - cu
        y = v - cv
        z = depth / depth.max() * x.max()
    else:
        x = (u - cu) * depth / f
        y = (v - cv) * depth / f
        z = depth

    x = np.reshape(x, (width * height, 1)).astype(np.float)
    y = np.reshape(y, (width * height, 1)).astype(np.float)
    z = np.reshape(z, (width * height, 1)).astype(np.float)
    pcd = np.concatenate((x, y, z), axis=1)
    pcd = pcd.astype(np.int)
    return pcd

def save_HSI_point_cloud(pcd, rgb, hy_data,filename, binary=True):
    """Save an RGB point cloud as a PLY file.

    :paras
      @pcd: Nx3 matrix, the XYZ coordinates
      @rgb: NX3 matrix, the rgb colors for each 3D point
    """
    band1_3 = hy_data['cube'][0:3]  # 切片操作，深拷  [3,673,832]
    band1_3 = band1_3.transpose((2, 1, 0))  # 转置操作  [832,673,3]
    band1_3 = band1_3[:, :, ::-1]
    band1_3 = np.squeeze(band1_3)
    band1_3 = np.reshape(band1_3, (-1, 3))
    
        
    band4_6 = hy_data['cube'][3:6]  # 切片操作，深拷
    band4_6 = band4_6.transpose((2, 1, 0))  # 转置操作
    band4_6 = band4_6[:, :, ::-1]
    band4_6 = np.squeeze(band4_6)
    band4_6 = np.reshape(band4_6, (-1, 3))
        
    band7_9 = hy_data['cube'][6:9]  # 切片操作，深拷
    band7_9 = band7_9.transpose((2, 1, 0))  # 转置操作
    band7_9 = band7_9[:, :, ::-1]
    band7_9 = np.squeeze(band7_9)
    band7_9 = np.reshape(band7_9, (-1, 3))
        
    band10_12 = hy_data['cube'][9:12]  # 切片操作，深拷
    band10_12 = band10_12.transpose((2, 1, 0))  # 转置操作
    band10_12 = band10_12[:, :, ::-1]                         #不知道需不需要加这个
    band10_12 = np.squeeze(band10_12)
    band10_12 = np.reshape(band10_12, (-1, 3))
        
    band13_15 = hy_data['cube'][12:15]  # 切片操作，深拷
    band13_15 = band13_15.transpose((2, 1, 0))  # 转置操作
    band13_15 = band13_15[:, :, ::-1]
    band13_15 = np.squeeze(band13_15)
    band13_15 = np.reshape(band13_15, (-1, 3))
        
    band16_18 =hy_data['cube'][15:18]  # 切片操作，深拷
    band16_18 = band16_18.transpose((2, 1, 0))  # 转置操作
    band16_18 = band16_18[:, :, ::-1]
    band16_18 = np.squeeze(band16_18)
    band16_18 = np.reshape(band16_18, (-1, 3))
        
    band19_21= hy_data['cube'][18:21]  # 切片操作，深拷
    band19_21 = band19_21.transpose((2, 1, 0))  # 转置操作
    band19_21 = band19_21[:, :, ::-1]
    band19_21 = np.squeeze(band19_21)
    band19_21 = np.reshape(band19_21, (-1, 3))
        
    band22_24 = hy_data['cube'][21:24]  # 切片操作，深拷
    band22_24 = band22_24.transpose((2, 1, 0))  # 转置操作
    band22_24 = band22_24[:, :, ::-1]
    band22_24 = np.squeeze(band22_24)
    band22_24 = np.reshape(band22_24, (-1, 3))

    band25_27 = hy_data['cube'][24:27]  # 切片操作，深拷
    band25_27 = band25_27.transpose((2, 1, 0))  # 转置操作
    band25_27 = band25_27[:, :, ::-1]
    band25_27 = np.squeeze(band25_27)
    band25_27 = np.reshape(band25_27, (-1, 3))

    band28_30 = hy_data['cube'][27:30]  # 切片操作，深拷
    band28_30 = band28_30.transpose((2, 1, 0))  # 转置操作
    band28_30 = band28_30[:, :, ::-1]
    band28_30 = np.squeeze(band28_30)
    band28_30 = np.reshape(band28_30, (-1, 3))

    assert pcd.shape[0] == rgb.shape[0]
    if rgb is None:
        gray_concat = np.tile(np.array([128], dtype=np.uint8), (pcd.shape[0], 3))
        # points_3d = np.hstack((pcd, gray_concat,band1_3,band4_6,band7_9,band10_12,band13_15,band16_18,band19_21,band22_24,band25_27,band28_30))
        points_3d = np.concatenate((pcd, gray_concat,band1_3,band4_6,band7_9,band10_12,band13_15,band16_18,band19_21,band22_24,band25_27,band28_30),axis=-1)

    else:
        # points_3d = np.hstack((pcd, rgb,band1_3,band4_6,band7_9,band10_12,band13_15,band16_18,band19_21,band22_24,band25_27,band28_30))
        points_3d = np.concatenate((pcd, rgb,band1_3,band4_6,band7_9,band10_12,band13_15,band16_18,band19_21,band22_24,band25_27,band28_30),axis=-1)

        
    python_types = (float, float, float, int, int, int,float, float, float,float, float, float,float, \
        float, float,float, float, float,float, float, float,float, float, float,float, float, float,float, \
        float, float,float, float, float,float, float, float)
        
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),('blue', 'u1'),\
        ('band1', 'f4'), ('band2', 'f4'), ('band3', 'f4'),('band4', 'f4'), ('band5', 'f4'), ('band6', 'f4'),\
        ('band7', 'f4'), ('band8', 'f4'), ('band9', 'f4'),('band10', 'f4'), ('band11', 'f4'), ('band12', 'f4'),\
        ('band13', 'f4'), ('band14', 'f4'), ('band15', 'f4'),('band16', 'f4'), ('band17', 'f4'), ('band18', 'f4'),\
        ('band19', 'f4'), ('band20', 'f4'), ('band21', 'f4'),('band22', 'f4'), ('band23', 'f4'), ('band24', 'f4'),\
        ('band25', 'f4'), ('band26', 'f4'), ('band27', 'f4'),('band28', 'f4'), ('band29', 'f4'), ('band30', 'f4')]
    if binary is True:
        # Format into NumPy structured array
        vertices = []
        for row_idx in range(points_3d.shape[0]):
            cur_point = points_3d[row_idx]
            vertices.append(tuple(dtype(point) for dtype, point in zip(python_types, cur_point)))
        vertices_array = np.array(vertices, dtype=npy_types)
        el = PlyElement.describe(vertices_array, 'vertex')

        # Write
        PlyData([el]).write(filename)
    else:
        x = np.squeeze(points_3d[:, 0])
        y = np.squeeze(points_3d[:, 1])
        z = np.squeeze(points_3d[:, 2])
        r = np.squeeze(points_3d[:, 3])
        g = np.squeeze(points_3d[:, 4])
        b = np.squeeze(points_3d[:, 5])

        band1 = np.squeeze(points_3d[:, 6])
        band2 = np.squeeze(points_3d[:, 7])
        band3 = np.squeeze(points_3d[:, 8])
        band4 = np.squeeze(points_3d[:, 9])
        band5 = np.squeeze(points_3d[:, 10])
        band6 = np.squeeze(points_3d[:, 11])
        band7 = np.squeeze(points_3d[:, 12])
        band8 = np.squeeze(points_3d[:, 13])
        band9 = np.squeeze(points_3d[:, 14])
        band10 = np.squeeze(points_3d[:, 15])
        band11 = np.squeeze(points_3d[:, 16])
        band12 = np.squeeze(points_3d[:, 17])
        band13 = np.squeeze(points_3d[:, 18])
        band14 = np.squeeze(points_3d[:, 19])
        band15 = np.squeeze(points_3d[:, 20])
        band16 = np.squeeze(points_3d[:, 21])
        band17 = np.squeeze(points_3d[:, 22])
        band18 = np.squeeze(points_3d[:, 23])
        band19 = np.squeeze(points_3d[:, 24])
        band20 = np.squeeze(points_3d[:, 25])
        band21 = np.squeeze(points_3d[:, 26])
        band22 = np.squeeze(points_3d[:, 27])
        band23 = np.squeeze(points_3d[:, 28])
        band24 = np.squeeze(points_3d[:, 29])
        band25 = np.squeeze(points_3d[:, 30])
        band26 = np.squeeze(points_3d[:, 31])
        band27 = np.squeeze(points_3d[:, 32])
        band28 = np.squeeze(points_3d[:, 33])
        band29 = np.squeeze(points_3d[:, 34])
        band30 = np.squeeze(points_3d[:, 35])
        
        ply_head = 'ply\n' \
                'format ascii 1.0\n' \
                'element vertex %d\n' \
                'property float x\n' \
                'property float y\n' \
                'property float z\n' \
                'property uchar red\n' \
                'property uchar green\n' \
                'property uchar blue\n' \
                'property float band1\n' \
                'property float band2\n' \
                'property float band3\n' \
                'property float band4\n' \
                'property float band5\n' \
                'property float band6\n' \
                'property float band7\n' \
                'property float band8\n' \
                'property float band9\n' \
                'property float band10\n' \
                'property float band11\n' \
                'property float band12\n' \
                'property float band13\n' \
                'property float band14\n' \
                'property float band15\n' \
                'property float band16\n' \
                'property float band17\n' \
                'property float band18\n' \
                'property float band19\n' \
                'property float band20\n' \
                'property float bang21\n' \
                'property float band22\n' \
                'property float band23\n' \
                'property float band24\n' \
                'property float band25\n' \
                'property float band26\n' \
                'property float band27\n' \
                'property float band28\n' \
                'property float band29\n' \
                'property float band30\n' \
                'end_header' % r.shape[0]
        # ---- Save ply data to disk
        np.savetxt(filename, np.column_stack((x, y, z, r, g, b,band1, band2, band3, band4, \
            band5, band6,band7, band8, band9, band10, band11, band12,band13, band14, band15, \
            band16, band17, band18,band19, band20,band21, band22, band23, band24,band25, band26, \
            band27, band28, band29, band30,)), \
            fmt="%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d", \
            header=ply_head, comments='')


def save_RGB_point_cloud(pcd, rgb, filename, binary=True):
    assert pcd.shape[0] == rgb.shape[0]
    if rgb is None:
        gray_concat = np.tile(np.array([128], dtype=np.uint8), (pcd.shape[0], 3))
        points_3d = np.hstack((pcd, gray_concat))
    else:
        points_3d = np.hstack((pcd, rgb))
    python_types = (float, float, float, int, int, int)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                    ('blue', 'u1')]
    if binary is True:
        # Format into NumPy structured array
        vertices = []
        for row_idx in range(points_3d.shape[0]):
            cur_point = points_3d[row_idx]
            vertices.append(tuple(dtype(point) for dtype, point in zip(python_types, cur_point)))
        vertices_array = np.array(vertices, dtype=npy_types)
        el = PlyElement.describe(vertices_array, 'vertex')

        # Write
        PlyData([el]).write(filename)
    else:
        x = np.squeeze(points_3d[:, 0])
        y = np.squeeze(points_3d[:, 1])
        z = np.squeeze(points_3d[:, 2])
        r = np.squeeze(points_3d[:, 3])
        g = np.squeeze(points_3d[:, 4])
        b = np.squeeze(points_3d[:, 5])

        ply_head = 'ply\n' \
                'format ascii 1.0\n' \
                'element vertex %d\n' \
                'property float x\n' \
                'property float y\n' \
                'property float z\n' \
                'property uchar red\n' \
                'property uchar green\n' \
                'property uchar blue\n' \
                'end_header' % r.shape[0]
        # ---- Save ply data to disk
        np.savetxt(filename, np.column_stack((x, y, z, r, g, b)), fmt="%d %d %d %d %d %d", header=ply_head, comments='')

def reconstruct_depth(depth, rgb,hy_data, dir, pcd_name, focal):
    """
    para disp: disparity, [h, w]
    para rgb: rgb image, [h, w, 3], in rgb format
    """
    rgb = np.squeeze(rgb)
    depth = np.squeeze(depth)

    mask = depth < 1e-8
    depth[mask] = 0
    depth = depth / depth.max() * 10000

    pcd = reconstruct_3D(depth, f=focal)
    rgb_n = np.reshape(rgb, (-1, 3))
    save_RGB_point_cloud(pcd, rgb_n, os.path.join(dir, pcd_name + '_RGB.ply'))
    print("RGB_point_cloud")
    save_HSI_point_cloud(pcd, rgb_n, hy_data,os.path.join(dir, pcd_name + '_HSI.ply'))
    print("HSI_point_cloud")


def recover_metric_depth(pred, gt):
    if type(pred).__module__ == torch.__name__:
        pred = pred.cpu().numpy()
    if type(gt).__module__ == torch.__name__:
        gt = gt.cpu().numpy()
    gt = gt.squeeze()
    pred = pred.squeeze()
    mask = (gt > 1e-8) & (pred > 1e-8)

    gt_mask = gt[mask]
    pred_mask = pred[mask]
    a, b = np.polyfit(pred_mask, gt_mask, deg=1)
    pred_metric = a * pred + b
    return pred_metric
