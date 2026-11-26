
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import sys
import os
import math
import json
from scipy.spatial.transform import Rotation
sys.path.append(os.path.dirname(__file__))
from .sinc import sinc1, sinc2, sinc3
from . import zutils as zv
from . import so3
from . import se3
from scipy.linalg import eig
from typing import Union, List, Optional


def getIntrinsicExtrinsicFromZVCalibJson(calib_json_path):
    with open(calib_json_path, 'r') as calib_json_file:
        calib_json = json.load(calib_json_file)
        intrinsic = None
        extrinsic = None
        calib_params_dict = None
        if "calCamera" in calib_json:
            calib_params_dict = calib_json["calCamera"]
        elif "cloudCamera" in calib_json:
            calib_params_dict = calib_json["cloudCamera"]
        else:
            pass

        assert calib_params_dict != None, "calib_params_dict must exist, please check the json file"
        
        # load instrinsic and extrinsic matrix from json file
        assert "Intrinsic" in calib_params_dict, "calib_params_dict must have Intrinsic key, check the json file"
        intrinsic = calib_params_dict["Intrinsic"]

        if "Extrinsic" in calib_params_dict:
            extrinsic = calib_params_dict["Extrinsic"]

        K = np.array(intrinsic["K"])
        if extrinsic != None:
            rvec = np.array(extrinsic["R"]).flatten()
            R_lidar2cam, _ = cv2.Rodrigues(rvec)
            t_lidar2cam = np.array(extrinsic["T"]).flatten()
            T_lidar2cam = Rt2T(R_lidar2cam,t_lidar2cam)
        else:
            T_lidar2cam = np.eye(4)
            R_lidar2cam, t_lidar2cam = T2Rt(T_lidar2cam)
        
        return K,T_lidar2cam,R_lidar2cam,t_lidar2cam


def set_T_initialGuess(r_ini, t_ini, R_base = zv.R_LVINI_NP):
    R_ini_input = EulerYPR2R(r_ini)
    R_ini = np.matmul(R_ini_input,R_base)
    t_ini_afterR = np.matmul(R_ini_input,np.array(t_ini))
    
    return Rt2T(R_ini, t_ini_afterR)
    
def Rt2XVec(R, t):
    Rmat = np.array(R)
    tvec = np.array(t)
    rvec = cv2.Rodrigues(Rmat)[0].flatten()
    print(tvec,rvec)
    xvec = np.concatenate((rvec, tvec))
    return xvec

def Rt2XVec_split(R, t):
    Rmat = np.array(R)
    tvec = np.array(t)
    rvec = cv2.Rodrigues(Rmat)[0].flatten()
    #xvec = np.concatenate((rvec, tvec))
    
    return rvec,tvec

def Rt2rt(R, t):
    Rmat = np.array(R)
    tvec = np.array(t)
    rvec = cv2.Rodrigues(Rmat)[0].flatten()
    #xvec = np.concatenate((rvec, tvec))
    return rvec,tvec

def Rt2T(R,t):
    Rmat = np.array(R)
    tvec = np.array(t)
    T = np.eye(4)
    T[:3,:3] = Rmat
    T[:3,3] = tvec
    return T

def rt2T(r,t):
    Rmat = cv2.Rodrigues(r)[0]
    tvec = np.array(t)
    T = np.eye(4)
    T[:3,:3] = Rmat
    T[:3,3] = tvec
    return T

def T2Rt(T):
    R = T[:3,:3]
    t = T[:3,3]
    return R,t

def T2rt(T):
    R = T[:3,:3]
    t = T[:3,3]
    r,_ = cv2.Rodrigues(R)
    return r,np.copy(t)

def T2xvec(T):
    rvec,tvec = T2rt(T)
    xvec = np.concatenate((rvec.flatten(), tvec))
    return xvec

def Rodrigues_R2r(R: torch.Tensor):
    """Convert rotation matrix to axis-angle representation (rvec)
    Args:
        R (torch.Tensor): (*, 3, 3) rotation matrix
    Returns:
        torch.Tensor: (*, 3) rotation vector
    """
    # Ensure R is a valid rotation matrix (optional, for debugging)
    assert R.shape[-2:] == (3, 3), "Input must be a batch of 3x3 matrices"
    R_w = R.view(-1,3,3)
    
    # Compute the trace of R
    trace_R = R_w.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
    theta = torch.acos(torch.clamp((trace_R - 1) / 2, -1, 1))  # Clamp to avoid NaN

    # Compute the rotation axis using skew-symmetric part
    W_hat = (R_w - R_w.transpose(-1, -2)) / (2 * torch.sin(theta).unsqueeze(-1).unsqueeze(-1))
    
    # Extract rotation vector components
    rvec = torch.stack([W_hat[..., 2, 1], W_hat[..., 0, 2], W_hat[..., 1, 0]], dim=-1)

    # Handle theta == 0 case (no rotation)
    rvec = torch.where(theta.unsqueeze(-1) == 0, torch.zeros_like(rvec), rvec * theta.unsqueeze(-1))

    return rvec.view(*(R.size()[0:-2]), 3)

def Rodrigues_r2R(r:torch.Tensor):
    """transform rvec to R using Rodrigues formula
    Args:
        x (torch.Tensor): *,3
    Returns:
        tuple: depth_img (*,3,3), transformed_pcd (B,3,N)
    """
    assert r.shape[-1:][0] == 3, "Input must be a batch of 3 vectors"
    w = r.view(-1, 3)
    t = w.norm(p=2, dim=1).view(-1, 1, 1)
    W = so3.mat(w)
    S = W.bmm(W)
    I = torch.eye(3).to(w)
    
    # Rodrigues' rotation formula.
    #R = cos(t)*eye(3) + sinc1(t)*W + sinc2(t)*(w*w');
    #  = eye(3) + sinc1(t)*W + sinc2(t)*S
    R = I + sinc1(t)*W + sinc2(t)*S

    return R.view(*(r.size()[0:-1]), 3, 3)

    
def Rodrigues(rot:torch.Tensor):
    '''
    Rodrigues Transform for Torch Tensor
    '''
    ele_num = rot.numel()
    assert ele_num == 9 or ele_num == 3,  "element number in rot should be 3 or 9"
    
    if (ele_num == 9):
        return Rodrigues_R2r(rot)
    elif (ele_num == 3):
        return Rodrigues_r2R(rot)
    else:
        assert False,  "element number in rot should be 3 or 9"
    
    
def TmatTensor2RtTensor(Tmat_tensor):
    R_tensor = Tmat_tensor[:3,:3]
    t_tensor = Tmat_tensor[:3,3]
    return R_tensor.clone(),t_tensor.clone()
    

def TmatTensor2rtTensor(Tmat_tensor):
    R_tensor = Tmat_tensor[:3,:3]
    rvec_tensor = Rodrigues_R2r(R_tensor)
    t_tensor = Tmat_tensor[:3,3]
    return rvec_tensor.clone(),t_tensor.clone()

def rtTensor2TmatTensor(r_tensor,t_tensor):
    R_tensor = Rodrigues_r2R(r_tensor)
    Tmat_tensor = torch.eye(4).to(R_tensor)
    Tmat_tensor[:3,:3] = R_tensor
    Tmat_tensor[:3,3] = t_tensor
    return Tmat_tensor


def batch_TmatTensor2rtTensor(Tmat_tensor):
    Tmat_w = Tmat_tensor.contiguous().view(-1,4,4)
    Rmat = Tmat_w[:,:3,:3]
    rvec = Rodrigues_R2r(Rmat)
    tvec = Tmat_w[:,:3,3]
    
    return rvec.clone().view(*(Tmat_tensor.size()[0:-2]),3), tvec.clone().view(*(Tmat_tensor.size()[0:-2]),3)

def batch_TmatTensor2RtTensor(Tmat_tensor):
    Tmat_w = Tmat_tensor.contiguous().view(-1,4,4)
    Rmat = Tmat_w[:,:3,:3]
    tvec = Tmat_w[:,:3,3]
    
    return Rmat.clone().view(*(Tmat_tensor.size()[0:-2]),3), tvec.clone().view(*(Tmat_tensor.size()[0:-2]),3)

def batch_Tmat2Rt(Tmat_arr:np.ndarray):
    Tmat_w = Tmat_arr.reshape(-1,4,4)
    Rmat = Tmat_w[:,:3,:3]
    tvec = Tmat_w[:,:3,3]
    
    return Rmat.copy(),tvec.copy()

def batch_TmatTensor2XVec6Tensor(Tmat_tensor):
    Tmat_w = Tmat_tensor.contiguous().view(-1,4,4)
    Rmat = Tmat_w[:,:3,:3]
    rvec = Rodrigues_R2r(Rmat)
    tvec = Tmat_w[:,:3,3]
    rvec = rvec.clone().view(*(Tmat_tensor.size()[0:-2]),3)
    tvec = tvec.clone().view(*(Tmat_tensor.size()[0:-2]),3)
    
    return torch.cat([rvec,tvec],dim = 1)

def batch_rtTensor2TmatTensor(r_tensor,t_tensor):
    r_w = r_tensor.contiguous().view(-1,3)
    t_w = t_tensor.contiguous().view(-1,3)
    b = r_w.shape[0]
    Rmat = Rodrigues_r2R(r_w)
    
    I_4 = torch.eye(4).to(Rmat)
    Tmat = torch.stack([I_4] * b, dim=0)
    
    Tmat[:,:3,:3] = Rmat
    Tmat[:,:3,3] = t_w
    
    return Tmat.view(*(r_tensor.size()[0:-1]),4,4)

def batch_RtTensor2TmatTensor(R_tensor,t_tensor):
    R_w = R_tensor.contiguous().view(-1,3,3)
    t_w = t_tensor.contiguous().view(-1,3)
    b = R_w.shape[0]
    
    I_4 = torch.eye(4).to(R_w)
    Tmat = torch.stack([I_4] * b, dim=0)
    
    Tmat[:,:3,:3] = R_w
    Tmat[:,:3,3] = t_w
    
    return Tmat.view(*(R_tensor.size()[0:-2]),4,4)


def XVec2rt(xvec):
    rvec_normalized = xvec[:3]
    rvec_scale = xvec[3]
    tvec_normalized = xvec[4:7]
    tvec_scale = xvec[7]

    rvec = rvec_normalized*rvec_scale
    R, _ = cv2.Rodrigues(rvec)
    tvec = np.dot(R,tvec_normalized*tvec_scale)
    return rvec,tvec

def XVecTensor2rt(xvec_tensor):
    xvec = xvec_tensor.cpu().detach().numpy()
    rvec_normalized = xvec[:3]
    rvec_scale = xvec[3]
    tvec_normalized = xvec[4:7]
    tvec_scale = xvec[7]

    rvec = rvec_normalized*rvec_scale
    R, _ = cv2.Rodrigues(rvec)
    tvec = np.dot(R,tvec_normalized*tvec_scale)
    return rvec,tvec

def XVecTensor2Rt(xvec_tensor):
    '''
    Convert an XVec tensor to  R and t numpy array
    '''
    xvec = xvec_tensor.cpu().detach().numpy()
    rvec_normalized = xvec[:3]
    rvec_scale = xvec[3]
    tvec_normalized = xvec[4:7]
    tvec_scale = xvec[7]

    rvec = rvec_normalized*rvec_scale
    R, _ = cv2.Rodrigues(rvec)
    tvec = np.dot(R,tvec_normalized*tvec_scale)
    return R,tvec

def XVecTensor2RtTensor(xvec_tensor):
    '''
    Convert an XVec tensor to  R and t numpy array
    '''
    xvec = xvec_tensor.cpu().detach().numpy()
    rvec_normalized = xvec[:3]
    rvec_scale = xvec[3]
    tvec_normalized = xvec_tensor[4:7]
    tvec_scale = xvec_tensor[7]

    rvec = rvec_normalized*rvec_scale
    R, _ = cv2.Rodrigues(rvec)
    R_tensor = torch.tensor(R,dtype=torch.float32,requires_grad=True).to(xvec_tensor.device)
    tvec_tensor = torch.matmul(R_tensor,tvec_normalized*tvec_scale).to(xvec_tensor.device)
    return R_tensor,tvec_tensor

def XVec2Rt(xvec):
    '''
    Convert an XVec tensor to  R and t numpy array
    '''
    rvec_normalized = xvec[:3]
    rvec_scale = xvec[3]
    tvec_normalized = xvec[4:7]
    tvec_scale = xvec[7]

    rvec = rvec_normalized*rvec_scale
    R, _ = cv2.Rodrigues(rvec)
    tvec = np.dot(R,tvec_normalized*tvec_scale)
    return R,tvec

def Rt2XVec(R,t):
    '''
    Convert Rt numpy array to XVec numpy array
    '''
    t_coor = np.dot(np.linalg.inv(R),t)
    rvec,_ = Rt2rt(R, t_coor)
    
    vnorm_r = zv.vnorm(rvec)
    vnorm_t = zv.vnorm(t_coor)
    
    if vnorm_r == 0:
        rvec_normalized = np.array([0.0,0.0,0.0])
    else:
        rvec_normalized = (rvec)/(vnorm_r)
        
    if vnorm_t == 0:
        t_coor_normalized = np.array([0.0,0.0,0.0])
    else:
        t_coor_normalized = (t_coor)/(vnorm_t)
    
    xvec = np.concatenate((rvec_normalized,[vnorm_r],t_coor_normalized,[vnorm_t]))
    return xvec

def Rt2XVecTensor(R,t):
    '''
    Convert Rt numpy array to XVec tensor
    '''
    t_coor = np.dot(np.linalg.inv(R),t)
    rvec,_ = Rt2rt(R, t_coor)
    
    vnorm_r = zv.vnorm(rvec)
    vnorm_t = zv.vnorm(t_coor)
    
    if vnorm_r == 0:
        rvec_normalized = np.array([0.0,0.0,0.0])
    else:
        rvec_normalized = (rvec)/(vnorm_r)
        
    if vnorm_t == 0:
        t_coor_normalized = np.array([0.0,0.0,0.0])
    else:
        t_coor_normalized = (t_coor)/(vnorm_t)
    
    xvec = np.concatenate((rvec_normalized,[vnorm_r],t_coor_normalized,[vnorm_t]))
    
    #torch.tesnor([1],requires_grad=True)
    return torch.from_numpy(xvec)

def rt2XVecTensor(rvec,tvec):
    '''
    Convert rt numpy array to XVec tensor
    '''
    R, _ = cv2.Rodrigues(rvec)
    t_coor = np.dot(np.linalg.inv(R),tvec)
    
    vnorm_r = zv.vnorm(rvec)
    vnorm_t = zv.vnorm(t_coor)
    
    if vnorm_r == 0:
        rvec_normalized = np.array([0.0,0.0,0.0])
    else:
        rvec_normalized = (rvec)/(vnorm_r)
        
    if vnorm_t == 0:
        t_coor_normalized = np.array([0.0,0.0,0.0])
    else:
        t_coor_normalized = (t_coor)/(vnorm_t)
    
    xvec = np.concatenate((rvec_normalized,[vnorm_r],t_coor_normalized,[vnorm_t]))
    
    #torch.tesnor([1],requires_grad=True)
    return torch.from_numpy(xvec)

def rt2XVec(rvec,tvec):
    '''
    Convert rt numpy array to XVec
    '''
    R, _ = cv2.Rodrigues(rvec)
    t_coor = np.dot(np.linalg.inv(R),tvec)
    
    vnorm_r = zv.vnorm(rvec)
    vnorm_t = zv.vnorm(t_coor)
    
    if vnorm_r == 0:
        rvec_normalized = np.array([0.0,0.0,0.0])
    else:
        rvec_normalized = (rvec)/(vnorm_r)
        
    if vnorm_t == 0:
        t_coor_normalized = np.array([0.0,0.0,0.0])
    else:
        t_coor_normalized = (t_coor)/(vnorm_t)
    
    xvec = np.concatenate((rvec_normalized,[vnorm_r],t_coor_normalized,[vnorm_t]))
    
    return xvec

def SafeRotationFromMatrix(mat,fallback="identity"):
    if not np.all(np.isfinite(mat)):
        print("infinite number in rmat")
        if fallback == "identity":
            return Rotation.identity()
        elif isinstance(fallback, np.ndarray) and fallback.shape == (3, 3):
            return Rotation.from_matrix(fallback)
        else:
            raise ValueError("Invalid fallback rotation")
        
    U, _, Vt = np.linalg.svd(mat)
    mat = U @ Vt
    if np.linalg.det(mat) < 0:
        print("np.linalg.det(mat) < 0, convert to right hand coordinate system")
        U[:, -1] *= -1
        mat = U @ Vt
    
    return Rotation.from_matrix(mat)

def R2EulerYPR(R:np.ndarray):
    """gt transformer to euler anlges and translation
    Args:
        gt (np.ndarray): 4x4

    Returns:
        angle_gt, trans_gt: (3,1),(3,1)
    """
    
    rotation = SafeRotationFromMatrix(R)
    anglez_gt, angley_gt, anglex_gt = rotation.as_euler('zyx')
    euler_angle = np.array([anglez_gt, angley_gt, anglex_gt])
    euler_angle = np.degrees(euler_angle)
    return euler_angle


def EulerYPR2R(ypr_vec):
    """convert Euler angle (yaw，pitch，roll) to rotation matrix
    Args:
        ypr_vec : (yaw，pitch，roll) deg
    Returns:
        np.ndarray: 3x3
    """
    # X
    roll = ypr_vec[2]/180*np.pi
    pitch = ypr_vec[1]/180*np.pi
    yaw= ypr_vec[0]/180*np.pi
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Y
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Z
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # combination, Z Y X
    R = np.dot(R_x, np.dot(R_y, R_z))
    
    return R


def EulerYPRTensor2RTensor(ypr_vec: torch.tensor) -> torch.tensor:
    """
    Yaw,Pitch,Roll (degree) to Rotation matrix

    Args:
        ypr_vec: torch.tensor, (3,)

    Returns:
        R: torch.tensor, (3,3)
    """
    
    is_batch = ypr_vec.dim() > 1
    if not is_batch:
        ypr_vec = ypr_vec.unsqueeze(0)

    B = ypr_vec.size()[0]
    
    
    R_x = torch.zeros((B,3,3),dtype=torch.float32)
    R_y = torch.zeros((B,3,3),dtype=torch.float32)
    R_z = torch.zeros((B,3,3),dtype=torch.float32)
    yaw_rad = torch.deg2rad(ypr_vec[:, 0])
    pitch_rad = torch.deg2rad(ypr_vec[:, 1])
    roll_rad = torch.deg2rad(ypr_vec[:, 2])
    
    for i in range(B):
        R_x[i, 0, 0] = 1
        R_x[i, 1, 1] = torch.cos(roll_rad)[i]
        R_x[i, 1, 2] = -torch.sin(roll_rad)[i]
        R_x[i, 2, 1] = torch.sin(roll_rad)[i]
        R_x[i, 2, 2] = torch.cos(roll_rad)[i]
        
        R_y[i, 0, 0] = torch.cos(pitch_rad)[i]
        R_y[i, 0, 2] = torch.sin(pitch_rad)[i]
        R_y[i, 1, 1] = 1
        R_y[i, 2, 0] = -torch.sin(pitch_rad)[i]
        R_y[i, 2, 2] = torch.cos(pitch_rad)[i]
        
        
        R_z[i, 0, 0] = torch.cos(yaw_rad)[i]
        R_z[i, 0, 1] = -torch.sin(yaw_rad)[i]
        R_z[i, 1, 0] = torch.sin(yaw_rad)[i]
        R_z[i, 1, 1] = torch.cos(yaw_rad)[i]
        R_z[i, 2, 2] = 1
        
    R = torch.bmm(torch.bmm(R_x, R_y), R_z)
    
    if not is_batch:
        R = R.squeeze(0)

    return R

def RTensor2EulerYPRTensor(R: torch.tensor) -> torch.tensor:
    """convert Rotation matrix to Euler angle (yaw, pitch, roll)
    Args:
        R: torch.tensor, (B, 3, 3) / (3, 3)
    Returns:
        ypr_vec: torch.tensor, (B, 3) / (3)
    """
    
    is_batch = R.dim() > 2
    if not is_batch:
        R = R.unsqueeze(0)  
    
    
    r00, r01, r02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    r10, r11, r12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    r20, r21, r22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    
    pitch = torch.asin(r02)  
    
    gimbal_lock = torch.abs(torch.abs(pitch) - math.pi / 2) < 1e-6
    yaw = torch.where(gimbal_lock, torch.atan2(r10, r11), torch.atan2(-r01, r00))
    roll = torch.where(gimbal_lock, torch.tensor(0.0, dtype=torch.float32), torch.atan2(-r12, r22))

    
    yaw_deg = torch.rad2deg(yaw)
    pitch_deg = torch.rad2deg(pitch)
    roll_deg = torch.rad2deg(roll)
    
    # combination
    ypr_vec = torch.stack([yaw_deg, pitch_deg, roll_deg], dim=1)
    
    if not is_batch:
        ypr_vec.squeeze(0)

    return ypr_vec


def GenerateRandom_R_FromYPR(r_rand_range):
    roll_min, roll_max = -r_rand_range[2], r_rand_range[2]
    pitch_min, pitch_max = -r_rand_range[1], r_rand_range[1]
    yaw_min, yaw_max = -r_rand_range[0], r_rand_range[0]
    
    euler_ang = np.array([
        np.random.uniform(yaw_min, yaw_max),
        np.random.uniform(pitch_min, pitch_max),
        np.random.uniform(roll_min, roll_max)
    ])
    
    Rmat = EulerYPR2R(euler_ang)
    return Rmat

def GenerateRandom_t(t_rand_range):
    x_min, x_max = -t_rand_range[0], t_rand_range[0]
    y_min, y_max = -t_rand_range[1], t_rand_range[1]
    z_min, z_max = -t_rand_range[2], t_rand_range[2]
    tvector = np.array([
        np.random.uniform(x_min, x_max),
        np.random.uniform(y_min, y_max),
        np.random.uniform(z_min, z_max)
    ])
    
    return tvector 

def GenerateRandomT_byCenters(T_centers:np.ndarray,
                              t_disturb_range = np.array([0.01,0.01,0.01]),
                              euler_disturb_range = np.array([0.1,0.1,0.1])
                              ) -> list: 
    '''Generate Random T matrix using given T mat centers with perturbation range
    ### Args
        generate_num:
        T_centers: B,4,4
    ### Return
        Generated T matrix [list]: B,4,4
    '''
    
    disturb_R = GenerateRandom_R_FromYPR(euler_disturb_range)
    disturb_t = GenerateRandom_t(np.array(t_disturb_range))
    disturb_Tmat = Rt2T(disturb_R,disturb_t)
    generate_Tmat_arr = np.matmul(disturb_Tmat,T_centers)
    
    return generate_Tmat_arr


def GenerateRandomT_byCenters_sameBias(T_centers:np.ndarray,
                              t_disturb_range = np.array([0.01,0.01,0.01]),
                              euler_disturb_range = np.array([0.1,0.1,0.1])
                              ) -> list: 
    '''Generate Random T matrix using given T mat centers with minor interuption
    ### Args
        generate_num:
        T_centers: B,4,4
    ### Return
        Generated T matrix [list]: B,4,4
    '''
    
    disturb_R = GenerateRandom_R_FromYPR(euler_disturb_range)
    disturb_t = GenerateRandom_t(np.array(t_disturb_range))
    disturb_Tmat = Rt2T(disturb_R,disturb_t)
    generate_Tmat_arr = np.matmul(disturb_Tmat,T_centers)
    
    return generate_Tmat_arr

def GenerateRandomT_byCenters_diffBias(T_centers: np.ndarray,
                            t_disturb_range: np.ndarray = np.array([0.01, 0.01, 0.01]),
                            euler_disturb_range: np.ndarray = np.array([0.1, 0.1, 0.1])) -> np.ndarray:
    '''Generate Random T matrix using given T mat centers with minor interruption, 
       where each T matrix in the batch is disturbed differently.
    
    ### Args
        T_centers: Batch of T matrices with shape (B,4,4)
        t_disturb_range: Translation disturbance range for x,y,z (3,)
        euler_disturb_range: Euler angle disturbance range for yaw,pitch,roll (3,)
    
    ### Return
        Generated T matrices with shape (B,4,4)
    '''
    batch_size = T_centers.shape[0]
    generate_Tmat_arr = np.zeros_like(T_centers)
    
    for i in range(batch_size):
        # Generate unique disturbance for each T matrix in the batch
        disturb_R = GenerateRandom_R_FromYPR(euler_disturb_range)
        disturb_t = GenerateRandom_t(t_disturb_range)
        disturb_Tmat = Rt2T(disturb_R, disturb_t)
        
        # Apply disturbance to the current T matrix
        generate_Tmat_arr[i] = np.matmul(disturb_Tmat, T_centers[i])
    
    return generate_Tmat_arr


def CalPoseErr(pose_Rmat,pose_tvec,gt_Rmat,gt_tvec,print_err = True):
    
    pose_ypr = R2EulerYPR(pose_Rmat)
    gt_ypr = R2EulerYPR(gt_Rmat)

    error_dict = {
        "ypr":np.zeros(3,dtype=np.float64),
        "xyz":np.zeros(3,dtype=np.float64),
        "er":-1,
        "et":-1,
        }
    
    error_dict["ypr"] = np.abs(pose_ypr - gt_ypr)
    error_dict["xyz"] = np.abs(pose_tvec - gt_tvec)
    error_dict["er"] = np.linalg.norm(pose_ypr - gt_ypr)
    error_dict["et"] = np.linalg.norm(pose_tvec - gt_tvec)

    #if error_dict["er"] > 180:
        #print("Error Euler Angle")
    
    if print_err:
        print(" er: ",error_dict["er"], end="")
        print(" et: ",error_dict["et"], end="")

    return error_dict

def CalPoseErr_T(T_pred,T_gt, print_err = False):
    R_pred,t_pred = T2Rt(T_pred)
    R_gt, t_gt = T2Rt(T_gt)

    return CalPoseErr(R_pred, t_pred, R_gt, t_gt, print_err = print_err)

def CalPoseErr_From_TRel(T_rel):
    gt_R = np.eye(3)
    gt_t = np.zeros(3)
    R_rel,t_rel = T2Rt(T_rel)

    return CalPoseErr(R_rel,t_rel,gt_R,gt_t,print_err = False)

def CalPoseErrBatch(model_output, label, print_err = True):
    batch_size = model_output.shape[0]
    total_er = 0
    total_et = 0
    total_ypr = np.array([0,0,0],dtype=np.float32)
    total_xyz = np.array([0,0,0],dtype=np.float32)
    
    for i in range(batch_size):
        rvec_pred = model_output[i][:3].cpu().detach().numpy()
        tvec_pred = model_output[i][3:6].cpu().detach().numpy()
        rvec_gt = label[i][:3].cpu().detach().numpy()
        tvec_gt = label[i][3:6].cpu().detach().numpy()
        
        Rmat_pred, _ = cv2.Rodrigues(rvec_pred)
        Rmat_gt,_ = cv2.Rodrigues(rvec_gt)
        #err_dict = CalPoseErr(Rmat_pred,tvec_pred,Rmat_gt,tvec_gt,print_err=False)
        err_dict = CalPoseErr(Rmat_pred,tvec_pred,Rmat_gt,tvec_gt,print_err=False)
        er = err_dict["er"]
        et = err_dict["et"]
        ypr = err_dict["ypr"]
        xyz = err_dict["xyz"]
        total_xyz += xyz
        total_ypr += ypr
        total_er += er
        total_et += et
            
    avg_er = total_er/batch_size
    avg_et = total_et/batch_size
    avg_ypr = total_ypr/batch_size
    avg_xyz = total_xyz/batch_size
    return avg_er, avg_et, avg_ypr, avg_xyz

def CalPoseSim(er,et):
    sim = (0.1*er + et)
    return sim

def CalPoseSimBatch_tensor(pred_xvec:torch.Tensor,gt_xvec:torch.Tensor,print_err=False):
    pred_rvec = pred_xvec[:,:3]
    target_rvec = gt_xvec[:,:3]

    pred_tvec = pred_xvec[:,3:6]
    target_tvec = gt_xvec[:,3:6]

    # convert to euler
    pred_R = Rodrigues_r2R(pred_rvec)
    target_R = Rodrigues_r2R(target_rvec)
    euler_pred = RTensor2EulerYPRTensor(pred_R)
    euler_target = RTensor2EulerYPRTensor(target_R)

    calib_er  = torch.mean(torch.norm(euler_pred - euler_target,dim = 1),dim = 0)
    calib_et = torch.mean(torch.norm(pred_tvec-target_tvec,dim = 1),dim = 0)
    
    calib_err_magnitude = CalPoseSim(calib_er,calib_et)

    return calib_err_magnitude

def CalPoseSimBatch_tensor_Indiv(pred_xvec:torch.Tensor,gt_xvec:torch.Tensor,print_err=False):
    pred_rvec = pred_xvec[:,:3]
    target_rvec = gt_xvec[:,:3]

    pred_tvec = pred_xvec[:,3:6]
    target_tvec = gt_xvec[:,3:6]

    # convert to euler
    pred_R = Rodrigues_r2R(pred_rvec)
    target_R = Rodrigues_r2R(target_rvec)
    euler_pred = RTensor2EulerYPRTensor(pred_R)
    euler_target = RTensor2EulerYPRTensor(target_R)
    
    calib_er = torch.norm(euler_pred - euler_target,dim = 1)
    calib_et = torch.norm(pred_tvec - target_tvec,dim = 1)
    
    calib_err_magnitude = CalPoseSim(calib_er,calib_et)

    return calib_err_magnitude

def r2EulerYPR(r:np.ndarray):
    Rmat, _ = cv2.Rodrigues(r)
    return R2EulerYPR(Rmat)

def rTensor2EulerYPRTensor(r:torch.Tensor):
    Rmat = Rodrigues(r)
    return RTensor2EulerYPRTensor(Rmat)


def CalAverage_tvec(trans_vecs: Union[List, np.ndarray], 
                               weights: Optional[Union[List, np.ndarray]] = None,
                               return_std: bool = False) -> Union[np.ndarray, tuple]:
    """Calculate the average of a given series of translation (weighted or not weighted)
    Args:
        trans_vecs: list of the translation vector, (N, 3) or (N, 2)
        weights: the optional list of the weights, (N,)
        return_std: if returning the std information
    
    Returns:
        return_std=False: translation average [x, y, z] or [x, y]
        return_std=True: (translation average, x std , y std, z std, std distance)
    """
    
    
    if not trans_vecs:
        raise ValueError("translation list should not be empty")
    
    trans_vecs = np.array(trans_vecs, dtype=np.float64)
    
    
    if trans_vecs.ndim != 2:
        raise ValueError("translation vectors should not be 2D array")
    
    n_vectors, n_dims = trans_vecs.shape
    if n_dims not in [2, 3]:
        raise ValueError("translation vectors should be (2,) or (3,)")
    
    # deal weights
    if weights is None:
        weights = np.ones(n_vectors)
    else:
        weights = np.array(weights, dtype=np.float64)
        if weights.shape != (n_vectors,):
            raise ValueError("weights shape does not match the number of translation vectors")
        if np.any(weights < 0):
            raise ValueError("the weights value cannot be negative")
    
    # normalized weights
    weights = weights / np.sum(weights)
    
    # calculate weighted sum
    weighted_sum = np.sum(trans_vecs * weights[:, np.newaxis], axis=0)
    mean_vector = weighted_sum
    
    if return_std:
        std_x = np.sqrt(np.average((trans_vecs[:, 0] - mean_vector[0])**2, weights=weights))
        std_y = np.sqrt(np.average((trans_vecs[:, 1] - mean_vector[1])**2, weights=weights))
        
        if n_dims == 3:
            std_z = np.sqrt(np.average((trans_vecs[:, 2] - mean_vector[2])**2, weights=weights))
        else:
            std_z = 0.0
        
        distances = np.linalg.norm(trans_vecs - mean_vector, axis=1)
        mean_distance = np.average(distances, weights=weights)
        std_distance = np.sqrt(np.average((distances - mean_distance)**2, weights=weights))
        
        return mean_vector, std_x, std_y, std_z, std_distance
    
    return mean_vector

def CalAverage_rvec(rot_vecs, weights=None):
    """Calculate the average of a given series of rotation (weighted or not weighted)
    
    Args:
        rot_vecs: list or numpy array
        weights: list or numpy array, optional
    
    Returns:
        numpy array
    """
    
    
    if not rot_vecs:
        raise ValueError("rotation vector list should not be empty")
    
    rot_vecs = np.array(rot_vecs)
    if rot_vecs.ndim != 2 or rot_vecs.shape[1] != 3:
        raise ValueError("shape of rotation vectors should be (N, 3)")
    
    n = len(rot_vecs)
    
    # deal weights
    if weights is None:
        weights = np.ones(n)
    else:
        weights = np.array(weights)
        if len(weights) != n:
            raise ValueError("number of weights should equal to the number of rotation vectors")
    
    # normalize weights
    weights = weights / np.sum(weights)
    
    
    quats = []
    for i, v in enumerate(rot_vecs):
        theta = np.linalg.norm(v)
        if theta < 1e-10:
            # zero rotation
            q = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            u = v / theta
            half_theta = 0.5 * theta
            w_val = np.cos(half_theta)
            xyz = np.sin(half_theta) * u
            q = np.array([w_val, xyz[0], xyz[1], xyz[2]])
        
        # ensure consistency
        if q[0] < 0:
            q = -q
        
        # apply the weights
        weighted_q = q * np.sqrt(weights[i])
        quats.append(weighted_q)
    
    Q = np.array(quats).T
    
    C = Q @ Q.T
    
    eigenvalues, eigenvectors = eig(C)
    idx = np.argmax(np.real(eigenvalues))
    mean_quat = np.real(eigenvectors[:, idx])
    
    mean_quat = mean_quat / np.linalg.norm(mean_quat)
    
    if mean_quat[0] < 0:
        mean_quat = -mean_quat
    
    w, x, y, z = mean_quat
    angle = 2 * np.arccos(np.clip(w, -1.0, 1.0))
    
    if np.abs(angle) < 1e-10:
        mean_rotvec = np.zeros(3)
    else:
        sin_half_angle = np.sin(angle / 2)
        if np.abs(sin_half_angle) < 1e-10:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = np.array([x, y, z]) / sin_half_angle
        mean_rotvec = angle * axis
    
    return mean_rotvec
    

if __name__ == "__main__":
    pass    