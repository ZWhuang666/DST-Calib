import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from utils import*
from config import*
import random
import warnings

from losses.chamfer_torch import (
    TorchChamferDistance,
    chamfer_loss,
)


def _load_chamfer_backend():
    cuda_available = torch.cuda.is_available()
    cuda_toolkit_available = False
    if cuda_available:
        try:
            from torch.utils.cpp_extension import CUDA_HOME

            cuda_toolkit_available = CUDA_HOME is not None
        except ImportError:
            pass

    print("CUDA available:", cuda_available)
    print("CUDA toolkit available:", cuda_toolkit_available)
    if cuda_available and cuda_toolkit_available:
        try:
            from losses.chamfer_distance import ChamferDistance

            print("CHAMFER CUDA EXTENSION COMPILE SUCCESS")
            return True, ChamferDistance
        except Exception as err:
            warnings.warn(
                "CUDA Chamfer Distance is unavailable; falling back to the "
                f"native PyTorch implementation. Original error: {err}",
                RuntimeWarning,
            )
    else:
        print(
            "CUDA Chamfer Distance prerequisites are unavailable; using the "
            "native PyTorch implementation."
        )

    return False, TorchChamferDistance


CHAMFER_C, ChamferDistance = _load_chamfer_backend()


def chamfer_distance_raw(template: torch.Tensor, source: torch.Tensor):
    cost_p0_p1, cost_p1_p0 = ChamferDistance()(template, source)
    return cost_p0_p1, cost_p1_p0


class Tmat_Loss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps  

    def forward(self, pred, target):
        assert pred.shape[1] == 6 and target.shape[1] == 6, "input data dim should be (B,6)"
        
        pred_rvec = pred[:, :3]
        pred_tvec = pred[:, 3:]

        pred_Tmat = pose.batch_rtTensor2TmatTensor(pred_rvec,pred_tvec)

        target_rvec = target[:, :3]
        target_tvec = target[:, 3:]

        target_Tmat = pose.batch_rtTensor2TmatTensor(target_rvec,target_tvec)
        
        I_Tmat = torch.eye(4).to(pred_Tmat)
        
        delta_Tmat = pred_Tmat.bmm(torch.linalg.inv(target_Tmat))
        error_Tmat = torch.mean(torch.abs(delta_Tmat - I_Tmat))
        
        return error_Tmat                 

class PointCloudDisLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input_data,output_data):
        out_pcd = output_data["est_pcd"]
        tar_pcd = input_data["calibed_pcd_batch"]
        out_pcd_transpose = out_pcd.transpose(1,2)  # B, N, 3
        tar_pcd_transpose = tar_pcd.transpose(1,2)  # B, N, 3
        dis_loss = torch.norm(torch.abs(out_pcd_transpose - tar_pcd_transpose),dim = 2)
        dis_mean = torch.mean(dis_loss)
        return dis_mean

class RotLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input_data, output_data):
        pred_rotvec = output_data["rvec"]
        target_rotvec = input_data["label_batch"][:,:3]
        
        pred_R =  pose.Rodrigues_r2R(pred_rotvec)
        target_R =  pose.Rodrigues_r2R(target_rotvec)
        
        I_R = torch.eye(3).to(pred_R)
        delta_R=pred_R.bmm(torch.linalg.inv(target_R))
        loss_rot = torch.mean(torch.abs(delta_R - I_R))
        
        return loss_rot                      
    
class TransLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    def forward(self,input_data,output_data):
        
        pred_trans = output_data["tvec"]
        target_trans = input_data["label_batch"][:, 3:6]
        
        trans_error = pred_trans - target_trans
        loss_trans = torch.mean(torch.norm(trans_error, dim = 1))

        return loss_trans

class LossZero(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()

    def forward(self,input_data,output_data):
        return torch.tensor(0)
    
class TransLoss_SelfSuper(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self,input_data,output_data):
        pred_trans = output_data["tvec"]
        ini_trans = torch.zeros(pred_trans.shape).to(pred_trans)

        trans_error = pred_trans - ini_trans
        loss_trans = torch.mean(torch.norm(trans_error, dim = 1))
        
        return loss_trans
    
def pcdTensor_randomDownsample(point_cloud, target_points):
    """ranom downsampling point clouds
    Args:
        point_cloud: (B, 3, N) 
        target_points: target number of downsampling points
    Returns:
        downsampled: (B, 3, target_points)
    """
    B, C, N = point_cloud.shape
    
    if target_points >= N:
        return point_cloud
    
    # select random index for each batch
    indices = torch.randperm(N, device=point_cloud.device)[:target_points]
    downsampled = point_cloud[:, :, indices]
    
    return downsampled

class ChamferDis_Loss_PE(nn.Module):
    def __init__(self, eps=1e-8, rand_scale = [1.0, 1.0]):
        '''Chamfer Loss that contain multiple depth layer
        '''
        super().__init__()
        self.eps = eps
        self.rand_scale = rand_scale

    def forward(self,input_data,output_data):
        
        # chamfer loss
        est_pcd = output_data["est_pcd"]
        tgt_pcd = input_data["depthCloudXYZ_trans_batch"]

        
        p0 = est_pcd
        p1 = tgt_pcd

        B = est_pcd.shape[0]

        total_chamfer_loss = torch.tensor(0)
        chamfer_loss_list = []
        for i in range(B):
            p0 = est_pcd[i]
            p1 = tgt_pcd[i]

            p1 = p1*random.uniform(self.rand_scale[0],self.rand_scale[1])     
            
            # valid filter
            p0_z_mask = (p0[2, :] > 3) & (p0[2, :] < 20) & (p0[1, :] > -10) & (p0[1, :] < 10)# (B, N)
            p1_z_mask = (p1[2, :] > 3) & (p1[2, :] < 20) & (p1[1, :] > -10) & (p1[1, :] < 10)# (B, N)

            p0 = p0[:,p0_z_mask].unsqueeze(0)
            p1 = p1[:,p1_z_mask].unsqueeze(0)

            p0 = pcdTensor_randomDownsample(p0, 10000)
            p1 = pcdTensor_randomDownsample(p1, 10000)

            if p0.shape[2] == 0 or p1.shape[2] == 0:
                diff_chamfer_loss = torch.tensor(0)
                chamfer_loss_list.append(diff_chamfer_loss)
                continue

            # Calculate Chamfer Distance
            # raw chamfer cost
            cost_p0_p1, cost_p1_p0 = chamfer_distance_raw(p0.transpose(1,2),p1.transpose(1,2))
            # p0 -> p1
            loss_p0_p1 = torch.mean(torch.sqrt(cost_p0_p1),dim=-1)
            # p1 -> p0
            loss_p1_p0 = torch.mean(torch.sqrt(cost_p1_p0),dim=-1)
            diff_chamfer_loss = loss_p0_p1*0.1 + loss_p1_p0*0.7
            chamfer_loss_list.append(diff_chamfer_loss)
            total_chamfer_loss = total_chamfer_loss + diff_chamfer_loss
            

        total_chamfer_loss = total_chamfer_loss / B
        sim_score = total_chamfer_loss
        
        return sim_score, chamfer_loss_list
    
class CalibrationErr_SelfSuper_EvaGuided(nn.Module):
    def __init__(self, eps=1e-8):
        '''Chamfer Loss that contain multiple depth layer
        '''
        super().__init__()
        self.eps = eps
    def forward(self,input_data,output_data):
        pred_rvec = output_data["rvec"]
        target_rvec = output_data["ss_sim_fin_xvec"][:,:3]
        pred_tvec = output_data["tvec"]
        target_tvec = output_data["ss_sim_fin_xvec"][:,3:]

        pred_R =  pose.Rodrigues_r2R(pred_rvec)
        target_R =  pose.Rodrigues_r2R(target_rvec)
        euler_pred = pose.RTensor2EulerYPRTensor(pred_R)
        euler_target = pose.RTensor2EulerYPRTensor(target_R)

        calib_er = torch.mean(torch.norm(euler_pred - euler_target, dim = 1), dim = 0)
        calib_et = torch.mean(torch.norm(pred_tvec - target_tvec, dim = 1), dim = 0)
        
        calib_err_magnitude = pose.CalPoseSim(calib_er,calib_et)
        
        return calib_err_magnitude


def getSimCalibList_fromSimCalibBatch(sim_calib_batch):
    '''
    get a [{"sim":float, "T": np.array},...] list from a {"sim":item(),"calib_xvec":torch batch}
    '''
    batch_sim = sim_calib_batch["sim"]
    batch_xvec = sim_calib_batch["calib_xvec"]
    batch_rvec = batch_xvec[:,:3]
    batch_tvec = batch_xvec[:,3:]
    batch_T = pose.batch_rtTensor2TmatTensor(batch_rvec,batch_tvec)

    sim_calib_list = []
    
    for i in range(len(batch_T)):
        sim_calib_list.append({"sim":batch_sim,"T":zv.tensor2Arr(batch_T[i])})

    return sim_calib_list

def getSimCalibList_fromSimCalibBatch_Indiv(sim_calib_batch):
    '''
    get a [{"sim":float, "T": np.array},...] list from a {"sim":item(),"calib_xvec":torch batch}
    '''
    batch_sim = sim_calib_batch["sim"]
    batch_xvec = sim_calib_batch["calib_xvec"]
    batch_rvec = batch_xvec[:,:3]
    batch_tvec = batch_xvec[:,3:]
    batch_T = pose.batch_rtTensor2TmatTensor(batch_rvec,batch_tvec)

    sim_calib_list = []
    
    for i in range(len(batch_T)):
        sim_calib_list.append({"sim":batch_sim[i].item(),"T":zv.tensor2Arr(batch_T[i])})

    return sim_calib_list


def MFOptimization(sim_calib_list,ratio = 0.2):
    '''Similarity Optimization
    '''
    sim_calib_list = sorted(sim_calib_list,key = lambda sim_err: sim_err["sim"])
    
    sel_num = int(len(sim_calib_list)*ratio)
    #print(sel_num)
    sim = 0
    rvec_list = []
    tvec_list = []
    sim_list = []

    sim_score_list = []
    total_sim_score = 0

    for i in range(sel_num):
        sim_calib = sim_calib_list[i]
        p_sim = sim_calib["sim"]
        sim_score = np.exp(-p_sim)
        sim_score_list.append(sim_score)
        total_sim_score += sim_score
        
        T = sim_calib["T"]
        rvec,tvec = pose.T2rt(T)
        rvec = rvec.reshape(-1)
        tvec = tvec.reshape(-1)

        sim += p_sim
        
        # add to total seq list
        rvec_list.append(rvec)
        tvec_list.append(tvec)
        sim_list.append(p_sim)

    for i in range(len(sim_score_list)):
        sim_score_list[i] = sim_score_list[i] / total_sim_score

    avg_rvec = pose.CalAverage_rvec(rvec_list,weights=sim_score_list)
    avg_R,_ = cv2.Rodrigues(avg_rvec)
    avg_tvec = pose.CalAverage_tvec(tvec_list,weights=sim_score_list)
    avg_T = pose.Rt2T(avg_R,avg_tvec)
    
    sim /= sel_num

    opt_res = {
        "opt_sim":sim,
        "opt_T":avg_T,
    }
    return opt_res



if __name__ == "__main__":
    pass
