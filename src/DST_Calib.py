#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.utils as vutils
from common.comFunction import *
from ZModule import*
from dataset_loader import*
import loss as loss_utils


def setDSTCalibDatasetMode():
    lidar_fov_K, lidar_fov_size, dcloud_K, dcloud_size, remap_K, remap_size = setRemapKAndSize(
        remap_proj_f=600,
        remap_proj_size=(256, 512),
        size_add_bias=(64,600),
    )
    ZV_Dataset._LIDAR_FOV_PROJ_K = lidar_fov_K
    ZV_Dataset._LIDAR_FOV_PROJ_SIZE = lidar_fov_size
    ZV_Dataset._DCLOUD_PROJ_K = dcloud_K
    ZV_Dataset._DCLOUD_PROJ_SIZE = dcloud_size
    ZV_Dataset._REMAP_PROJ_K = remap_K
    ZV_Dataset._REMAP_SIZE = remap_size
    ZV_Dataset._LIP_DENSIFY_METHOD = "max"
    ZV_Dataset._LIP_DENSIFY = True
    ZV_Dataset._DENSE_KSIZE_VALUE = (9, 9)
    ZV_Dataset._DCP_CLOUD_REGION_CUT = True
    ZV_Dataset._DCP_CLOUD_RANDOM_CUT = True
    ZV_Dataset._CLOUD_DOWN_SAMPLE = False
    ZV_Dataset._DEPTH_SCALE = True
    ZV_Dataset._RGB_TENSOR_TYPE = "dcp"
    ZV_Dataset._GENERATE_DEPTHCLOUD = True


def resetWeights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

class EvaModule_SB(nn.Module):
    def __init__(self, grid_size=5,
                 dropout_rate=0.3, dropout_rate2=0.05):
        super().__init__()
        self.grid_size = grid_size

        # dropout rate
        self.dropout_rate = dropout_rate
        self.dropout_rate2 = dropout_rate2

        # single branch feature extraction using the input of difference map
        self.fea_branch = nn.Sequential(
            *list(models.resnet18(weights=models.ResNet18_Weights.DEFAULT).children())[:-2], 
            CBAM(512)
        ) 
        
        # Block Processor 
        self.block_processor = nn.Sequential(
            nn.Conv2d(512, 288, 3, padding=1),
            LayerNorm2d(288),  # using LayerNorm to replace BatchNorm
            nn.ReLU(),
            CBAM(288),

            nn.Conv2d(288, 144, 3, padding=1),
            LayerNorm2d(144),  # using LayerNorm to replace BatchNorm
            nn.ReLU(),
            CBAM(144),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        fc_input_dim = self.grid_size**2 * 144
        
        # Pose Regression
        self.rotation_head = nn.Sequential(
            nn.Linear(fc_input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            ResidualBlockFC(1024, 576, dropout_rate2),
            nn.Linear(576, 3)
        )

        self.translation_head = nn.Sequential(
            nn.Linear(fc_input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            ResidualBlockFC(1024, 576, dropout_rate2),
            nn.Linear(576, 3)
        ) 

        self._initialize_weights()
    
    def _initialize_weights(self):
        '''
        Initialize Network Weights
        '''
        nn.init.xavier_normal_(self.rotation_head[-1].weight, 0.02)
        nn.init.xavier_normal_(self.translation_head[-1].weight, 0.02)
        
        
    def forward(self, data):
        # generate difference map
        diff_map_batch = data["diff_map_batch"]

        # Load feature extraction and block processor
        fea_branch = self.fea_branch
        block_processor = self.block_processor
        
        # Step 1: Obtain the feature map from difference map
        fml = zv.cropToDivisible(fea_branch(diff_map_batch),self.grid_size)
        batch_size, channels, h, w = fml.size()
        gh, gw = h // self.grid_size, w // self.grid_size

        # Step 2: reshapes into blocks
        fml_reshaped = fml.view(batch_size, channels,
                                self.grid_size, gh,
                                self.grid_size, gw)
        blocks = fml_reshaped.permute(0, 2, 4, 1, 3, 5).contiguous()  # [B, gx, gy, C, gh, gw]
        blocks = blocks.view(-1, channels, gh, gw)  # [B * gx * gy, C, gh, gw]

        # Step 3: process the blocks
        processed_blocks = block_processor(blocks)  # [B * gx * gy, D] or [B * gx * gy, C, gh, gw]
        processed_blocks = processed_blocks.view(batch_size, self.grid_size * self.grid_size, -1)  # [B, N, D]
        merged = processed_blocks.view(batch_size, -1)  # [B, N * D]

        # Step 4: regress the relative pose (which can then be used to calculate the evaluation score)
        rvec = self.rotation_head(merged)
        tvec = self.translation_head(merged)
        xvec = torch.cat((rvec,tvec),dim = 1)
        
        return {
            "xvec":xvec,
            "rvec":rvec,
            "tvec":tvec,
        }

class PoseEstimator(nn.Module):
    def __init__(self,dropout_rate=0.0, dropout_rate2=0.0):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.dropout_rate2 = dropout_rate2

        # -------------- pose estimation ---------------- #
        self.fc_input_dim = 4028

        # rotation and translation head
        self.rotation_head = nn.Sequential(
            nn.Linear(self.fc_input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate / 2),
            ResidualBlockFC(1024, 576, self.dropout_rate2 / 4),
            nn.Linear(576, 3)
        )

        self.translation_head = nn.Sequential(
            nn.Linear(self.fc_input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate / 2),
            ResidualBlockFC(1024, 576, self.dropout_rate2 / 4),
            nn.Linear(576, 3)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        '''
        Initialize Network Weights
        '''
        
        nn.init.xavier_normal_(self.rotation_head[-1].weight, 0.00)
        nn.init.zeros_(self.rotation_head[-1].bias)
    
        nn.init.xavier_normal_(self.translation_head[-1].weight, 0.00)
        nn.init.zeros_(self.translation_head[-1].bias)

    def forward(self,data):
        B = data["lip_batch"].shape[0]

        x = torch.zeros((B,self.fc_input_dim), device=data["lip_batch"].device)
        rvec = self.rotation_head(x)
        tvec = self.translation_head(x)
        
        # Prediction heads
        return {
            "rvec":rvec,
            "tvec":tvec,
        }


class DST_Calib(nn.Module):
    def __init__(self,grid_size=5,
                 dropout_rate=0.3, dropout_rate2=0.05):
        super().__init__()
        
        # config the evaluation module (SB)
        self.sim_estimator = EvaModule_SB(grid_size=grid_size, 
                                        dropout_rate=dropout_rate, dropout_rate2=dropout_rate2)
        
        # configure the pose estimator (PE)
        self.pose_estimator = PoseEstimator(0,0)

        # intialize the model weights
        self._initialize_weights()

        if hasBatchNorm(self):
            self.disabled_batchnorm_runstat_count = disableBatchNormRunningStats(self)
        else:
            self.disabled_batchnorm_runstat_count = 0
    
    def _initialize_weights(self):
        '''
        Initialize Network Weights
        '''
        pass
        
    def reset_PoseEstimator(self):
        self.pose_estimator = PoseEstimator(0,0)
        self.pose_estimator.to(next(self.parameters()).device)
        self.pose_estimator.train()

    def _generateDiffMap(self,data,mode = "eva"):
        if mode == "eva":
            lidar_proj_batch = data["lip_batch"]
        else:
            lidar_proj_batch = data["est_lip_batch"]

        cam_proj_batch = data["depthCloudXYZ_proj_batch"]

        diff_map_batch = generateDiffMap_torch_batch(lidar_proj_batch,cam_proj_batch,diff_thres=[1.0,1.0])
        
        diff_map_batch = zv.tensorNormalizedMaxMin_perBatch(diff_map_batch,min_val=-1.0,max_val = 1.0)

        data["diff_map_batch"] = diff_map_batch
        
        return data
    
    def forward(self,data,mode = "eva"):
        '''
        Args:
            mode: "eva" | "calib"
        '''
        # input
        lip_batch = data["lip_batch"]
        B = lip_batch.shape[0]
        T_lidar2vir_batch = data["T_lidar2vir_batch"] # Initial extrinsic estimate
        K_batch = data["K_batch"]
        pcd_batch = data["pcd_batch"]

        proj_shape = (lip_batch.shape[2],lip_batch.shape[3])

        # initial estimation
        est_T = T_lidar2vir_batch.clone().to(pcd_batch)

        if mode == "eva":
            # evaluation
            self.sim_estimator.requires_grad_(True)
            data = self._generateDiffMap(data,mode = mode)
            sim_est_output = self.sim_estimator(data)
            rvec = sim_est_output["rvec"]
            tvec = sim_est_output["tvec"]
            xvec = sim_est_output["xvec"]

            # update result
            iter_est_T = pose.batch_rtTensor2TmatTensor(rvec,tvec)
            est_T = iter_est_T.bmm(est_T)
            est_lip_batch,est_pcd = transform.batch_projectLiDARDepth(K_batch,est_T,
                                    pcd_batch,proj_shape = proj_shape,proj_channel=3)
            
            data["est_pcd"] = est_pcd         
            data["est_lip_batch"] = est_lip_batch

            return {
                "xvec":xvec,
                "rvec":rvec,
                "tvec":tvec,
                "est_T":est_T,
                "iter_est_T":iter_est_T,
                "est_lip_batch":est_lip_batch,
                "est_pcd":est_pcd,
            }
        
        elif mode == "calib":
            # estimate pose
            data = self._generateDiffMap(data,mode = "eva")
            
            with torch.no_grad():
                sim_calib_output = self.sim_estimator(data)
                sim_calib_xvec = sim_calib_output["xvec"]
            
            # online self-supervised calibraiton via pose estimator
            pe_output = self.pose_estimator(data)
            rvec = pe_output["rvec"]
            tvec = pe_output["tvec"]
            
            # (EVA Sim)
            sim_sim, sim_sim_nxt_xvec, sim_sim_fin_xvec,sim_sim_nxt_output,sim_est_pcd,sim_est_lip_batch = self.CalSim_byGivenRTVEC(data,sim_calib_output["rvec"],sim_calib_output["tvec"],est_T)
            
            # (PE Sim)
            ss_sim, ss_sim_nxt_xvec,ss_sim_fin_xvec,ss_sim_nxt_output,ss_est_pcd,ss_est_lip_batch = self.CalSim_byGivenRTVEC(data,rvec,tvec,est_T)

            # updata estimated T
            iter_est_T = pose.batch_rtTensor2TmatTensor(rvec,tvec)
            est_T = iter_est_T.bmm(est_T)
            xvec = torch.cat((rvec,tvec),dim = 1)

            return {
                "xvec":xvec,
                "rvec":rvec,
                "tvec":tvec,
                "iter_est_T":iter_est_T,
                "est_T":est_T,

                "sim_sim":sim_sim,
                "sim_calib_xvec":sim_calib_xvec,
                "sim_sim_nxt_xvec": sim_sim_nxt_xvec,
                "sim_sim_fin_xvec": sim_sim_fin_xvec,

                "ss_sim":ss_sim,
                "ss_sim_nxt_output":ss_sim_nxt_output,
                "ss_sim_nxt_xvec":ss_sim_nxt_xvec,
                "ss_sim_fin_xvec":ss_sim_fin_xvec,
                
                "est_lip_batch":ss_est_lip_batch,
                "est_pcd":ss_est_pcd,
            }
        else:
            return ValueError("mode error")
        
    def CalSim_byGivenRTVEC(self,data,rvec,tvec,ini_T):
        '''Calibration and Calculate the similarity score after calibration
        '''
        lip_batch = data["lip_batch"]
        pcd_batch = data["pcd_batch"]
        proj_shape = (lip_batch.shape[2],lip_batch.shape[3])
        B = lip_batch.shape[0]
        K_batch = data["K_batch"]
        
        # calibration res
        sim_iter_T = pose.batch_rtTensor2TmatTensor(rvec,tvec)
        sim_est_T = sim_iter_T.bmm(ini_T)
        est_lip_batch,est_pcd = transform.batch_projectLiDARDepth(K_batch,sim_est_T,
                                pcd_batch,proj_shape = proj_shape,proj_channel=3)
        
        data["est_pcd"] = est_pcd         
        data["est_lip_batch"] = est_lip_batch
        
        data = self._generateDiffMap(data,mode = "calib")

        # cal uncertainty score
        with torch.no_grad():
            sim_nxt_output = self.sim_estimator(data)
        sim_nxt_iter_T = pose.batch_rtTensor2TmatTensor(sim_nxt_output["rvec"],sim_nxt_output["tvec"])
        
        sim_nxt_iter_xvec = pose.batch_TmatTensor2XVec6Tensor(sim_nxt_iter_T)

        sim_fin_iter_T = sim_nxt_iter_T.bmm(sim_iter_T)
        sim_fin_iter_xvec = pose.batch_TmatTensor2XVec6Tensor(sim_fin_iter_T)

        sim_sim = pose.CalPoseSimBatch_tensor_Indiv(sim_fin_iter_xvec,pose.batch_TmatTensor2XVec6Tensor(sim_iter_T),print_err = False) # (B,)
        
        return sim_sim,sim_nxt_iter_xvec,sim_fin_iter_xvec,sim_nxt_output,est_pcd,est_lip_batch
        
if __name__ == "__main__":
    
    pass
