#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import time
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
from threading import Thread
import os
import json
from loss import*
from config import*
from utils import*
from camera import*
from pcloud import*
from depthReactify import*
from tqdm import tqdm
import random
from operator import itemgetter
from pcloud import*
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
import gc

class ZV_Dataset(Dataset):
    '''
    Subset of ZV Dataloader
    '''
    
    # REMMAPPING FOR SENSOR DATA
    _REMAPPING = True
    _LIDAR_FOV_PROJ_K = np.array([[800,0,768],[0,800,256],[0,0,1]],dtype=np.float32)
    _LIDAR_FOV_PROJ_SIZE = (512,1536)
    
    # REMMAPPING FOR Camera Depth Cloud
    _DCLOUD_PROJ_K = np.array([[700,0,320],[0,700,192],[0,0,1]],dtype=np.float32)
    _DCLOUD_PROJ_SIZE = (384,640)

    # REMMAPPING FOR Projections
    _REMAP_PROJ_K = np.array([[600,0,256],[0,600,128],[0,0,1]],dtype=np.float32)
    _REMAP_SIZE = (256,512)

    # Data Generating Type
    _GENERATE_DEPTHCLOUD = True
    _DCP_CLOUD_REGION_CUT = True
    _DCP_CLOUD_RANDOM_CUT = False
    _DCP_CLOUD_BIO_CUT = False
    _CLOUD_DOWN_SAMPLE = False
    _LIP_DENSIFY_METHOD = "max"

    
    _RGB_TENSOR_TYPE = "lip"
    _DEPTH_INPUT_TYPE = "points_map"
    
    _DEPTH_PROJ_CHANNEL = 3
    
    _LIP_DENSIFY = True
    _DENSE_KSIZE_VALUE = (9,9)
    
    _DEPTH_SCALE = True

    _DATA_GENERATE_TYPE = "Std"
    _EXTERNAL_GETITEM_FUNCS = {}
    _EXTERNAL_BATCH_DEAL_FUNCS = {}

    _TMP_JSON_PATH = "./output/tmp.json"
    def __init__(self,name = "train", config:dict = {}, path_dict:dict = {},device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        self.name = str(name)  # dataset_name
        self.dataset_name_list = name

        self.rgb_path_list = []
        self.lip_path_list = []
        self.depth_path_list = []
        self.pcd_path_list = []
        self.cam_path_list = []
        self.calib_path_list = []
        
        self.config: dict = {}
        self.batch_size: int = 4
        self.scene_num = 0
        self.projection_channel = 3
        self.projection_type = "Depth"
        
        
        self.loadConfig(config)
        
        self.data = None
        
        # calib json
        self.calib_json_list:list = []

        
        self.cloud_sim_score_list: list[np.ndarray] = []
        
        self.scene_ava_list: list[np.ndarray] = []
        

        # load data from paths
        self.path_dict = path_dict
        
        
        self.R_iniGuess = np.eye(3)
        self.t_iniGuess = np.zeros(3)
        self.T_rel_persistent = False
        self.T_rel = np.eye(4)
        self.same_T_vir1_2_vir2_perBatch = False
        
    
    def __len__(self):
        return int(self.scene_num)
        
    
    def getBatchNum(self):
        return int(self.scene_num/self.batch_size)
        

    @classmethod
    def registerDataGenerateType(cls, data_generate_type, getitem_func=None, batch_deal_func=None):
        if getitem_func is not None:
            cls._EXTERNAL_GETITEM_FUNCS[data_generate_type] = getitem_func
        if batch_deal_func is not None:
            cls._EXTERNAL_BATCH_DEAL_FUNCS[data_generate_type] = batch_deal_func


    def set_T_initialGuess(self, r_ini, t_ini, R_base = zv.R_LVINI_NP):
        R_ini_input = pose.EulerYPR2R(r_ini)
        R_ini = np.matmul(R_ini_input,R_base)
        t_ini_afterR = np.matmul(R_ini_input,np.array(t_ini))
        
        self.R_iniGuess_input = (R_ini_input)
        self.t_iniGuess_input = np.array(t_ini)
        self.R_iniGuess = R_ini
        self.t_iniGuess = t_ini_afterR
        

    def set_T_initialGuess_fromT(self, T, R_base = zv.R_LVINI_NP):
        R_ini,t_ini = pose.T2Rt(T)
        self.R_iniGuess_input = np.matmul(R_ini,np.linalg.inv(R_base))
        self.t_iniGuess_input = np.matmul(np.linalg.inv(self.R_iniGuess_input), t_ini)
        self.R_iniGuess = R_ini
        self.t_iniGuess = t_ini
        #self.T_iniGuess = T

    def get_T_initialGuess(self):
        return pose.Rt2T(self.R_iniGuess,self.t_iniGuess)

    def set_T_relative_from_rEuler_t(self, r_rel, t_rel, R_base = zv.R_LVINI_NP):
        R_rel = pose.EulerYPR2R(r_rel)
        R_rel = np.matmul(R_rel,t_rel)
        t_rel = np.array(t_rel)
        self.R_rel = R_rel
        self.t_rel = t_rel
        self.T_rel = pose.Rt2T(self.R_rel, self.t_rel)
    
    def set_T_relative(self, T_rel):
        self.T_rel = T_rel

    def updateTmpJson(self):
        json_content = {
            "T_rel_persistent":self.T_rel_persistent,
            "T_rel":self.T_rel.tolist(),
            "R_iniGuess":self.R_iniGuess.tolist(),
            "t_iniGuess":self.t_iniGuess.tolist()
        }
        with open(self._TMP_JSON_PATH, 'w', encoding='utf-8') as f:  
            json_str = json.dumps(json_content, indent=4)  
            json_str = zv.compactJsonArrays(json_str, keys=list(json_content.keys()))
            f.write(json_str)
            

    def syncTmpJson(self):
        if os.path.exists(self._TMP_JSON_PATH):
            with open(self._TMP_JSON_PATH, 'r') as tmp_json_file:
                tmp_json_content = json.load(tmp_json_file)
                self.T_rel_persistent = tmp_json_content["T_rel_persistent"]
                self.T_rel = np.array(tmp_json_content["T_rel"])
                self.R_iniGuess = np.array(tmp_json_content["R_iniGuess"])
                self.t_iniGuess = np.array(tmp_json_content["t_iniGuess"])

    def loadConfig(self,config:dict):
        if config != {}:
            self.config: dict = config
            self.batch_size = self.config["batch_size"]

            if "cloud_downsample_ratio" in self.config:
                self.cloud_downsample_ratio = self.config["cloud_downsample_ratio"]   # point cloud downsample ratio
            else:
                self.cloud_downsample_ratio = 400000
            
            # data generate initial range
            if "r_Euler_range2" in self.config:
                self.r_rand_range2 = self.config["r_Euler_range2"]
            else:
                self.r_rand_range2 = np.array([3.0,1.0,1.0])

            if "t_range2" in self.config:
                self.t_rand_range2 = self.config["t_range2"]
            else:
                self.t_rand_range2 = np.array([0.3,0.1,0.1])

            if "r_Euler_range" in self.config:
                self.r_rand_range = self.config["r_Euler_range"]
            else:
                self.r_rand_range = np.array([3.0,1.0,1.0])

            if "t_range" in self.config:
                self.t_rand_range = self.config["t_range"]
            else:
                self.t_rand_range = np.array([0.3,0.1,0.1])

            if "t_offset" in self.config:
                self.t_offset = self.config["t_offset"]
            else:
                self.t_offset = np.array([0.0,0.0,0.0])

            if "r_Euler_offset" in self.config:
                self.r_offset = self.config["r_Euler_offset"]
            else:
                self.r_offset = np.array([0.0,0.0,0.0])


            if "projection_type" in self.config:
                self.projection_type = self.config["projection_type"]
                print(f"{BLUE}projection_type{RESET} {RED}[{self.projection_type}]{RESET}")
            else:
                self.projection_type = "Intensity"
                print(f"{BLUE}projection_type{RESET} {RED}[{self.projection_type}]{RESET}")
            
        else:
            self.config: dict = {}
            self.batch_size: int = 4
            self.cloud_downsample_ratio = 1    # point cloud downsample ratio
            
    def setDataGenerateRandRange_byInput(self,r_rand_range,t_rand_range):
        '''Set the first data-generation range from the input.
        '''
        self.r_rand_range = r_rand_range
        self.t_rand_range = t_rand_range

    def setDataGenerateRandRange2_byInput(self,r_rand_range2,t_rand_range2):
        '''Set the second data-generation range from the input.
        '''
        self.r_rand_range2 = r_rand_range2
        self.t_rand_range2 = t_rand_range2
    
    def loadDataFromPaths(self,path_dict:dict = {}, sameTperBatch = False):
        '''Load Data from given paths

        '''
        print(f"{BLUE}load Data from Paths{RESET} [{GREEN} All at Once {RESET}]")
        self.same_T_vir1_2_vir2_perBatch = sameTperBatch
        
        # copy public attributes for multi process loading
        self.DATA_GENERATE_TYPE = self._DATA_GENERATE_TYPE      # [Std | Calib]  
        self.GENERATE_DEPTHCLOUD = self._GENERATE_DEPTHCLOUD    # [True | False] 
        self.RGB_TENSOR_TYPE = self._RGB_TENSOR_TYPE            # ["lip" | "dcp"] 
        self.LIP_DENSIFY = self._LIP_DENSIFY                    # [True | False] 
        self.DENSE_KSIZE_VALUE = self._DENSE_KSIZE_VALUE        # [(x,x)] 
        self.DCP_CLOUD_REGION_CUT = self._DCP_CLOUD_REGION_CUT  # [True | False] 
        self.DCP_CLOUD_BIO_CUT = self._DCP_CLOUD_BIO_CUT       # [True | False] 
        self.CLOUD_DOWN_SAMPLE = self._CLOUD_DOWN_SAMPLE        # [True | False] 
        
        if path_dict != {}:
            self.path_dict = path_dict
        
        if self.path_dict == {}:
            return 
        self.scene_ava_list = []
        self.rgb_path_list = self.path_dict["rgb_path_list"]
        self.pcd_path_list = self.path_dict["pcd_path_list"]
        self.calib_path_list = self.path_dict["calib_path_list"]
        self.depth_path_list = self.path_dict["depth_path_list"]
        self.scene_num = len(self.rgb_path_list)
        
        self.buffer_epoch_ava_num = self.scene_num

        print("scene num:",self.scene_num)
        if self.scene_num > 0:
            print(f"rgb path list: from {self.rgb_path_list[0]} to {self.rgb_path_list[len(self.rgb_path_list)-1]}, total {len(self.rgb_path_list)} pics.")
            print(f"pcd path list: from {self.pcd_path_list[0]} to {self.pcd_path_list[len(self.pcd_path_list)-1]}, total {len(self.pcd_path_list)} pics.")
            print(f"depth path list: from {self.depth_path_list[0]} to {self.depth_path_list[len(self.depth_path_list)-1]}, total {len(self.depth_path_list)} pics.")
            print(f"calib path list: from {self.calib_path_list[0]} to {self.calib_path_list[len(self.calib_path_list)-1]}, total {len(self.calib_path_list)} pics.")
        
        self._DEPTHMAP_REMAPPING = False

        for i in range(50):
            rand_idx = random.randint(0,self.scene_num-1)
            depth_path = self.depth_path_list[rand_idx]
            depth_map = loadDepthMap(depth_path)
            is_too_large = not zv.isProjSizeSim(self._DCLOUD_PROJ_SIZE, depth_map.shape, thres = 100.0)
            if (is_too_large):
                self._DEPTHMAP_REMAPPING = True
                print(f"{RED}Depth map size is too large, enable remapping for depth cloud projection{RESET}")
                break
        self.virtula_camera = VCamera(self._REMAP_SIZE,self._REMAP_PROJ_K)

        assert ((self.scene_num == len(self.pcd_path_list)) and (self.scene_num == len(self.calib_path_list))), "scene num conflict, scene num is not the same between rgb, pcd, and calib"

        self.updateTmpJson()  

    def getSceneIdxFromBatchIdx(self,batch_idx):
        scene_idx = int(batch_idx*self.batch_size)
        return scene_idx

    def getPVirCamera(self):
        return self.virtula_camera
    
    def double_sided_pose_aug(self,T_center):
        T_vir2_arr = pose.GenerateRandomT_byCenters([T_center],
                                                    t_disturb_range=self.t_rand_range2,
                                                    euler_disturb_range=self.r_rand_range2)[0]
        
        if self.T_rel_persistent:
            #print("T persistent")
            T_vir1_arr = np.matmul(np.linalg.inv(self.T_rel),T_vir2_arr)
        else:
            T_vir1_arr = pose.GenerateRandomT_byCenters([T_vir2_arr],
                                                        t_disturb_range=self.t_rand_range,
                                                        euler_disturb_range=self.r_rand_range)[0]
        return T_vir1_arr, T_vir2_arr

    def _get_itemFunc_Std(self,idx):
        '''Datageneration Method for LiDAR-camera calibration trainning and testing, tailored for single-branch network architecture.
        '''

        print_time = False
        time_p = time.time()
        
        scene_idx = int(idx)
        
        # Load the calibration ground truth (T matrix from LiDAR to camera)
        calib_json_path = self.calib_path_list[scene_idx]
        intrinsic_K,T_lidar2cam,R_lidar2cam,t_lidar2cam = pose.getIntrinsicExtrinsicFromZVCalibJson(calib_json_path)
        T_lidar2cam_tensor = zv.arr2Tensor(T_lidar2cam)
        self.syncTmpJson()
        
        # Remapping
        if self._REMAPPING:
            K_vircam = self._REMAP_PROJ_K
        else:
            K_vircam = intrinsic_K
        
        K_tensor = zv.arr2Tensor(K_vircam)
        K_ini_tensor = zv.arr2Tensor(intrinsic_K)

        # Double Sided Pose Augmentation
        T_vir1_arr, T_vir2_arr = self.double_sided_pose_aug(T_lidar2cam)
        T_vir1_tensor = zv.arr2Tensor(T_vir1_arr)
        T_vir2_tensor = zv.arr2Tensor(T_vir2_arr)
        
        #Load Point Cloud
        pcd_path = self.pcd_path_list[scene_idx]
        points_xyz_raw, intensity = loadPCDFromPath(pcd_path)
        
        if print_time == True:
            print(f"load pcd file:{time.time()-time_p}s")
            time_p = time.time()
        
        
        #if self.GENERATE_DEPTHCLOUD:
        if self.DCP_CLOUD_REGION_CUT:
            points_xyz_ini = cloudXYZRegionCut(points_xyz_raw,z_range = [-10,5])


        if self.LIP_DENSIFY:
            # (H*W, 3)
            points_xyz = cloudXYZDensify_byVCameraView(points_xyz_ini,self._LIDAR_FOV_PROJ_K,T_vir1_arr,proj_shape = self._LIDAR_FOV_PROJ_SIZE,dense_ksize = self.DENSE_KSIZE_VALUE,method = self._LIP_DENSIFY_METHOD)
        else:
            # (H*W, 3)
            points_xyz = cloudXYZDensify_byVCameraView(points_xyz_ini,self._LIDAR_FOV_PROJ_K,T_vir1_arr,proj_shape = self._LIDAR_FOV_PROJ_SIZE,dense_ksize = (0,0),method = self._LIP_DENSIFY_METHOD)
            
                
        '''
        --------------------- Infer Depth Cloud from Camera Image -----------------------
        '''
        if self.GENERATE_DEPTHCLOUD:
            # load camera depth estimation result            
            camDepth_path = self.depth_path_list[scene_idx]
            points_map_arr = loadPointsMap(camDepth_path)
            points_map_arr_shape = points_map_arr.shape
            assert len(points_map_arr_shape) >= 2, "points_map_arr_shape should be like (H,W,X), where X >= 0"
            if len(points_map_arr_shape) > 2:
                if points_map_arr_shape[2] > 1:
                    # (H,W,3)
                    depth_map = points_map_arr[:,:,2]
                else:
                    # (H,W,1)
                    depth_map = points_map_arr[:,:,0]
            else:
                # (H,W)
                depth_map = points_map_arr[:,:]

            if self._DEPTHMAP_REMAPPING:
                depth_map = zv.remapArr2D_Float(depth_map,intrinsic_K,self._DCLOUD_PROJ_K,self._DCLOUD_PROJ_SIZE)
                depth_map_K = self._DCLOUD_PROJ_K
                pass
            else:
                depth_map_K = intrinsic_K
                
            if self.DCP_CLOUD_REGION_CUT:
                # (H*W,3)
                if self._DCP_CLOUD_RANDOM_CUT:
                    gen_depth_cut_depth = random.uniform(55,60)
                else:
                    gen_depth_cut_depth = 57.5
                
                depth_map[(depth_map < 3) | (depth_map > gen_depth_cut_depth)] = 0.0
                            
            if self._DEPTH_SCALE:
                points_xyz_ldp_at_gt = cloudXYZProjAndDensify(points_xyz_raw,depth_map_K,T_lidar2cam,proj_shape = depth_map.shape,dense_ksize = self.DENSE_KSIZE_VALUE,method = self._LIP_DENSIFY_METHOD)
                                
                # (H*W, 3), (H*W)
                gen_depthCloud_xyz, gen_depthCloud_scale = depthCloudScale(depth_map, points_xyz_ldp_at_gt,
                                                                            depth_map_K,proj_shape = depth_map.shape,
                                                                            scale_diff_thres=[8.0,8.0],
                                                                            scale_ratio=1.0,
                                                                            densify_ref=False,
                                                                            extract_from_ref_valid=False,
                                                                            filt_zero_depth=False,
                                                                            visual_process=False)    
            
            else:
                # (H*W, 3)
                gen_depthCloud_xyz, _ = transform.np_unproject_depth_to_pointcloud(depth_map,depth_map_K,
                                                                                    T = zv.T_EYE_NP,filt_zero_depth=False)


            ini_depthCloudXYZ_tensor = zv.arr2Tensor(gen_depthCloud_xyz.T)
            
            # transform to LiDAR coordinate system
            gen_depthCloud_xyz = transform.np_transPCloud_T(gen_depthCloud_xyz,np.linalg.inv(T_lidar2cam))  # (H*W, 3)
            
            # downsampling
            if self.cloud_downsample_ratio != 1 and self.CLOUD_DOWN_SAMPLE:
                gen_depthCloud_xyz = PCloudXYZ_Sample(gen_depthCloud_xyz, self.cloud_downsample_ratio)
            
            
            depthCloudXYZ_tensor = zv.arr2Tensor(gen_depthCloud_xyz.T)  # 3 x N
            
            if self._DEPTH_SCALE:
                depthCloudScale_tensor = zv.arr2Tensor(gen_depthCloud_scale)
            else:
                depthCloudScale_tensor = torch.tensor([],dtype=torch.float32)


        # point cloud downsampling
        if self.cloud_downsample_ratio != 1 and self.CLOUD_DOWN_SAMPLE:
            points_xyz = PCloudXYZ_Sample(points_xyz, self.cloud_downsample_ratio)
            
        pcd_tensor= zv.arr2Tensor(points_xyz.T)
        
        if print_time == True:
            print(f"pcd down sampling:{time.time()-time_p}s")
            time_p = time.time()   

        if self.GENERATE_DEPTHCLOUD:
            pass
        else:
            depthCloudXYZ_tensor = torch.tensor([],dtype=torch.float32)
            depthCloudScale_tensor = torch.tensor([],dtype=torch.float32)
            ini_depthCloudXYZ_tensor = torch.tensor([],dtype=torch.float32)
            pass

        rgb_tensor = torch.tensor([],dtype=torch.float32)
        camDepth_tensor = torch.tensor([],dtype=torch.float32)      
        
        if print_time == True:
            print(f"load gt:{time.time()-time_p}s")
            time_p = time.time()   

        return dict(rgb_batch = rgb_tensor,
                    camDepth_batch = camDepth_tensor,
                    depthCloudXYZ_batch = depthCloudXYZ_tensor,
                    depthCloudScale_batch = depthCloudScale_tensor,
                    ini_depthCloudXYZ_batch = ini_depthCloudXYZ_tensor,
                    T_lidar2cam_batch = T_lidar2cam_tensor,
                    T_vir1_batch = T_vir1_tensor,
                    T_vir2_batch = T_vir2_tensor,
                    K_batch = K_tensor,
                    K_ini_batch = K_ini_tensor,
                    pcd_batch = pcd_tensor,
                    scene_idx_batch = torch.tensor(scene_idx)
                    )
    
    def _get_itemFunc_Calib(self,idx):
        print_time = False
        time_p = time.time()
        
        scene_idx = int(idx)
        
        # Load the calibration ground truth (T matrix from LiDAR to camera)
        calib_json_path = self.calib_path_list[scene_idx]
        intrinsic_K,T_lidar2cam,R_lidar2cam,t_lidar2cam = pose.getIntrinsicExtrinsicFromZVCalibJson(calib_json_path)
        T_lidar2cam_tensor = zv.arr2Tensor(T_lidar2cam)
        self.syncTmpJson()

        if self._REMAPPING:
            K_vircam = self._REMAP_PROJ_K
        else:
            K_vircam = intrinsic_K
        K_tensor = zv.arr2Tensor(K_vircam)
        K_ini_tensor = zv.arr2Tensor(intrinsic_K)

        if print_time == True:
            print(f"load calib json:{time.time()-time_p}s")
            time_p = time.time()
        
        if print_time == True:
            print(f"load rgb & depth images:{time.time()-time_p}s")
            time_p = time.time()

        
        T_ini = pose.Rt2T(self.R_iniGuess,self.t_iniGuess)    
                    
        T_ini_tensor = zv.arr2Tensor(T_ini)
            
        
        # load point clouds
        pcd_path = self.pcd_path_list[scene_idx]
        points_xyz, intensity = loadPCDFromPath(pcd_path)
        
        if print_time == True:
            print(f"load pcd file:{time.time()-time_p}s")
            time_p = time.time()
        
        
        if self.DCP_CLOUD_REGION_CUT:
            # (H*W,3)
            points_xyz = cloudXYZRegionCut(points_xyz,z_range = [-10,5])

        if self.LIP_DENSIFY:
            # (H*W, 3)
            points_xyz = cloudXYZDensify_byVCameraView(points_xyz,self._LIDAR_FOV_PROJ_K,T_ini,proj_shape = self._LIDAR_FOV_PROJ_SIZE,dense_ksize = self.DENSE_KSIZE_VALUE,method = self._LIP_DENSIFY_METHOD)
        else:
            # (H*W, 3)
            points_xyz = cloudXYZDensify_byVCameraView(points_xyz,self._LIDAR_FOV_PROJ_K,T_ini,proj_shape = self._LIDAR_FOV_PROJ_SIZE,dense_ksize = (0,0),method = self._LIP_DENSIFY_METHOD)    
        

        # generate camera depth cloud
        if self.GENERATE_DEPTHCLOUD:
            # load camera depth estimation result
            camDepth_path = self.depth_path_list[scene_idx]
            points_map_arr = loadPointsMap(camDepth_path)
            points_map_arr_shape = points_map_arr.shape
            assert len(points_map_arr_shape) >= 2, "points_map_arr_shape should be like (H,W,X), where X >= 0"
            if len(points_map_arr_shape) > 2:
                if points_map_arr_shape[2] > 1:
                    # (H,W,3)
                    depth_map = points_map_arr[:,:,2]
                else:
                    # (H,W,1)
                    depth_map = points_map_arr[:,:,0]
            else:
                # (H,W)
                depth_map = points_map_arr[:,:]
            
            
            if self._DEPTHMAP_REMAPPING:
                depth_map = zv.remapArr2D_Float(depth_map,intrinsic_K,self._DCLOUD_PROJ_K,self._DCLOUD_PROJ_SIZE)
                depth_map_K = self._DCLOUD_PROJ_K
                pass
            else:
                depth_map_K = intrinsic_K

            if self.DCP_CLOUD_REGION_CUT:
                # (H*W,3)
                if self._DCP_CLOUD_RANDOM_CUT:
                    gen_depth_cut_depth = random.uniform(55,60)
                else:
                    gen_depth_cut_depth = 57.5
                depth_map[(depth_map < 3) | (depth_map > gen_depth_cut_depth)] = 0.0
                
            # depth cloud generation
            gen_depthCloud_xyz, _ = transform.np_unproject_depth_to_pointcloud(depth_map,depth_map_K,
                                                                               T = zv.T_EYE_NP,filt_zero_depth=False)

            ini_depthCloudXYZ_tensor = zv.arr2Tensor(gen_depthCloud_xyz.T)
            depthCloudXYZ_tensor = zv.arr2Tensor(gen_depthCloud_xyz.T)  # 3 x N
            
        pcd_tensor= zv.arr2Tensor(points_xyz.T)
        
        if print_time == True:
            print(f"pcd down sampling:{time.time()-time_p}s")
            time_p = time.time()   

        rgb_tensor = torch.tensor([],dtype=torch.float32)
        camDepth_tensor = torch.tensor([],dtype=torch.float32)      
        
        if print_time == True:
            print(f"load gt:{time.time()-time_p}s")
            time_p = time.time()   

        return dict(rgb_batch = rgb_tensor,
                    camDepth_batch = camDepth_tensor,
                    depthCloudXYZ_batch = depthCloudXYZ_tensor,
                    ini_depthCloudXYZ_batch = ini_depthCloudXYZ_tensor,
                    T_lidar2cam_batch = T_lidar2cam_tensor,
                    T_ini_batch = T_ini_tensor,
                    K_batch = K_tensor,
                    K_ini_batch = K_ini_tensor,
                    pcd_batch = pcd_tensor,
                    scene_idx_batch = torch.tensor(scene_idx)
                    )

    def __getitem__(self, idx):
        assert self.DATA_GENERATE_TYPE in ("Std", "Calib", "Default"), "DATA_GENERATE_TYPE must be in (Std, Calib, Default)"
        if self.DATA_GENERATE_TYPE == "Std":
            return self._get_itemFunc_Std(idx)
        elif self.DATA_GENERATE_TYPE == "Calib":
            return self._get_itemFunc_Calib(idx)
        else:
            return self._get_itemFunc_Std(idx)

    def generateTArrForProjections(self,batch = None, method = "same_all_batch"):
        '''Generate T projections for training process
        Args:
            method: "diff_per_batch" | "same_all_batch"
        Return:
            T_batch
        '''
        
        if method == "diff_per_batch":
            T_ini_list = [np.eye(4) for _ in range(self.batch_size)]
            T_ini_arr = np.stack(T_ini_list, axis=0)
            
            T_vir1_to_vir2_arr = pose.GenerateRandomT_byCenters_diffBias(T_ini_arr,
                                                                        t_disturb_range=self.t_rand_range,
                                                                        euler_disturb_range=self.r_rand_range)
                
            return T_vir1_to_vir2_arr
            
        elif method == "same_all_batch":
            
            T_ini_list = [np.eye(4) for _ in range(self.batch_size)]
            T_ini_arr = np.stack(T_ini_list, axis=0)
            
            T_vir1_to_vir2_arr = pose.GenerateRandomT_byCenters_sameBias(T_ini_arr,
                                                                        t_disturb_range=self.t_rand_range,
                                                                        euler_disturb_range=self.r_rand_range)
                
            
            return T_vir1_to_vir2_arr
        else:
            assert 0, "Input method do not exist"
    
    def batchDeal(self,batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        T_lidar2cam_batch = batch["T_lidar2cam_batch"]
        T_vir1_batch = batch["T_vir1_batch"]
        T_vir2_batch = batch["T_vir2_batch"]

        if self.same_T_vir1_2_vir2_perBatch:
            # Adopt the same Relatvie T transform in a single batch
            T_eye_list = [np.eye(4) for _ in range(self.batch_size)]
            T_eye_arr = np.stack(T_eye_list, axis=0)
            T_rand_arr = pose.GenerateRandomT_byCenters(T_eye_arr,
                                                        t_disturb_range=self.t_rand_range,
                                                        euler_disturb_range=self.r_rand_range)
            
            T_rel_batch = zv.arr2Tensor(T_rand_arr).to(T_vir2_batch)
            T_vir1_batch = torch.linalg.inv(T_rel_batch).bmm(T_vir2_batch)

            batch["T_vir1_batch"] = T_vir1_batch


        if self.T_rel_persistent:
            # Adopt the same Relative T transform across all batches
            T_rel_list = [self.T_rel for _ in range(self.batch_size)]
            T_rel_arr = np.stack(T_rel_list, axis=0)
            
            T_rel_batch = zv.arr2Tensor(T_rel_arr).to(T_vir2_batch)
            T_vir1_batch = torch.linalg.inv(T_rel_batch).bmm(T_vir2_batch)
            
            batch["T_vir1_batch"] = T_vir1_batch

        
        depthCloudXYZ_batch = batch["depthCloudXYZ_batch"]
        if "depthCloudScale_batch" in batch:
            depthCloudScale_batch = batch["depthCloudScale_batch"]
        
        pcd_batch = batch["pcd_batch"]    
        rgb_batch = batch["rgb_batch"]
        
        T_lidar2vir_batch = T_vir1_batch
        T_vir2cam_batch = T_lidar2cam_batch.bmm(torch.linalg.inv(T_vir1_batch))
        T_vir1_2_vir2_batch = T_vir2_batch.bmm(torch.linalg.inv(T_vir1_batch))

        lip_batch,pcd_trans_batch = self.virtula_camera.batch_projectLiDARDepth(T_lidar2vir_batch,pcd_batch,proj_channel=self._DEPTH_PROJ_CHANNEL) # (b, 3, H, W)
        
        
        if self.GENERATE_DEPTHCLOUD:
            calibed_pcd_batch = se3.transform(T_vir2_batch,pcd_batch)
        else:
            
            calibed_pcd_batch = se3.transform(T_vir2_batch,pcd_batch)

            calibed_depthCloudXYZ_batch = torch.zeros_like(depthCloudXYZ_batch).to(depthCloudXYZ_batch)

        
        lip2_batch,pcd_trans2_batch = self.virtula_camera.batch_projectLiDARDepth(T_vir2_batch,pcd_batch,proj_channel=self._DEPTH_PROJ_CHANNEL) # (b, 3, H, W)
            
            
        if self.GENERATE_DEPTHCLOUD:
            if self._DEPTH_SCALE:
                depthCloudXYZ_proj_batch, depthCloudXYZ_trans_batch = self.virtula_camera.batch_projectLiDARDepth_withScale(T_vir2_batch,depthCloudXYZ_batch,
                                                                                                                            depthCloudScale_batch,
                                                                                                                            proj_channel=self._DEPTH_PROJ_CHANNEL) # (b, 3, H, W)
            else:
                depthCloudXYZ_proj_batch, depthCloudXYZ_trans_batch = self.virtula_camera.batch_projectLiDARDepth(T_vir2_batch,depthCloudXYZ_batch,proj_channel=self._DEPTH_PROJ_CHANNEL,max_z=100) # (b, 3, H, W)
        
        else:
            depthCloudXYZ_proj_batch = torch.tensor([],dtype=torch.float32)        
            depthCloudXYZ_trans_batch = torch.tensor([],dtype=torch.float32)       
        
        
        r_gt_batch,t_gt_batch = pose.batch_TmatTensor2rtTensor(T_vir1_2_vir2_batch) # rvec, tvec
        gt_batch = torch.cat([r_gt_batch,t_gt_batch],dim=1)   # xvec6 [rvec, tvec] size: (b, 6)
        

        # batch update
        if self.RGB_TENSOR_TYPE == "lip":
            pBatch_rgb_tensor = lip2_batch[0]
        elif self.RGB_TENSOR_TYPE == "dcp":
            pBatch_rgb_tensor = depthCloudXYZ_proj_batch[0]
        elif self.RGB_TENSOR_TYPE == "rgb":
            pBatch_rgb_tensor = rgb_batch[0]
        else:
            pBatch_rgb_tensor = lip2_batch[0]
        batch["rgb_tensor"] = pBatch_rgb_tensor
        batch["depthCloudXYZ_proj_batch"] = depthCloudXYZ_proj_batch
        batch["lip_batch"] = lip_batch
        batch["lip2_batch"] = lip2_batch
        batch["calibed_pcd_batch"] = calibed_pcd_batch
        batch["pcd_trans_batch"]  = pcd_trans_batch
        batch["pcd_trans2_batch"]  = pcd_trans2_batch
        batch["T_lidar2vir_batch"] = T_lidar2vir_batch
        batch["T_lidar2vir2_batch"] = T_vir2_batch
        batch["T_vir1_2_vir2_batch"] = T_vir1_2_vir2_batch
        batch["T_vir2cam_batch"] = T_vir2cam_batch
        batch["depthCloudXYZ_trans_batch"] = depthCloudXYZ_trans_batch
        batch["label_batch"] = gt_batch
        
        
        return batch
    
    def batchDeal_Calib(self,batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}

        T_lidar2cam_batch = batch["T_lidar2cam_batch"]
        T_ini_batch = batch["T_ini_batch"]
        depthCloudXYZ_batch = batch["depthCloudXYZ_batch"]
        
        #K_batch = batch["K_batch"]
        pcd_batch = batch["pcd_batch"]    
        rgb_batch = batch["rgb_batch"]

        T_lidar2cam_arr = zv.tensor2Arr(T_lidar2cam_batch)

        T_ini_arr = zv.tensor2Arr(T_ini_batch)

        # generate Relative Pose Extrinsic matrix T_vir2cam
        T_ini2cam_arr = np.matmul(T_lidar2cam_arr,np.linalg.inv(T_ini_arr))
        T_ini2cam_batch = zv.arr2Tensor(T_ini2cam_arr).to(self.device)

        lip_batch,pcd_trans_batch = self.virtula_camera.batch_projectLiDARDepth(T_ini_batch,pcd_batch,proj_channel=self._DEPTH_PROJ_CHANNEL) # (b, 3, H, W)

        T_eye_batch = torch.eye(4).expand(self.batch_size, 4, 4).to(depthCloudXYZ_batch)
        depthCloudXYZ_proj_batch, depthCloudXYZ_trans_batch = self.virtula_camera.batch_projectLiDARDepth(T_eye_batch,depthCloudXYZ_batch,
                                                                                                          proj_channel=self._DEPTH_PROJ_CHANNEL,
                                                                                                          max_z = 100) # (b, 3, H, W)
        
        r_gt_batch,t_gt_batch = pose.batch_TmatTensor2rtTensor(T_ini2cam_batch) # rvec, tvec
        gt_batch = torch.cat([r_gt_batch,t_gt_batch],dim=1)   # xvec6 [rvec, tvec] size: (b, 6)
        
        # batch update
        pBatch_rgb_tensor = depthCloudXYZ_proj_batch[0]
        
        batch["rgb_tensor"] = pBatch_rgb_tensor
        batch["depthCloudXYZ_proj_batch"] = depthCloudXYZ_proj_batch
        batch["lip_batch"] = lip_batch
        batch["T_lidar2vir_batch"] = T_ini_batch
        batch["pcd_trans_batch"]  = pcd_trans_batch
        batch["label_batch"] = gt_batch
        batch["depthCloudXYZ_trans_batch"] = depthCloudXYZ_trans_batch
        
        return batch
    
class ZV_Dataset_Loader():
    '''
    Support load datasets
    '''
    def __init__(self,device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        
        self.train_datasets_list : list[ZV_Dataset] = [] # the list that contains zvdatasets
        self.val_datasets_list: list[ZV_Dataset] = []
        self.test_datasets_list: list[ZV_Dataset] = []
        
        self.RGB_folder = ""
        self.CAM_folder = ""
        self.PCloud_folder = ""
        self.Calib_folder = ""
        
        self.config: dict = {}
        self.train_method = "select dataset"
        self.dataset_root_path = ""
        self.epoch: int = 10
        self.batch_size: int = 8
        self.scene_num = 0
        self.val_scene_num = 0
        self.projection_channel = 3
        self.cloud_downsample_ratio = 1    # point cloud downsample ratio
        
        self.data = None
        

    def loadFromJsonFile(self,json_path,device = "cuda"):
        '''
        rgb_path_list, pcd_path_list, calib_path_list
        '''
        
        start_t = time.time()
        
        #json_path = tkinter.filedialog.askopenfilename(title = "select config json")
        config_file = open(json_path, 'r')
        config_loader = json.load(config_file)
        self.config = config_loader

        if "batch_gen_method" in config_loader:
            self.batch_gen_method = config_loader["batch_gen_method"]
        
        # Load dataset loading method and dataset path root and projection details
        self.train_method = config_loader["train_method"]
        self.dataset_root_path =  config_loader["dataset_root"]
        self.epoch_num = config_loader["epoch"]
        self.batch_size = config_loader["batch_size"]

        if "cloud_downsample_ratio" in config_loader:
            self.cloud_downsample_ratio = config_loader["cloud_downsample_ratio"]
        else:
            self.cloud_downsample_ratio = 1
        
        if self.train_method == "select dataset":
            # load mode params of select index mode
            assert "select_dataset_mode_params" in config_loader, "json file doesn't have attribute <select dataset mode params>, please check your content!"
            mode_params = config_loader["select_dataset_mode_params"]
            if "dataset_folder_format" in config_loader:
                dataset_folder_format = config_loader["dataset_folder_format"]
            else:
                dataset_folder_format = {
                    "RGB_folder":"RGB",
                    "PCloud_folder":"pointclouds",
                    "Depth_folder":"Depth",
                    "Calib_folder":"calib"
                }

            if "train_subset" in mode_params:
                self.train_subsets_path = mode_params["train_subset"]
                self.train_sample_mode = mode_params["train_sample_mode"]
                self.train_subset = self.loadSubsets(self.train_subsets_path,self.train_sample_mode,
                                                                 dataset_folder_format,"train")
            else:
                self.train_subset = None
                print(f"{YELLOW}Train datasets are not configured{RESET}")

            if "val_subset" in mode_params:
                self.val_subsets_path = mode_params["val_subset"]
                self.val_sample_mode = mode_params["val_sample_mode"]
                self.val_subset = self.loadSubsets(self.val_subsets_path,self.val_sample_mode,
                                                    dataset_folder_format,"val")
            else:    
                self.val_subset = None
                print(f"{YELLOW}Val datasets are not configured{RESET}")

            if "test_subset" in mode_params:
                self.test_subsets_path = mode_params["test_subset"]
                self.test_sample_mode = mode_params["test_sample_mode"]
                self.test_subset = self.loadSubsets(self.test_subsets_path,self.test_sample_mode,
                                                    dataset_folder_format,"test")
            else:    
                self.test_subset = None
                print(f"{YELLOW}Test datasets are not configured{RESET}")

            
        else:
            raise("Invalid Dataset Format")

    def __len__(self):
        return int(self.scene_num/self.batch_size)
    
    def createDataloader(self,subset:ZV_Dataset,num_workers = 4, pre_loading = True, shuffle = True, persistend_workers = True,pre_fetch_batch = 4,pin_memory = True,timeout = 0):
        
        loader = DataLoader(
            subset, 
            batch_size=self.batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,   
            pin_memory=pin_memory,   
            drop_last = True,
            timeout = timeout,
            prefetch_factor = pre_fetch_batch,
            persistent_workers=persistend_workers
        )
        if pre_loading:
            print(f"{MAGENTA}Subset Preloading... {RESET}")
            _ = iter(loader)
           
        return loader
    
    def createTrainDataLoader(self,num_workers = 4, pre_loading = True, shuffle = True, persistend_workers = True,pre_fetch_batch = 4,pin_memory = True,timeout = 0):
        if self.train_subset == None:
            print(f"{YELLOW}train_subset is not set, please make sure it is not required{RESET}")
        self.train_dataloader = self.createDataloader(self.train_subset,
                                                     num_workers = num_workers,
                                                     pre_loading = pre_loading,
                                                     shuffle = shuffle, 
                                                     persistend_workers = persistend_workers,
                                                     pre_fetch_batch = pre_fetch_batch,
                                                     pin_memory = pin_memory,
                                                     timeout = timeout)
        return self.train_dataloader

    def createValDataLoader(self,num_workers = 4, pre_loading = True, shuffle = True, persistend_workers = True,pre_fetch_batch = 4,pin_memory = True,timeout = 0):
        if self.val_subset == None:
            print(f"{YELLOW}val_subset is not set, please make sure it is not required{RESET}")
        self.val_dataloader = self.createDataloader(self.val_subset,
                                                     num_workers = num_workers,
                                                     pre_loading = pre_loading,
                                                     shuffle = shuffle, 
                                                     persistend_workers = persistend_workers,
                                                     pre_fetch_batch = pre_fetch_batch,
                                                     pin_memory = pin_memory,
                                                     timeout = timeout)
        return self.val_dataloader

    def createTestDataLoader(self,num_workers = 4, pre_loading = True, shuffle = True, persistend_workers = True,pre_fetch_batch = 4,pin_memory = True,timeout = 0):
        if self.test_subset == None:
            print(f"{YELLOW}test_subset is not set, please make sure it is not required{RESET}")
        self.test_dataloader = self.createDataloader(self.test_subset,
                                                     num_workers = num_workers,
                                                     pre_loading = pre_loading,
                                                     shuffle = shuffle, 
                                                     persistend_workers = persistend_workers,
                                                     pre_fetch_batch = pre_fetch_batch,
                                                     pin_memory = pin_memory,
                                                     timeout = timeout)
        return self.test_dataloader
    
    def deleteDataLoaders(self):
        self.safe_cleanup_dataloader(self.test_dataloader)
        self.safe_cleanup_dataloader(self.train_dataloader)
        self.safe_cleanup_dataloader(self.val_dataloader)
        self.test_dataloader = None
        self.val_dataloader = None
        self.train_dataloader = None
        
    def safe_cleanup_dataloader(self,dataloader):
        try:
            if hasattr(dataloader, '_iterator') and dataloader._iterator is not None:
                dataloader._iterator._shutdown_workers()
            if hasattr(dataloader, '_workers'):
                for worker in dataloader._workers:
                    if worker.is_alive():
                        worker.terminate()
                        worker.join(timeout=5.0)
            dataloader._iterator = None
            print("clean dataloader")
        except Exception as e:
            print(f"clean DataLoader error: {e}")
        finally:
            import gc
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def getTrainBatchLen(self):
        return self.train_subset.getBatchNum()
    
    def getTrainSceneNum(self):
        return self.train_subset.scene_num
    
    def getValBatchLen(self):
        return self.val_subset.getBatchNum()
    
    def getValSceneNum(self):
        return self.val_subset.scene_num
    
    def getTestBatchLen(self):
        return self.test_subset.getBatchNum()

    def getTestSceneNum(self):
        return self.test_subset.scene_num
    
    def loadSubsets(self,subsets_path_list:list,
                    subsets_samp_mode:list,
                    dataset_folder_format:dict,
                    subset_type = "train"):
        path_dict = {
            "rgb_path_list":[],
            "pcd_path_list":[],
            "calib_path_list":[],
            "depth_path_list":[],
        }
        
        for i in range(len(subsets_path_list)):
            subset_path = subsets_path_list[i]
            # generate path folders
            self.RGB_folder = self.dataset_root_path + "/" + subset_path + "/" + dataset_folder_format["RGB_folder"]
            self.PCloud_folder = self.dataset_root_path + "/" + subset_path + "/" + dataset_folder_format["PCloud_folder"]
            self.Calib_folder = self.dataset_root_path + "/" + subset_path + "/" + dataset_folder_format["Calib_folder"]
            self.Depth_folder = self.dataset_root_path + "/" + subset_path + "/" + dataset_folder_format["Depth_folder"]
            if os.path.exists(self.Depth_folder) == False:
                # Fall back to default depth folder path if it does not exist
                depth_folder_fallback_path = self.dataset_root_path + "/" + subset_path + "/" + "Depth"
                print(f"{YELLOW}Depth_folder:{self.Depth_folder} does not exist{RESET}, fall back to path: {YELLOW}{depth_folder_fallback_path}{RESET}")
                self.Depth_folder = depth_folder_fallback_path
                assert os.path.exists(self.Depth_folder), f"{RED}Depth_folder:{self.Depth_folder} does not exist{RESET}"

            # load total data path list (calib, rgb, pcd)
            total_calib_path_list = zv.listFilesByExtension(self.Calib_folder,"json")
            total_rgb_path_list = zv.listFilesByExtension(self.RGB_folder, "png")
            total_pcd_path_list = zv.listFilesByExtension(self.PCloud_folder,"pcd")
            if ZV_Dataset._DEPTH_INPUT_TYPE == "depth_image":
                total_depth_path_list = zv.listFilesByExtension(self.Depth_folder,"png")
            else:
                total_depth_path_list = zv.listFilesByExtension(self.Depth_folder,"npz")
                if total_depth_path_list == []:
                    total_depth_path_list = zv.listFilesByExtension(self.Depth_folder,"npy")
                                   
            total_src_scene_num = len(total_rgb_path_list)
            
            calib_json_num = len(total_calib_path_list)
            assert calib_json_num > 0, "calib_json num should large than 0"
            if calib_json_num == 1:
                calib_json_single_path = total_calib_path_list[0]
                total_calib_path_list = []
                # load train calib path list
                for j in range(total_src_scene_num):
                    total_calib_path_list.append(calib_json_single_path)
                    
            samp_mode = subsets_samp_mode[i]
            path_dict["rgb_path_list"] += zv.selectEleFromListWithSampleMode(total_rgb_path_list, samp_mode)
            path_dict["pcd_path_list"] += zv.selectEleFromListWithSampleMode(total_pcd_path_list, samp_mode)
            path_dict["depth_path_list"] += zv.selectEleFromListWithSampleMode(total_depth_path_list, samp_mode)
            path_dict["calib_path_list"] += zv.selectEleFromListWithSampleMode(total_calib_path_list, samp_mode)
            
        
        # create datasets with given data paths
        p_dataset = ZV_Dataset(subsets_path_list,config=self.config,path_dict = path_dict,device = self.device)

        return p_dataset

# outside the class
def zvDatasetLoaderWrapper(dataset_instance:ZV_Dataset):
    dataset_instance.loadDataFromPaths()


