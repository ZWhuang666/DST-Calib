from depthReactify import*
from scipy import ndimage
from moge.model.v2 import MoGeModel # Let's try MoGe-2
import gc
import json
from scipy import stats
from collections import defaultdict
from tqdm import tqdm
import time
import platform


# ---------------------Depth Map Inference (Moge2) ------------------ #
def createMogeModel(checkpoint_path = ""):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    if checkpoint_path != "":
        Moge2_model = MoGeModel.from_pretrained(checkpoint_path).to(DEVICE)     
    else:
        # Load the model from huggingface hub (or load from local).
        if os.path.exists("./weights/Moge2/model.pt"):
            Moge2_model = MoGeModel.from_pretrained("./weights/Moge2/model.pt").to(DEVICE)             
        else:
            Moge2_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(DEVICE)    
    
    return Moge2_model

def inferImage_Moge(img_src,model, depth_channel = 1, input_channel = "CHW", output_channel = "CHW",save_depth = False):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if input_channel == "CHW":
        img_src = np.transpose(img_src,(1, 2, 0))
    
    # Infer 
    input_image = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)                       
    input_image_tensor = torch.tensor(input_image / 255, dtype=torch.float32, device=DEVICE).permute(2, 0, 1)    

    time1 = time.time()
    output = model.infer(input_image_tensor)
    time2 = time.time()
    """
    `output` has keys "points", "depth", "mask", "normal" (optional) and "intrinsics",
    The maps are in the same size as the input image. 
    {
        "points": (H, W, 3),    # point map in OpenCV camera coordinate system (x right, y down, z forward). For MoGe-2, the point map is in metric scale.
        "depth": (H, W),        # depth map
        "normal": (H, W, 3)     # normal map in OpenCV camera coordinate system. (available for MoGe-2-normal)
        "mask": (H, W),         # a binary mask for valid pixels. 
        "intrinsics": (3, 3),   # normalized camera intrinsics
    }
    """
    print("Moge infer time(s)",time2-time1)
    points_map = output["points"].detach().cpu().numpy()
    mask_map = points_map[:,:,2] > 60   
   
    points_map[mask_map] = [0,0,0] # set invalid value to zero
    
    print("infer depth shape:", points_map.shape)
    return points_map

def inferImageByPath_Moge(img_path,model, depth_channel = 1, input_channel = "CHW", output_channel = "CHW", save_depth = False):
    img_src= cv2.imread(img_path)
    points_map = inferImage_Moge(img_src,model, depth_channel, input_channel, output_channel, save_depth)
    return points_map

def inferImageFolder_Moge(img_folder,dst_folder,visual_process = False,replace_existed = False):
    Moge2_model = createMogeModel()
    img_path_list = zv.listFilesByExtension(img_folder,file_extension="png")
    if os.path.exists(dst_folder) == False:
        os.mkdir(dst_folder)
    
    #current_num = len(img_path_list)
    if replace_existed:
        current_num = 0
    else:
        current_num = len(zv.listFilesByExtension(dst_folder, file_extension="npz"))

    for i in range(current_num,len(img_path_list)):
        img_path = img_path_list[i]
        print(img_path)
        img_name = zv.getFileNameFromPath(img_path)
        points_map = inferImageByPath_Moge(img_path,Moge2_model,depth_channel = 1, input_channel = "HWC", output_channel = "HWC")
        points_map_path = dst_folder + "/" + img_name.split(".")[0]
        #np.save(points_map_path,points_map) # save points map (H,W,3)
        np.savez_compressed(points_map_path,data=points_map[:,:,2])    # save depth map (H,W)
        #zv.visualNpArrHWC(points_map[:,:,2],"points_map")
        if visual_process:
            points_xyz,rgb = createPointXYZRGBFromPointMapAndImgSrc_byPath(points_map_path + ".npy",img_path)
            pcd = createColorPointCloud(points_xyz,rgb)
            visualizePCDs([pcd])


def generatePCDFromImgPath(img_path:str,
                       pcd_path:str,
                       model,
                       K,
                       T_pred = np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]],dtype=np.float64),
                       visual_process = False,
                       print_time = False):
    
    time1 = time.time()
    # depth estimation and adjustment 
    img_src= cv2.imread(img_path) # H x W x 3 
    img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB) 
    points_map = inferImage_Moge(img_src,model, depth_channel = 1, input_channel = "HWC", output_channel = "HWC", save_depth = False) # H x W x 1
    depth = points_map[:,:,2] # get depth map from point map (H,W)
    
    
    proj_shape = (points_map.shape[0],points_map.shape[1])
    points_xyz, intensity = loadPCDFromPath(pcd_path)
    
    intensity = zv.arrNormalizedMaxMin(intensity, 0, 1)
    

    time2 = time.time()
    if print_time:
        print("create pcd 1 (s): ", time2-time1)
    
    # using base function for depth cloud generation to create depth cloud
    est_xyz,est_rgb = generateDepthCloudXYZRGBFromImg(img_src,points_xyz,intensity,depth,K,T_pred,
                                    visual_process=False,print_time = print_time)  
    
    
    # Rgion Cut for depth cloud and LiDAR point cloud
    if 1:
        est_xyz, est_rgb = cloudXYZRGBRegionCut(est_xyz,est_rgb,z_range=[4,np.inf])
        pass
    
    
    time3 = time.time()
    depth_correct_time = time3-time2
    if print_time:
        print("generateDepthCloudXYZRGB (s): ", time3-time2)
    
    T_eye = np.eye(4)
    if visual_process:
        est_xyz_lip = transform.projectPCloud2DepthMap(est_xyz,K,T_eye,projection_shape = proj_shape)
        visualDepthProjection(est_xyz_lip) # the backward projection after depth cloud reactify
    if print_time:
        time4 = time.time()
        print("projectPCloud2DepthMap (s): ", time4-time3)
    print("gen_pcd point num", est_xyz.shape[0])
    if print_time:
        time5 = time.time()
        print("Create PCD 2 (s): ", time5-time4)
        print("est pcd cost (s): ", time5-time1)
        
    
    return est_xyz,est_rgb,depth_correct_time

def generatePCDFromImg_ByFolder_AutoInitialGuess(img_folder:str,
                       pcd_folder:str,
                       dst_folder:str,
                       calib_folder:str,
                       model,
                       T_pred = np.array([None]),
                       t_perturb_range = 0.5,
                       euler_perturb_range = 5,
                       visual_process = False,
                       save_type = "depth_map",
                       replace_existed = False
                       ):
    
    calib_json_path_list = zv.listFilesByExtension(calib_folder,"json")
    img_path_list = zv.listFilesByExtension(img_folder,"png")
    total_pairs_num = len(img_path_list)
    pcd_path_list = zv.listFilesByExtension(pcd_folder,"pcd")[:total_pairs_num]
    
    assert len(img_path_list) == len(pcd_path_list), f"the number of imgs {len(img_path_list)} should equal to the number of pcd {len(pcd_path_list)}"
    if os.path.exists(dst_folder) == False:
        os.mkdir(dst_folder)

    assert len(calib_json_path_list) > 0, "calib json folder is empty"
    if len(calib_json_path_list) > 1:
        assert len(calib_json_path_list) == total_pairs_num, "If there are multiple calib json in the folder, its number should be the same with that of the point clouds"
    else:
        tmp_calib_json_path_list = []
        calib_json_path = calib_json_path_list[0]
        for i in range(total_pairs_num):
            tmp_calib_json_path_list.append(calib_json_path)
        calib_json_path_list = tmp_calib_json_path_list

    if replace_existed:
        current_num = 0
    else:
        current_num = len(zv.listFilesByExtension(dst_folder, file_extension="npz"))  # only count the number of depth map files in the folder

    if current_num >= total_pairs_num:
        print(f"All {total_pairs_num} pairs have been processed. No more files to process.")
        return 0.0  # Return 0.0 as no processing was done

    T_ini_list = []
    intrinsic_K_list = []     # intrinsic parameters for each image
    for i in range(total_pairs_num):
        calib_json_path = calib_json_path_list[i]
        intrinsic_K,T_lidar2cam,R_lidar2cam,t_lidar2cam = pose.getIntrinsicExtrinsicFromZVCalibJson(calib_json_path)
        T_gt = pose.Rt2T(R_lidar2cam,t_lidar2cam)
        T_ini_list.append(T_gt)
        intrinsic_K_list.append(intrinsic_K)
    
    T_ini_arr = np.stack(T_ini_list, axis=0)

    #if T_pred.any() == None:
    if any(x is None for x in T_pred.flat):
        print(f"{MAGENTA}T_pred is None, generate random T_pred by T_ini_arr{RESET}")
        T_pred_arr = pose.GenerateRandomT_byCenters(T_ini_arr,
                                                    t_disturb_range=[t_perturb_range*0.6,t_perturb_range*0.2,t_perturb_range*0.2],  # 0.3,0.1,0.1
                                                    euler_disturb_range=[euler_perturb_range*0.6,euler_perturb_range*0.2,euler_perturb_range*0.2]) # 3.0,1.0,1.0
    else:
        print(f"{MAGENTA}T_pred is given, generate using initial guess of T{RESET}")
        T_pred_list = [T_pred for _ in range(total_pairs_num)]
        T_pred_arr = np.stack(T_pred_list, axis=0)
        
    total_depth_correct_time = 0
    
    for i in range(current_num,total_pairs_num):
        img_path = img_path_list[i]
        pcd_path = pcd_path_list[i]
        current_T_pred = T_pred_arr[i]
        img_src = zv.loadImage(img_path,channel_type="HWC")
        src_proj_shape = (img_src.shape[0],img_src.shape[1])
        
        K = intrinsic_K_list[i]
        gen_xyz,gen_rgb,depth_correct_time = generatePCDFromImgPath(img_path,pcd_path,model,K,
                                             T_pred = current_T_pred,
                                             visual_process=visual_process,
                                             print_time = True)
        total_depth_correct_time += depth_correct_time
        if platform.system() == "Windows":
            gen_pcd_name = img_path.split("\\")[-1].split(".")[:-1][0]
        else:
            gen_pcd_name = img_path.split("/")[-1].split(".")[:-1][0]
        print(gen_pcd_name)
        gen_pcd_path = dst_folder + "/" + gen_pcd_name
        print(gen_pcd_path)
        
        if visual_process:
            # load points xyz and intensity from pcd file for visualization
            points_xyz, intensity = loadPCDFromPath(pcd_path)
            true_xyz = transform.np_transPCloud_T(points_xyz,T_gt) # target pcd using the gt transform
            true_colors = np.stack((intensity, intensity, intensity), axis=1).reshape(-1, 3)

            gen_pcd = createColorPointCloud(gen_xyz,gen_rgb)
            true_pcd = createColorPointCloud(true_xyz,true_colors)
            visualizePCDs([gen_pcd,true_pcd])

        if save_type == "depth_map":
            gen_depth_map = transform.projectPCloud2DepthMap(gen_xyz,K,T = zv.T_EYE_NP,projection_shape=src_proj_shape)
            print("gen_depth_map shape:", gen_depth_map.shape)
            np.savez_compressed(gen_pcd_path, data=gen_depth_map)
            if visual_process:
                load_depth_map = np.load(gen_pcd_path+".npz")['data']
                load_xyz,_ = transform.np_unproject_depth_to_pointcloud(load_depth_map,K,T=zv.T_EYE_NP,filt_zero_depth=0)
                load_xyz,load_rgb = transform.np_renderCloudWithRGBImage(load_xyz,img_src,K)
                load_pcd = createColorPointCloud(load_xyz,load_rgb)
                visualizePCDs([load_pcd,true_pcd])
        else:
            savePointCloud(gen_pcd, gen_pcd_path + ".pcd")

    avg_depth_correct_time = total_depth_correct_time/ (total_pairs_num - current_num)

    return avg_depth_correct_time
