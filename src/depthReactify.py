import cv2
from utils import *
import open3d as o3d
import numpy as np
from pcloud import*
import random
from scipy.optimize import linear_sum_assignment, least_squares
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

def longest_monotonic_subsequence(points):
    n = len(points)
    dp = [1] * n
    prev = [-1] * n

    for i in range(1, n):
        for j in range(i):
            if points[i][1] >= points[j][1] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                prev[i] = j

    # find the end of the longest subsequence
    max_len = max(dp)
    idx = dp.index(max_len)

    lis = []
    while idx != -1:
        lis.append(points[idx])
        idx = prev[idx]
    return np.array(lis[::-1])
        
def depthAnchorsProcessing(input_depth_anchors, target_anchors_num, process_method = "Monotonic (M)",visual_process = False):
    """Select Detph Anchors and keep target_anchors_num achors
    Args:
        input_depth_anchors: List[List[float, float]]，each element is [cam_depth, lidar_depth]
        target_anchors_num: int，the output number of depth achors
    Returns:
        filtered_anchors: List[List[float, float]]
    """

    if process_method == "Monotonic (S)":
        anchors = np.array(input_depth_anchors)
        anchors_sorted = anchors[np.argsort(anchors[:, 0])]
        cam_depths = anchors_sorted[:, 0]
        lidar_depths = anchors_sorted[:, 1]

        # Step 1: select candidates via depth distribution
        min_depth, max_depth = cam_depths[0], cam_depths[-1]
        bins = np.linspace(min_depth, max_depth, target_anchors_num * 2 + 1)

        candidate_points = []
        for i in range(len(bins) - 1):
            bin_start, bin_end = bins[i], bins[i+1]
            mask = (cam_depths >= bin_start) & (cam_depths < bin_end if i < len(bins) - 2 else cam_depths <= bin_end)
            bin_points = anchors_sorted[mask]
            if len(bin_points) == 0:
                continue
            elif len(bin_points) == 1:
                candidate_points.append(bin_points[0])
            else:
                x = bin_points[:, 0]
                y = bin_points[:, 1]
                coeffs = np.polyfit(x, y, 1)
                y_fit = np.polyval(coeffs, x)
                residuals = np.abs(y - y_fit)
                best_idx = np.argmin(residuals)
                candidate_points.append(bin_points[best_idx])

        candidate_points = np.array(candidate_points)
        n = len(candidate_points)

        # Step 2: contruct subsequence that statisfies monotonic
        dp = [[] for _ in range(n)]

        for i in range(n):
            dp[i] = [candidate_points[i]]
            for j in range(i):
                pj, pi = candidate_points[j], candidate_points[i]
                if pi[0] > pj[0] and pi[1] >= pj[1]:
                    slope = (pi[1] - pj[1]) / (pi[0] - pj[0])
                    prev_seq = dp[j]
                    if len(prev_seq) < 2:
                        dp[i] = max(dp[i], dp[j] + [pi], key=len)
                    else:
                        last_slope = (prev_seq[-1][1] - prev_seq[-2][1]) / (prev_seq[-1][0] - prev_seq[-2][0])
                        if slope >= last_slope:
                            dp[i] = max(dp[i], dp[j] + [pi], key=len)

        
        best_seq = max(dp, key=len)

        # Step 3: uniform sampling if the anchors number exceeds the target number
        if len(best_seq) > target_anchors_num:
            sampled = [best_seq[0]]
            inner = best_seq[1:-1]
            idxs = np.linspace(0, len(inner)-1, target_anchors_num - 2, dtype=int)
            sampled += [inner[i] for i in idxs]
            sampled.append(best_seq[-1])
            final_anchors = np.array(sampled)
        else:
            final_anchors = np.array(best_seq)
            
        selected_anchors = final_anchors.tolist()
        
        if visual_process:
            plt.figure(figsize=(10, 6))
            plt.scatter(cam_depths, lidar_depths, color='red', label='Original Anchors', alpha=0.3)
            plt.scatter(candidate_points[:, 0], candidate_points[:, 1], color='blue', label='Candidate Anchors', alpha=0.5)
            plt.plot(final_anchors[:, 0], final_anchors[:, 1], color='green', marker='o', label='Final Anchors', linewidth=2)
            
            for i in range(1, len(final_anchors)):
                x0, y0 = final_anchors[i-1]
                x1, y1 = final_anchors[i]
                plt.arrow(x0, y0, x1-x0, y1-y0, head_width=0.005, color='green', alpha=0.6)
            
            plt.xlabel("Camera Depth (Normalized)")
            plt.ylabel("LiDAR Depth")
            plt.title("Final Anchors: Monotonic Depth & Increasing Slope")
            plt.legend()
            plt.grid(True)
            plt.show()
            
    elif process_method == "Monotonic (M)":
        anchors = np.array(input_depth_anchors)
        anchors_sorted = anchors[np.argsort(anchors[:, 0])]
        cam_depths = anchors_sorted[:, 0]
        lidar_depths = anchors_sorted[:, 1]

        # Step 1: select candidates via depth distribution
        min_depth, max_depth = cam_depths[0], cam_depths[-1]
        bins = np.linspace(min_depth, max_depth, target_anchors_num * 2 + 1)

        candidate_points = []
        for i in range(len(bins) - 1):
            bin_start, bin_end = bins[i], bins[i+1]
            mask = (cam_depths >= bin_start) & (cam_depths < bin_end if i < len(bins) - 2 else cam_depths <= bin_end)
            bin_points = anchors_sorted[mask]
            if len(bin_points) == 0:
                continue
            elif len(bin_points) == 1:
                candidate_points.append(bin_points[0])
            else:
                x = bin_points[:, 0]
                y = bin_points[:, 1]
                coeffs = np.polyfit(x, y, 1)
                y_fit = np.polyval(coeffs, x)
                residuals = np.abs(y - y_fit)
                best_idx = np.argmin(residuals)
                candidate_points.append(bin_points[best_idx])

        candidate_points = np.array(candidate_points)
        n = len(candidate_points)

        # Step 2: contruct subsequence that statisfies monotonic
        dp = [[] for _ in range(n)]

        for i in range(n):
            dp[i] = [candidate_points[i]]
            for j in range(i):
                pj, pi = candidate_points[j], candidate_points[i]
                if pi[0] > pj[0] and pi[1] >= pj[1]:
                    slope = (pi[1] - pj[1]) / (pi[0] - pj[0])
                    prev_seq = dp[j]
                    if len(prev_seq) < 2:
                        dp[i] = max(dp[i], dp[j] + [pi], key=len)
                    else:
                        last_slope = (prev_seq[-1][1] - prev_seq[-2][1]) / (prev_seq[-1][0] - prev_seq[-2][0])
                        if i < (n/2):
                            if slope >= last_slope and (slope / last_slope) < 1.2:
                                dp[i] = max(dp[i], dp[j] + [pi], key=len)
                        else:
                            if slope <= last_slope and (slope / last_slope) > 0.8:
                                dp[i] = max(dp[i], dp[j] + [pi], key=len)

        best_seq = max(dp, key=len)

        # Step 3: uniform sampling if the anchors number exceeds the target number
        if len(best_seq) > target_anchors_num:
            sampled = [best_seq[0]]
            inner = best_seq[1:-1]
            idxs = np.linspace(0, len(inner)-1, target_anchors_num - 2, dtype=int)
            sampled += [inner[i] for i in idxs]
            sampled.append(best_seq[-1])
            final_anchors = np.array(sampled)
        else:
            final_anchors = np.array(best_seq)
        
        selected_anchors:list = final_anchors.tolist()
        if 1 and len(selected_anchors) >= 3:
            d_out_thresh = 1.5
            d_anchor_0 = (selected_anchors[1][1] - selected_anchors[0][1])/(selected_anchors[1][0] - selected_anchors[0][0])  
            d_anchor_1 = (selected_anchors[2][1] - selected_anchors[1][1])/(selected_anchors[2][0] -selected_anchors[1][0]) 
            
            if (d_anchor_1 / d_anchor_0 > d_out_thresh):
                selected_anchors.remove(selected_anchors[0])

            selected_anchors
        final_anchors = np.array(selected_anchors)
           
        if visual_process:
            plt.figure(figsize=(10, 6))
            plt.scatter(cam_depths, lidar_depths, color='red', label='Original Anchors', alpha=0.3)
            plt.scatter(candidate_points[:, 0], candidate_points[:, 1], color='blue', label='Candidate Anchors', alpha=0.5)
            plt.plot(final_anchors[:, 0], final_anchors[:, 1], color='green', marker='o', label='Final Anchors', linewidth=2)
            
            for i in range(1, len(final_anchors)):
                x0, y0 = final_anchors[i-1]
                x1, y1 = final_anchors[i]
                plt.arrow(x0, y0, x1-x0, y1-y0, head_width=0.005, color='green', alpha=0.6)
            
            plt.xlabel("Camera Depth (Normalized)")
            plt.ylabel("LiDAR Depth")
            plt.title("Final Anchors: Monotonic Depth & Increasing Slope")
            plt.legend()
            plt.grid(True)
            plt.show()

   
    return selected_anchors

def depthAnchorSampling(input_depth_anchors, sample_num = 10, sample_method = "random"):
    '''Sampling depth anchors
    Args: 
        input_depth_anchors: sorted depth anchors (from near to far)
        sample_num: the number of anchors after sampling
        sample_method: random | uniform
    Return:
        sample_anchors (N,2)
    '''
    input_num = len(input_depth_anchors)
    sample_anchors = []
    if sample_method == "random":
        # random sampling
        for i in range(sample_num):
            ind = random.randint(0,input_num - 1)
            anchor = input_depth_anchors[ind]
            sample_anchors.append(anchor)
    elif sample_method == "uniform":
        # sort the input anchors
        gap = int(input_num / sample_num)
        ind = 0
        for i in range(sample_num):
            if ind < len(input_depth_anchors):
                anchor = input_depth_anchors[ind]
                sample_anchors.append(anchor)
                ind += gap
            else:
                break
    return sample_anchors

def DepthAnchorRefinement(depth, pcd, K, T_pred,proj_shape = (376, 1241),
                            anchors_sample_num = 400,
                            anchors_target_num = 16,
                            retain_zero_depth = True,
                            visual_anchor_process = False,print_debug = False):
    '''Depth Anchor Refinement
    Args:
        depth: (arr) estimated depth map from camera image  (H x W)
        pcd: Real point cloud capture by LiDAR  (N,3)
        K: intrinsic matrix of the camera
        T_pred: initial guess of T matrix that indicates the transformation from LiDAR to camera coordinate systems (4,4)
    Return:
        A Rectified Point Cloud with shape (N,3)
    '''
    H,W = proj_shape
    assert H == depth.shape[0] and W == depth.shape[1], "The shape of depth map and the shape of ldp should be the same!"
    
    ldp = transform.projectPCloud2DepthMap(pcd,K,T_pred,proj_shape)
    
    # depth valid processing
    ldp_valid_bound = ldp > 0
    depth_valid_bound = depth > 0
    valib_bound = ldp_valid_bound & depth_valid_bound
    
    ldp_valid = ldp[valib_bound].reshape(-1)
    depth_valid = depth[valib_bound].reshape(-1)
    if print_debug:
        print(ldp_valid.shape)
    valid_aligned_depth_pair = np.stack((depth_valid,ldp_valid),axis = 1)
    valid_num = valid_aligned_depth_pair.shape[0]
    if print_debug:
        print(valid_aligned_depth_pair.shape)
    
    # anchors sampling
    anchors_ini:list = sorted(valid_aligned_depth_pair.tolist(),key = lambda ach: ach[0])
    if print_debug:
        print("initial anchors", anchors_ini)
        print("initial anchors num", len(anchors_ini))
    anchors_sampled:list = depthAnchorSampling(anchors_ini, anchors_sample_num, sample_method="uniform")
    if print_debug:
        print("sampled anchors", anchors_sampled)
        print("sampled num ", len(anchors_sampled))
    
    # anchors filtering
    anchors = depthAnchorsProcessing(anchors_sampled,target_anchors_num=anchors_target_num,
                                     process_method="Monotonic (M)",
                                     visual_process=visual_anchor_process)
    
    if print_debug:
        print("anchors after Processing", anchors)
        print("anchors num after processing",len(anchors))
        
    # convert depth map to point cloud
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    i, j = np.meshgrid(np.arange(W), np.arange(H))
    z = depthAchorReactify(depth,anchors,extand_end=True)
    
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy
    
    convert_xyz = np.stack((x, y, z), axis=2).reshape(-1, 3) # (N,3)
    
    valid = (z > 0).reshape(-1)

    if retain_zero_depth:
        convert_xyz[~valid] = [0,0,0]
    else:
        convert_xyz = convert_xyz[valid]
    
    return convert_xyz,valid


def depthCloudPostProcessing(depth_cloud_xyz,depth_cloud_rgb,replace_filt_zero = True):
    '''Post Processing for depth cloud (xyz)
    Args:
        depth_cloud_xyz: xyz coordinate array of depth point cloud
        depth_cloud_rgb: rgb array of depth point cloud (normalized rgb color from 0-1)
    Return:
        processed_depth_cloud (N,3), processed_rgb (N,3)
    '''
    assert depth_cloud_xyz.shape[0] == depth_cloud_rgb.shape[0], "num of xyz should equal to num of rgb"
    # filtering by depth value
    depth_value_inbound = (depth_cloud_xyz[:,2] > 3).reshape(-1)

    if replace_filt_zero:
        depth_cloud_xyz[~depth_value_inbound] = [0,0,0]
    else:
        depth_cloud_xyz = depth_cloud_xyz[depth_value_inbound]
        depth_cloud_rgb = depth_cloud_rgb[depth_value_inbound]
    
    return depth_cloud_xyz,depth_cloud_rgb

def generateDepthCloudXYZRGBFromImg(rgb:np.ndarray,
                                ref_pcd_xyz:np.ndarray,
                                ref_pcd_intensity:np.ndarray,
                                depth:np.ndarray,
                                K,
                                T_pred = np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]],dtype=np.float64),
                                retain_zero_depth = True,
                                visual_process = False,
                                print_time = False):
    # depth estimation and adjustment 
    if rgb.shape[2] > 3:
        img_src = np.transpose(rgb,(1, 2, 0))
    else:
        img_src = rgb
        
    if len(depth.shape) > 2 and depth.shape[2] > 3:
        depth = np.transpose(depth,(1, 2, 0))
        if depth.shape[2] > 1:
            depth = depth[:,:,0]
    #img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB) 
    proj_shape = (depth.shape[0],depth.shape[1])
    
    rgb = img_src.reshape(-1, 3).astype(np.float32) / 255.0 # normalized color map
        
    # convert and reactify point cloud from depth map
    rectified_est_xyz,valid = DepthAnchorRefinement(depth, ref_pcd_xyz, K, T_pred, 
                                                 anchors_sample_num=400,
                                                 anchors_target_num=16,
                                                 proj_shape = proj_shape,
                                                 retain_zero_depth=retain_zero_depth,
                                                 visual_anchor_process=visual_process) # (N,3)
    if retain_zero_depth:
        rectified_rgb = rgb
    else:
        rectified_rgb = rgb[valid]
    
    # post processing for reactified point cloud
    est_xyz,est_rgb = depthCloudPostProcessing(rectified_est_xyz,rectified_rgb,replace_filt_zero=True)
    return est_xyz,est_rgb

def visualDepthProjection(ldp):
    depth_img = zv.arrNormalizedMaxMin(ldp,0,255).astype(np.uint8)
    print("depth_min", ldp.min())
    print("depth_max",ldp.max())
    cv2.imshow("depth_img",depth_img)
    cv2.waitKey(0)
    
def demo_projectPCloud2DepthMap(pcd_path):
    points_xyz, intensity = loadPCDFromPath(pcd_path)
    K = np.array([[1176.670168,0.000000,605.408551],[0.000000,1176.428331,487.497136],[0.000000,0.000000,1.000000]])
    T = np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]],dtype=np.float64)
    depth_arr = transform.projectPCloud2DepthMap(points_xyz,K,T,(800,1200))
    depth_img = zv.arrNormalizedMaxMin(depth_arr,0,255).astype(np.uint8)
    print("depth_min", depth_arr.min())
    print("depth_max",depth_arr.max())
    cv2.imshow("depth_img",depth_img)
    cv2.waitKey(0)

def depthAchorReactify(depth, anchors = [], extand_end = True):
    # generate depth min list
    depth_reactify = depth.copy()
    
    if anchors != [] and len(anchors) > 1:
        for i in range(1,len(anchors)):
            depth_ind = np.where((depth >= anchors[i-1][0]) & (depth <= anchors[i][0]))
            depth_reactify[depth_ind] = anchors[i-1][1] + (anchors[i][1] - anchors[i-1][1])*(depth[depth_ind] - anchors[i-1][0])/(anchors[i][0] - anchors[i-1][0])
        if extand_end:
            #last_d = (anchors[1][1] - anchors[0][1]) / (anchors[1][0]-anchors[0][0] + 0.000001)
            last_d = (anchors[-1][1] - anchors[-2][1]) / (anchors[-1][0]-anchors[-2][0] + 0.000001)
            depth_ind = np.where((depth >= anchors[-1][0]))
            depth_reactify[depth_ind] = anchors[-1][1] + last_d*(depth_reactify[depth_ind]-anchors[-1][0])
            last_d = (anchors[1][1] - anchors[0][1]) / (anchors[1][0]-anchors[0][0] + 0.000001)
            depth_ind = np.where((depth < anchors[0][0]))
            depth_reactify[depth_ind] = anchors[0][1] - last_d*(anchors[0][0] - depth_reactify[depth_ind])
        
    return depth_reactify    


if __name__ == "__main__":
    pass
