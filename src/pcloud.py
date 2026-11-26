import open3d as o3d
import argparse
import cv2
from utils import*
from config import*
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy import ndimage
from scipy.interpolate import griddata
from scipy.linalg import svd
from scipy.spatial.transform import Rotation as Scipy_R
from tqdm import tqdm
import torch
import torch.nn.functional as F
import kornia.morphology as morph
import kornia.contrib as contrib
import os
from pathlib import Path
import random
from scipy.ndimage import convolve


# ----------------- IO -------------------------
def savePointCloud(pcd: o3d.geometry.PointCloud, filename: str):
    '''Save input pcd 
    '''
    # support .ply or .pcd format
    if not filename.endswith(('.ply', '.pcd')):
        raise ValueError("Only .ply and .pcd formats are supported.")
    o3d.io.write_point_cloud(filename, pcd)

def loadPointCloud(filename: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(filename)
    if not pcd.has_colors():
        print("Warning: Loaded point cloud does not contain color information.")
    return pcd

def loadXYZRGBFromPointCloud(filename: str) -> tuple:
    pcd = o3d.io.read_point_cloud(filename)  # support .ply/.pcd/.xyz
    
    points = np.asarray(pcd.points)    # (N, 3)，x, y, z
    colors = np.asarray(pcd.colors)    # (N, 3)，[0,1]
    return points,colors

def loadXYZRGBFromPCD(pcd) -> tuple:
    
    points = np.asarray(pcd.points)    # (N, 3)，x, y, z
    colors = np.asarray(pcd.colors)    # (N, 3)，[0,1]
    return points,colors

def loadPointsMap(filepath:str):
    file_format = filepath.split(".")[-1]
    if file_format == "npz":
        #print("current file",filepath)
        points_map = np.load(filepath)['data']
    else:
        points_map = np.load(filepath)
    return points_map

def loadDepthMap(filepath:str):
    points_map_arr = loadPointsMap(filepath)
    points_map_arr_shape = points_map_arr.shape
    assert len(points_map_arr_shape) >= 2, "points_map_arr_shape should be like (H,W,X), where X >= 0"
    if len(points_map_arr_shape) > 2:
        if points_map_arr_shape[2] > 1:
            # (H,W,3) -> (H,W)
            depth_map = points_map_arr[:,:,2]
        else:
            # (H,W,1) -> (H,W)
            depth_map = points_map_arr[:,:,0]
    else:
        # (H,W) -> (H,W)
        depth_map = points_map_arr[:,:]
    
    return depth_map

def extractDepthMapFromPointsMap(points_map_arr:np.ndarray):
    '''Extract Depth Map From Points Map
    Args:
        points_map_arr: (H,W,X), X >= 0
    '''
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
    return depth_map

def savePCD_XYZI(pcd_dst_path,points_xyz,intensity,ascii = True):
    '''
    '''
    output_pcd = o3d.t.geometry.PointCloud()
    output_pcd.point.positions = o3d.core.Tensor(points_xyz)
    output_pcd.point.intensity = o3d.core.Tensor(intensity.reshape(-1,1))
    
    o3d.t.io.write_point_cloud(
        pcd_dst_path,
        output_pcd,
        write_ascii=ascii,
        compressed=False
    )
    print(f"save pcd success at: {pcd_dst_path}")

def RenderCloudByRGB_Np(points_xyz:np.ndarray, Tmat:np.ndarray, K:np.ndarray, rgb:np.ndarray):
    """
    Args：
        points_xyz: (N, 3)
        Tmat: (4, 4)
        K: (3, 3)
        rgb: (C, H, W)
    Returns：
        colored point cloud (N, 6) [x, y, z, r, g, b]
    """
    
    points_homogeneous = np.hstack([points_xyz, np.ones((len(points_xyz), 1))])
    
    points_cam = (Tmat @ points_homogeneous.T).T[:, :3]
    
    # Z>0
    valid_z = points_cam[:, 2] > 0
    points_valid = points_cam[valid_z]
    orig_indices = np.where(valid_z)[0]
    
    # project to pixel
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x = (fx * points_valid[:, 0] / points_valid[:, 2]) + cx
    y = (fy * points_valid[:, 1] / points_valid[:, 2]) + cy
    u = np.round(x).astype(int)
    v = np.round(y).astype(int)
    
    
    H, W = rgb.shape[1], rgb.shape[2]
    valid_u = (u >= 0) & (u < W)
    valid_v = (v >= 0) & (v < H)
    valid_uv = valid_u & valid_v
    
    colors = np.zeros((len(points_xyz), 3), dtype=np.uint8)
    
    rgb_img = rgb
    if valid_uv.any():
        final_indices = orig_indices[valid_uv]
        colors[final_indices,0] = rgb_img[2,v[valid_uv], u[valid_uv]]
        colors[final_indices,1] = rgb_img[1,v[valid_uv], u[valid_uv]]
        colors[final_indices,2] = rgb_img[0,v[valid_uv], u[valid_uv]]
        
    return np.hstack([points_xyz, colors])

def SavePCD_XYZRGB(filename, points_xyzrgb):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyzrgb[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(points_xyzrgb[:, 3:] / 255.0)
    o3d.io.write_point_cloud(filename, pcd)

def SavePCD_XYZRGB_channelSplit(filename, points_xyz,rgb):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    pcd.colors = o3d.utility.Vector3dVector(zv.arrNormalizedMaxMin(rgb,0,1))
    o3d.io.write_point_cloud(filename, pcd)

def LoadPCD_XYZRGB(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    points = np.asarray(pcd.points).astype(np.float32)
    colors = (np.asarray(pcd.colors) * 255).astype(np.float32)
    return np.hstack([points, colors])

def LoadPCD_XYZRGB_channelSplit(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    return points,colors

def LoadPCD_XYZI(filepath):
    point_cloud_t = o3d.t.io.read_point_cloud(filepath)  
    points_xyz = point_cloud_t.point["positions"].numpy()  # N x 3
    intensity = point_cloud_t.point["intensity"].numpy()  # N

    return points_xyz,intensity

def convertPCD_XYZRGB_2_XYZI(points_xyzrgb):
    points_xyz = points_xyzrgb[:,:3].astype(np.float32)
    rgb = points_xyzrgb[:,3:].astype(np.float32)
    intensity = rgb[:,0].reshape(-1).astype(np.float32)
    return points_xyz,intensity

def convert_PCD_XYZRGB_2_XYZI_byScene(root_folder):
    scene_folder_list = os.listdir(root_folder)
    dst_folder = root_folder + "/output/"
    os.makedirs(dst_folder, exist_ok=True)
    for scene_folder_name in scene_folder_list:
        scene_folder_path = os.path.join(root_folder,scene_folder_name)
        # list pcd in the folder
        scene_pcd_path_list = zv.listFilesByExtension(scene_folder_path,"pcd")
        for scene_pcd_path in scene_pcd_path_list:
            print(scene_pcd_path)
            scene_pcd_name = zv.getFileNameFromPath(scene_pcd_path)
            save_dist_path = os.path.join(scene_folder_path,scene_pcd_name + "_convertInt.pcd")
            points_xyzrgb = LoadPCD_XYZRGB(scene_pcd_path)
            print("read pcd rgb")
            points_xyz, intensity = convertPCD_XYZRGB_2_XYZI(points_xyzrgb)
            print("covert pcd rgb")
            savePCD_XYZI(os.path.join(dst_folder,scene_folder_name + "_" + scene_pcd_name),points_xyz,intensity,ascii = True)
            print("savePCD_XYZI to path: ",save_dist_path)
        scene_rgb_path_list = zv.listFilesByExtension(scene_folder_path,"png")
        for scene_rgb_path in scene_rgb_path_list:
            scene_rgb_name = zv.getFileNameFromPath(scene_rgb_path)
            dst_rgb_path = os.path.join(dst_folder,scene_folder_name + "_" + scene_rgb_name)
            zv.copyFile(scene_rgb_path,dst_rgb_path)


def convert_feather_to_pcd(feather_dir, pcd_dir,ascii = False):
    '''convert feather format point cloud files to PCD format
    '''
    os.makedirs(pcd_dir, exist_ok=True)
    feather_files = list(Path(feather_dir).glob('*.feather'))
    
    for feather_path in tqdm(feather_files, desc="convert format"):
        # read feather
        df = pd.read_feather(feather_path)
        points_xyz = df[['x', 'y', 'z']].values # （N，3）
        intensity = df[['intensity']].values # （N）
        print("points_xyz",points_xyz)
        print("intensity",intensity)
        # create Open3D object
        output_pcd = o3d.t.geometry.PointCloud()
        output_pcd.point.positions = o3d.core.Tensor(np.array(points_xyz,dtype = np.float32))
        output_pcd.point.intensity = o3d.core.Tensor(np.array(intensity,dtype = np.float32).reshape(-1,1))
        
        pcd_dst_path = Path(pcd_dir) / f"{feather_path.stem}.pcd"

        o3d.t.io.write_point_cloud(
            str(pcd_dst_path),
            output_pcd,
            write_ascii=ascii,
            compressed=False
        )

        print(f"save pcd success at: {pcd_dst_path}")

def getPCDFormat(pcd_file_path):
    with open(pcd_file_path, 'rb') as f:
        for line in f:
            try:
                line_str = line.decode('utf-8').strip().lower()
            except:
                continue

            if line_str.startswith('data'):
                if 'ascii' in line_str:
                    return 'ascii'
                elif 'binary_compressed' in line_str:
                    return 'binary_compressed'
                elif 'binary' in line_str:
                    return 'binary'
                else:
                    return 'unknown'
    return 'unknown'

def loadPCDFromPath(pcd_path):
    format = getPCDFormat(pcd_path)
    #print(format)
    if format == "ascii":
        with open(pcd_path, 'r') as f:
            lines = f.readlines()
        # locate beginning position of data
        for i, line in enumerate(lines):
            if line.strip() == 'DATA ascii':
                data_start = i + 1
                break
        # analyze point 
        point_data = np.loadtxt(lines[data_start:])
        # assume the structure is: x y z intensity
        points_xyz = point_data[:, :3]
        intensity = point_data[:, 3]
    else:
        point_cloud_t = o3d.t.io.read_point_cloud(pcd_path)  
        points_xyz = point_cloud_t.point["positions"].numpy()  # N x 3
        intensity = point_cloud_t.point["intensity"].numpy()  # N  
    return points_xyz,intensity

def loadAndVisualPCDFromPath(pcd_path):
    points_xyz,intensity = loadPCDFromPath(pcd_path)
    colors = np.stack((intensity, intensity, intensity), axis=1).reshape(-1, 3)
    visualizeColorPointCloud(points_xyz,colors)
    
def convertPCDFormat(
    input_file: str,
    output_file: str,
    ascii: bool = True,
    keep_intensity: bool = True
) -> None:
    '''Open3D Tensor API PCD convertion （bin/ASCII），reserve intensity information（XYZI）
    
    Args:
        input_file (str): input PCD file path
        output_file (str): output PCD file path
        ascii (bool): True is ASCII，False is binary
        keep_intensity (bool): if keeping the intensity information
    '''
    
    pcd = o3d.t.io.read_point_cloud(input_file)
    
    
    if not pcd.is_empty():
        
        output_pcd = o3d.t.geometry.PointCloud()
        
        output_pcd.point.positions = pcd.point.positions
        
        if keep_intensity and 'intensity' in pcd.point:
            output_pcd.point.intensity = pcd.point.intensity
        elif keep_intensity and 'colors' in pcd.point:
            colors = pcd.point.colors.numpy()
            intensity = colors[:, 0:1]  # extract R channel as intensity
            output_pcd.point.intensity = o3d.core.Tensor(intensity)
        
        o3d.t.io.write_point_cloud(
            output_file,
            output_pcd,
            write_ascii=ascii,
            compressed=False
        )
        
        print(f"convertion success: {(input_file)} -> {(output_file)}")
    else:
        raise ValueError("cannot read the point cloud，please check the file format")

def pcdConvert2ASCII_byFolder(pcd_folder,dst_folder):
    pcd_path_list = zv.listFilesByExtension(pcd_folder,file_extension="pcd")

    if os.path.exists(dst_folder) == False:
        os.mkdir(dst_folder)
    
    for pcd_path in pcd_path_list:
        pcd_name = os.path.basename(pcd_path)
        dst_path = dst_folder + "/" + pcd_name
        convertPCDFormat(pcd_path,dst_path,ascii=True,keep_intensity=True)


def PCloudXYZI_DownSample(points_xyz, intensity,downsample_ratio):
    assert points_xyz.shape[0] == intensity.shape[0], "num of points xyz should equal to num of intensity"
    
    # select random point
    num_points = len(points_xyz)
    indices = np.random.choice(num_points, int(num_points * downsample_ratio), replace=False)

    points_xyz_down = points_xyz[indices]
    intensity_down = intensity[indices]
    
    return points_xyz_down, intensity_down

def downsamplePCD_byFolder(pcd_folder,dst_folder,downsample_ratio = 0.1,ascii = True):
    pcd_path_list = zv.listFilesByExtension(pcd_folder,file_extension="pcd")
    
    if os.path.exists(dst_folder) == False:
        os.mkdir(dst_folder)
    
    for pcd_path in pcd_path_list:
        pcd_name = os.path.basename(pcd_path)
        dst_path = dst_folder + "/" + pcd_name
        points_xyz,intensity = loadPCDFromPath(pcd_path)
        points_xyz,intensity = PCloudXYZI_DownSample(points_xyz, intensity,downsample_ratio = downsample_ratio)
        savePCD_XYZI(dst_path,points_xyz,intensity, ascii = ascii)
        

# ---------------------Point Cloud Create and Visualization ---------------- #
def createPointXYZFromPointMap(point_map):
    points_xyz = point_map.reshape(-1,3)
    return points_xyz

def createRGBFromImgSrc(img_src):
    rgb = img_src.reshape(-1,3)/255
    return rgb

def createPointXYZRGBFromPointMapAndImgSrc(point_map,img_src):
    points_xyz = createPointXYZFromPointMap(point_map)
    rgb = createRGBFromImgSrc(img_src)
    # filter out z<=0
    mask = (points_xyz[:,2] > 0)
    points_xyz = points_xyz[mask]
    rgb = rgb[mask]
    return points_xyz,rgb

def createPointXYZRGBFromPointMapAndImgSrc_byPath(point_map_path,img_src_path):
    point_map = np.load(point_map_path)['data']
    img_src= cv2.imread(img_src_path) # H x W x 3 
    img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB) 
    points_xyz,rgb = createPointXYZRGBFromPointMapAndImgSrc(point_map,img_src)
    return points_xyz,rgb

def createCoordinateColoredPointCloud(pcd):
    xyz = np.asarray(pcd.points)
    xyz_norm = (xyz - xyz.min(0)) / (xyz.max(0) - xyz.min(0) + 1e-8)
    pcd_coord_colored = o3d.geometry.PointCloud()
    pcd_coord_colored.points = o3d.utility.Vector3dVector(xyz)
    pcd_coord_colored.colors = o3d.utility.Vector3dVector(xyz_norm)
    return pcd_coord_colored

def createXYZPointCloud(xyz:np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def createColorPointCloud(xyz:np.ndarray,colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def visualizeXYZPointCloud(xyz:np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    visualizePCD(pcd)
    
def visualizeColorPointCloud(xyz:np.ndarray,rgb:np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    visualizePCD(pcd)
    
def visualizePCD(pcd):
    print("Showing point cloud")
    o3d.visualization.draw_geometries([pcd])

def visualizePCDs(pcds:list):
    print("Showing point cloud")
    o3d.visualization.draw_geometries(pcds)

def RenderPCloudOnImage(pcloud,intrinsic_K,extrinsic_R,extrinsic_t,rendered_src_image):
    
    points = pcloud.point["positions"].numpy()  
    intensities = pcloud.point["intensity"].numpy()  
    
    transformed_points = (extrinsic_R @ points.T).T + extrinsic_t
   
    xyz_camera = transformed_points.T  # [3, N]
    uv_homogeneous = intrinsic_K @ xyz_camera  # [3, N]
    uv = uv_homogeneous[:2] / uv_homogeneous[2]  # [2, N]
    uv = uv.T  # [N, 2]

    
    valid_indices = xyz_camera[2] > 0  # z > 0
    uv_valid = uv[valid_indices]  # get valid u v
    uv_valid = np.round(uv_valid).astype(int)  
    depth_camera = xyz_camera[2] # [N]
    depth_valid  = depth_camera[valid_indices]

    
    intensity_valid = intensities[valid_indices]  # get valid intensity values
    intensity_normalized = np.clip(intensity_valid / intensity_valid.max(), 0, 1)  # normalize to [0, 1]
    intensity_colors = (intensity_normalized * 255).astype(np.uint8)  # convert to gray
    
    
    image_shape = (800, 1200)
    render_image = np.zeros(image_shape, dtype=np.uint8)
    depth_image = np.zeros(image_shape, dtype=np.uint8)
    depth_arr = np.zeros(image_shape,dtype=np.float32)
    depth_arr.fill(-1)

    
    in_bounds = (uv_valid[:, 0] >= 0) & (uv_valid[:, 0] < image_shape[1]) & \
                (uv_valid[:, 1] >= 0) & (uv_valid[:, 1] < image_shape[0])
    uv_valid = uv_valid[in_bounds]

    
    intensity_colors = intensity_colors[in_bounds]

    render_image[uv_valid[:, 1], uv_valid[:, 0]] = intensity_colors.reshape(-1)  

    return render_image


def create_gaussian_kernel(size=5, sigma=1.0):
    """gaussian convolution
    """
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * 
                     np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

def max_convolution_dense(cloud_map, kernel_size=3):
    """max convolution
    """
    # create conv kernal
    kernel = np.ones((kernel_size, kernel_size))
    
    # execute
    max_convolved = ndimage.grey_dilation(
        cloud_map, 
        size=(kernel_size, kernel_size), 
        mode='constant', 
        cval=0
    )
    
    return max_convolved

def renderImageWithCloudIntensity(points_xyz:np.ndarray,
                                intensity:np.ndarray,
                                RGB_img:np.ndarray, 
                                K:np.ndarray, 
                                T:np.ndarray,
                                z_cut_thres = 0.5,
                                densify_kernal = (0,0),
                                cmap_name = "hot"):
    '''Render RGB image with intensity values of a point cloud

    Args:
        points_xyz (np.ndarray): (N,3)
        intensity (np.ndarray): (N,1) or (N)
        RGB_img (np.ndarray): (H,W,3)
        K: intrinsic matrix of projection virtual camera
        T: extrinsic matrix from point cloud to RGB iamge
    '''

    assert RGB_img.shape[2] == 3, "The channel of RGB img should be three, and the shape of RGB image should be (H,W,3)"
    proj_shape = (RGB_img.shape[0],RGB_img.shape[1])
    intensity = intensity.reshape(-1,1)

    z_cut_mask = points_xyz[:,2] > z_cut_thres
    points_xyz_valid = points_xyz[z_cut_mask]
    intensity_valid = intensity[z_cut_mask]
      
    lip = transform.projectPCloud2ValueMap(points_xyz_valid,intensity_valid,K,T,projection_shape=proj_shape).reshape(proj_shape[0],proj_shape[1])  # (H,W,1)
    if densify_kernal == (0,0):
        lip_dense = lip
    else:
        lip_dense = max_convolution_dense(lip,kernel_size=densify_kernal[0])
        

    valid_lip_mask = lip_dense[:,:] > 0
    rendered_rgb_img = np.copy(RGB_img)
    #zv.visualNpArrHWC(rendered_rgb_img)
    
    valid_intensity = lip_dense[valid_lip_mask]
    rendering_rgb = transform.arr2RGB(valid_intensity,cmap_name = cmap_name)
    #zv.visualNpArrHWC(lip)
    rendered_rgb_img[valid_lip_mask] = rendering_rgb
    #print(np.max(valid_intensity),np.min(valid_intensity))

    return rendered_rgb_img

def renderImageWithCloudDepth(points_xyz:np.ndarray,
                            RGB_img:np.ndarray, 
                            K:np.ndarray, 
                            T:np.ndarray,
                            z_cut_thres = 0.5,
                            densify_kernal = (0,0),
                            cmap_name = "hot"):
    '''Render RGB image with intensity values of a point cloud

    Args:
        points_xyz (np.ndarray): (N,3)
        intensity (np.ndarray): (N,1) or (N)
        RGB_img (np.ndarray): (H,W,3)
        K: intrinsic matrix of projection virtual camera
        T: extrinsic matrix from point cloud to RGB iamge
    '''

    assert RGB_img.shape[2] == 3, "The channel of RGB img should be three, and the shape of RGB image should be (H,W,3)"
    proj_shape = (RGB_img.shape[0],RGB_img.shape[1])

    z_cut_mask = points_xyz[:,2] > z_cut_thres
    points_xyz_valid = points_xyz[z_cut_mask]
    
    lip = transform.projectPCloud2DepthMap(points_xyz_valid,K,T,projection_shape=proj_shape).reshape(proj_shape[0],proj_shape[1])  # (H,W,1)
    if densify_kernal == (0,0):
        lip_dense = lip
    else:
        #lip_dense = depthCompletionMorphInterp_nodilate(lip,ksize = densify_kernal)
        #lip_dense = convolution_diffusion(lip,kernel_size=densify_kernal[0],iterations=1)
        lip_dense = max_convolution_dense(lip,kernel_size=densify_kernal[0])
        #lip_dense = binary_dilation_dense(lip,kernel_size=3)

    valid_lip_mask = lip_dense[:,:] > 0
    rendered_rgb_img = np.copy(RGB_img)
    #zv.visualNpArrHWC(rendered_rgb_img)
    
    valid_intensity = lip_dense[valid_lip_mask]
    rendering_rgb = transform.arr2RGB(valid_intensity,cmap_name = cmap_name)
    #zv.visualNpArrHWC(lip)
    rendered_rgb_img[valid_lip_mask] = rendering_rgb
    #print(np.max(valid_intensity),np.min(valid_intensity))

    return rendered_rgb_img

def PCloudXYZI_Sample(points_xyz, intensity, sample_num):
    """Downsampling the point cloud
    Args:
        points_xyz: N x 3 numpy array
        intensity: N numpy array
        sample_num: int
    
    Returns:
        downsampled_xyz: sample_num x 3 numpy array
        downsampled_intensity: sample_num numpy array
    """
    intensity = np.asarray(intensity).reshape(-1)
    current_num = points_xyz.shape[0]
    
    if current_num > sample_num:
        indices = random.sample(range(current_num), sample_num)
        downsampled_xyz = points_xyz[indices]
        downsampled_intensity = intensity[indices]
    
    elif current_num < sample_num:
        downsampled_xyz = np.zeros((sample_num, 3), dtype=points_xyz.dtype)
        downsampled_intensity = np.zeros((sample_num,), dtype=intensity.dtype)
        
        downsampled_xyz[:current_num] = points_xyz
        downsampled_intensity[:current_num] = intensity
    
    else:
        downsampled_xyz = points_xyz
        downsampled_intensity = intensity
    
    return downsampled_xyz, downsampled_intensity




def cloudBioCutMax(cloud_xyz_1: np.ndarray, cloud_xyz_2: np.ndarray, cut_axis=[0,1,1],cut_dir = [1,1,1],replace_zero = False):
    '''
    Cut two point clouds to a shared coordinate range along specified axes.

    Args:
        cloud_xyz_1: (N1, 3)
        cloud_xyz_2: (N2, 3)
        cut_axis: list of 3 ints (0/1), specify x/y/z axes to apply cut

    Returns:
        cutted_xyz_1: (M1, 3)
        cutted_xyz_2: (M2, 3)
        mask1: np.ndarray bool (N1,) — mask for cloud_xyz_1
        mask2: np.ndarray bool (N2,) — mask for cloud_xyz_2
    '''
    
    cloud1_max = np.max(cloud_xyz_1, axis=0)
    cloud2_max = np.max(cloud_xyz_2, axis=0)
    
    cut_thres = np.minimum(cloud1_max, cloud2_max)  # length-3 array [x_th, y_th, z_th]

    
    mask1 = np.ones(len(cloud_xyz_1), dtype=bool)
    mask2 = np.ones(len(cloud_xyz_2), dtype=bool)

    
    for axis in range(3):
        if cut_axis[axis]:
            th = cut_thres[axis]
            mask1 &= ((cut_dir[axis]*cloud_xyz_1[:, axis]) <= th)
            mask2 &= ((cut_dir[axis]*cloud_xyz_2[:, axis]) <= th)

    if replace_zero:
        cutted_xyz_1 = cloud_xyz_1.copy()
        cutted_xyz_1[~mask1] = [0,0,0]
        cutted_xyz_2 = cloud_xyz_2.copy()
        cutted_xyz_2[~mask2] = [0,0,0]
    else:
        cutted_xyz_1 = cloud_xyz_1[mask1]
        cutted_xyz_2 = cloud_xyz_2[mask2]
    
    return cutted_xyz_1, cutted_xyz_2, mask1, mask2

def cloudBioCutMinMax(cloud_xyz_1: np.ndarray, cloud_xyz_2: np.ndarray, cut_axis=[0,1,1],cut_dir = [1,1,1]):
    '''
    Cut two point clouds to a shared coordinate range along specified axes.

    Args:
        cloud_xyz_1: (N1, 3)
        cloud_xyz_2: (N2, 3)
        cut_axis: list of 3 ints (0/1), specify x/y/z axes to apply cut

    Returns:
        cutted_xyz_1: (M1, 3)
        cutted_xyz_2: (M2, 3)
        mask1: np.ndarray bool (N1,) — mask for cloud_xyz_1
        mask2: np.ndarray bool (N2,) — mask for cloud_xyz_2
    '''
    cloud1_max = np.max(cloud_xyz_1, axis=0)
    cloud2_max = np.max(cloud_xyz_2, axis=0)
    
    cut_thres_max = np.minimum(cloud1_max, cloud2_max)  # length-3 array [x_th, y_th, z_th]

    cloud1_min = np.min(cloud_xyz_1, axis=0)
    cloud2_min = np.min(cloud_xyz_2, axis=0)
    
    cut_thres_min = np.maximum(cloud1_min, cloud2_min)  # length-3 array [x_th, y_th, z_th]
    
    mask1 = np.ones(len(cloud_xyz_1), dtype=bool)
    mask2 = np.ones(len(cloud_xyz_2), dtype=bool)

    for axis in range(3):
        if cut_axis[axis]:
            th = cut_thres_max[axis]
            mask1 &= ((cloud_xyz_1[:, axis]) <= th)
            mask2 &= ((cloud_xyz_2[:, axis]) <= th)

            th = cut_thres_min[axis]
            mask1 &= ((cloud_xyz_1[:, axis]) >= th)
            mask2 &= ((cloud_xyz_2[:, axis]) >= th)

    cutted_xyz_1 = cloud_xyz_1[mask1]
    cutted_xyz_2 = cloud_xyz_2[mask2]
    
    return cutted_xyz_1, cutted_xyz_2, mask1, mask2

def cloudXYZIRegionCut(points_xyz, intensity,
                   x_range = [-np.inf, np.inf],
                   y_range = [-np.inf, np.inf],
                   z_range = [-np.inf, np.inf],):
    assert points_xyz.shape[0] == intensity.shape[0], "num of points xyz should equal to num of intensity"
    
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range

    mask = (points_xyz[:, 0] >= x_min) & (points_xyz[:, 0] <= x_max) & \
        (points_xyz[:, 1] >= y_min) & (points_xyz[:, 1] <= y_max) & \
        (points_xyz[:, 2] >= z_min) & (points_xyz[:, 2] <= z_max)

    points_xyz_cropped = points_xyz[mask]
    intensity_cropped = intensity[mask]
    
    return points_xyz_cropped, intensity_cropped

def cloudXYZRegionCut(points_xyz, 
                   x_range = [-np.inf, np.inf],
                   y_range = [-np.inf, np.inf],
                   z_range = [-np.inf, np.inf],
                   replace_zero = False):
    
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range

    mask = (points_xyz[:, 0] >= x_min) & (points_xyz[:, 0] <= x_max) & \
        (points_xyz[:, 1] >= y_min) & (points_xyz[:, 1] <= y_max) & \
        (points_xyz[:, 2] >= z_min) & (points_xyz[:, 2] <= z_max)

    if replace_zero:
        points_xyz_cropped = points_xyz[:,:]
        points_xyz_cropped[~mask] = [0,0,0]
    else:
        points_xyz_cropped = points_xyz[mask]
    
    return points_xyz_cropped

def PCloudXYZ_Sample(points_xyz, sample_num):
    '''Downsampling the point cloud
    Args:
        points_xyz: N x 3 numpy array
        sample_num: int
    Returns:
        downsampled_xyz: sample_num x 3 numpy array
        downsampled_intensity: sample_num numpy array
    '''
    current_num = points_xyz.shape[0]
    
    
    if current_num > sample_num:
        indices = random.sample(range(current_num), sample_num)
        downsampled_xyz = points_xyz[indices]
    elif current_num < sample_num:
        downsampled_xyz = np.zeros((sample_num, 3), dtype=points_xyz.dtype)
        
        downsampled_xyz[:current_num] = points_xyz
    else:
        downsampled_xyz = points_xyz

    return downsampled_xyz

def PCloudXYZS_Sample(points_xyz, scale_arr, sample_num):
    '''Downsampling the point cloud
    Args:
        points_xyz: N x 3 numpy array
        scale_arr: N numpy array
        sample_num: int
    Returns:
        downsampled_xyz: sample_num x 3 numpy array
        downsampled_intensity: sample_num numpy array
    '''
    current_num = points_xyz.shape[0]
    
    if current_num > sample_num:
        indices = random.sample(range(current_num), sample_num)
        downsampled_xyz = points_xyz[indices]
        down_sampled_scale = scale_arr[indices]
    elif current_num < sample_num:
        downsampled_xyz = np.zeros((sample_num, 3), dtype=points_xyz.dtype)
        downsampled_xyz[:current_num] = points_xyz

        down_sampled_scale = np.ones((sample_num), dtype=scale_arr.dtype)
        down_sampled_scale[:current_num] = scale_arr
    else:
        downsampled_xyz = points_xyz
        down_sampled_scale  = scale_arr
    return downsampled_xyz,down_sampled_scale

def PCloudXYZRGB_Sample(points_xyz, rgb, sample_num):
    '''Downsampling the point cloud
    Args:
        points_xyz: N x 3 numpy array
        rgb: N x 3 numpy array
        sample_num: int
    Returns:
        downsampled_xyz: sample_num x 3 numpy array
        downsampled_intensity: sample_num numpy array
    '''
    current_num = points_xyz.shape[0]
    
    if current_num > sample_num:
        indices = random.sample(range(current_num), sample_num)
        downsampled_xyz = points_xyz[indices]
        downsampled_rgb = rgb[indices]
    elif current_num < sample_num:
        downsampled_xyz = np.zeros((sample_num, 3), dtype=points_xyz.dtype)
        downsampled_rgb = np.zeros((sample_num,3), dtype=rgb.dtype)
        
        downsampled_xyz[:current_num] = points_xyz
        downsampled_rgb[:current_num] = rgb
    else:
        downsampled_xyz = points_xyz
        downsampled_rgb = rgb
    
    return downsampled_xyz, downsampled_rgb


def depthCompletionMorphInterp_nodilate(Ds, mask=None, ksize = (7,7), fill_hole=False):
    
    if mask is None:
        mask = (~np.isnan(Ds)) & (Ds != 0)
    D = Ds.copy()
    D[~mask] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize = ksize)
    closed = cv2.morphologyEx((D>0).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    # only fill holes
    hole = (closed.astype(bool)) & (~(D>0))
    # distanceTransform
    mask_valid = (D>0).astype(np.uint8)
    dist, idx = cv2.distanceTransformWithLabels(1-mask_valid, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    flat_idx = idx[hole]
    ys, xs = np.where(mask_valid)
    coords = np.column_stack((ys, xs))
    nearest = coords[flat_idx - 1]  # labels start from 1
    D[hole] = D[nearest[:,0], nearest[:,1]]

    if fill_hole:
        h, w = Ds.shape
        ys, xs = np.indices((h, w))
        valid = D > 0
        pts = np.column_stack((xs[valid], ys[valid]))
        vals = D[valid]
        pts_empty = np.column_stack((xs[~valid], ys[~valid]))
        if len(pts) and len(pts_empty):
            interp = griddata(pts, vals, pts_empty, method='linear')
            nan = np.isnan(interp)
            if nan.any():
                nn = griddata(pts, vals, pts_empty[nan], method='nearest')
                interp[nan] = nn
            D[~valid] = interp

    return D

def cloudXYZRGBRegionCut(points_xyz, points_rgb,
                   x_range = [-np.inf, np.inf],
                   y_range = [-np.inf, np.inf],
                   z_range = [-np.inf, np.inf],
                   replace_zero = False):
    assert points_xyz.shape[0] == points_rgb.shape[0], "num of points xyz should equal to num of rgb"
    
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range

    
    mask = (points_xyz[:, 0] >= x_min) & (points_xyz[:, 0] <= x_max) & \
        (points_xyz[:, 1] >= y_min) & (points_xyz[:, 1] <= y_max) & \
        (points_xyz[:, 2] >= z_min) & (points_xyz[:, 2] <= z_max)

    if replace_zero:
        points_xyz_cropped = points_xyz.copy()
        points_rgb_cropped = points_rgb.copy()
        points_xyz_cropped[~mask] = [0,0,0]
    else:
        points_xyz_cropped = points_xyz[mask]
        points_rgb_cropped = points_rgb[mask]
    return points_xyz_cropped,points_rgb_cropped 


def cloudXYZDensify_byVCameraView(cloud_xyz,K,T = np.eye(4),proj_shape = (376, 1241),dense_ksize = (9,9),method = "nodilate"):
    ldp_src:np.ndarray = transform.projectPCloud2DepthMap(cloud_xyz,K,T,projection_shape = proj_shape)
    if dense_ksize == (0,0):
        ldp_src_dense = ldp_src
    else:
        if method == "nodilate":
            ldp_src_dense = depthCompletionMorphInterp_nodilate(ldp_src,ksize = dense_ksize)
        else:
            ldp_src_dense = max_convolution_dense(ldp_src,kernel_size=dense_ksize[0])
        
    
    dense_xyz,ldp_src_valid_mask = transform.np_unproject_depth_to_pointcloud(ldp_src_dense,K,np.linalg.inv(T),filt_zero_depth=False)
    return dense_xyz

def cloudXYZProjAndDensify(cloud_xyz,K,T = np.eye(4),proj_shape = (376, 1241),dense_ksize = (9,9),method = "nodilate"):
    ldp_src:np.ndarray = transform.projectPCloud2DepthMap(cloud_xyz,K,T,projection_shape = proj_shape)
    if dense_ksize == (0,0):
        ldp_src_dense = ldp_src
    else:
        if method == "nodilate":
            ldp_src_dense = depthCompletionMorphInterp_nodilate(ldp_src,ksize = dense_ksize)
        else:
            ldp_src_dense = max_convolution_dense(ldp_src,kernel_size=dense_ksize[0])

    return ldp_src_dense

    
def generateDiffMap_torch_batch(src_ldp:torch.Tensor,tgt_ldp:torch.Tensor,diff_thres = [1.0,1.0]):
    '''Generate Difference Map from the input projections
    Args:
        src_ldp: (B, 1, H, W)
        tgt_ldp: (B, 1, H, W)
        K: (B,3,3)
        diff_thres: 
    '''
    assert src_ldp.shape == tgt_ldp.shape, "Shape of src ldp and tgt ldp should be the same"
    B = src_ldp.shape[0]
    input_proj_shape = (src_ldp.shape[2],src_ldp.shape[3])
    
    sc_src_ldp = src_ldp[:,0,:,:].squeeze(1) # (B,H,W)
    sc_tgt_ldp = tgt_ldp[:,0,:,:].squeeze(1) # (B,H,W)
    diff_depth = sc_tgt_ldp - sc_src_ldp # (B,H,W)
    
    src_pos_mask = (sc_src_ldp > 0)
    tgt_pos_mask = (sc_tgt_ldp > 0)
    diff_valid_mask = ((diff_depth > -diff_thres[0]) & (diff_depth < diff_thres[1]) & src_pos_mask & (sc_tgt_ldp > 0))
    diff_valid_mask_neg = ((diff_depth < -diff_thres[0]) & src_pos_mask)
    diff_valid_mask_pos = ((diff_depth > diff_thres[1]) & src_pos_mask)
    
    diff_map = torch.zeros(size = (B,3,input_proj_shape[0],input_proj_shape[1])).to(diff_depth) 
    
    diff_map[:,0,:,:] = sc_src_ldp                                       
    diff_map[:,1,:,:][tgt_pos_mask] = sc_tgt_ldp[tgt_pos_mask]                                  
    diff_map[:,2,:,:][diff_valid_mask_neg] = -sc_src_ldp[diff_valid_mask_neg]   
    diff_map[:,2,:,:][diff_valid_mask_pos] = sc_src_ldp[diff_valid_mask_pos]   
    
    return diff_map
    
def depthCloudScale(ldp_src,ldp_ref,
                    K,proj_shape = (376, 1241),
                    scale_diff_thres = [0.02,0.02],
                    scale_ratio = 0.8,
                    densify_ref = False,
                    extract_from_ref_valid = True,
                    filt_zero_depth = True,
                    visual_process = False):
    '''Scaling the depth cloud
    Args:
        ldp_src: (H,W)
        ldp_ref: (H,W)
        K: (3,3)
        proj_shape: (H,W)
        scale_ratio: (1)
        
    Return:
        dst_cloud_xyz: (N,3)
    '''
    
    # create scale arr
    scale_map = np.ones(proj_shape)  #(H,W)

    if visual_process:
        zv.visualNpArrHWC(ldp_ref,"ldp_ref")
    
    # 
    if densify_ref:
        ldp_ref = depthCompletionMorphInterp_nodilate(ldp_ref,ksize = (7,7))
    
    # 
    src2ref_diff_map = ldp_ref - ldp_src
    
    # extract valid mask
    diff_valid_mask = ((src2ref_diff_map[:,:] > -scale_diff_thres[0]) & (src2ref_diff_map[:,:] < scale_diff_thres[1]) & (ldp_src[:,:] > 1) & (ldp_ref[:,:] > 1))
    
    ref_depth = ldp_ref[diff_valid_mask]
    ldp_src_dst = ldp_src.copy() 
    src_depth = ldp_src_dst[diff_valid_mask]
    dst_depth = src_depth + (ref_depth-src_depth)*scale_ratio

    ldp_src_dst[diff_valid_mask] = dst_depth
    dst_scale = src_depth / ldp_src_dst[diff_valid_mask]
    scale_map[diff_valid_mask] = dst_scale

    if extract_from_ref_valid:
        ldp_ref_valid_mask = ldp_ref[:,:] > 0
        ldp_src_dst[~ldp_ref_valid_mask] = 0
    
    
    dst_cloud_xyz,valid_mask = transform.np_unproject_depth_to_pointcloud(ldp_src_dst,K,np.eye(4),filt_zero_depth=filt_zero_depth)
    
    if filt_zero_depth:
        dst_scale_map = scale_map.reshape(-1)   # (H*W)
        dst_scale_map = dst_scale_map[valid_mask]
    else:
        dst_scale_map = scale_map.reshape(-1)

    return dst_cloud_xyz, dst_scale_map

if __name__ == "__main__":
    pass
