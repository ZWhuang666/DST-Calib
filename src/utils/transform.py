from . import se3,so3
from . import zutils as zv
import torch
import torch.nn.functional as F
import numpy as np
from math import pi as PI
from collections.abc import Iterable
import warnings
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def arr2RGB(values, cmap_name="viridis", vmin=None, vmax=None):
    '''convert 1D numpy array to RGB colors
    Args:
        values : np.ndarray 
            (N,)
        cmap_name : str
            matplotlib built-in colormap (such as 'viridis', 'plasma', 'jet', 'coolwarm')
        vmin : float, optional
            minimum mapping range
        vmax : float, optional
            maxminum mapping range

    Returns:
        np.ndarray
            (N, 3)，[R, G, B]，with range [0, 255]

    Support the following color mapping:
        ['magma', 'inferno', 'plasma', 'viridis', 'cividis', 'twilight', 'twilight_shifted', 'turbo', 'Blues', 'BrBG', 'BuGn', 
        'BuPu', 'CMRmap', 'GnBu', 'Greens', 'Greys', 'OrRd', 'Oranges', 'PRGn', 'PiYG', 'PuBu', 'PuBuGn', 'PuOr', 'PuRd', 
        'Purples', 'RdBu', 'RdGy', 'RdPu', 'RdYlBu', 'RdYlGn', 'Reds', 'Spectral', 'Wistia', 'YlGn', 'YlGnBu', 'YlOrBr', 
        'YlOrRd', 'afmhot', 'autumn', 'binary', 'bone', 'brg', 'bwr', 'cool', 'coolwarm', 'copper', 'cubehelix', 'flag', 
        'gist_earth', 'gist_gray', 'gist_heat', 'gist_ncar', 'gist_rainbow', 'gist_stern', 'gist_yarg', 'gnuplot', 'gnuplot2', 
        'gray', 'hot', 'hsv', 'jet', 'nipy_spectral', 'ocean', 'pink', 'prism', 'rainbow', 'seismic', 'spring', 'summer', 
        'terrain', 'winter', 'Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 
        'tab20b', 'tab20c', 'grey', 'gist_grey', 'gist_yerg', 'Grays', 'magma_r', 'inferno_r', 'plasma_r', 'viridis_r', 
        'cividis_r', 'twilight_r', 'twilight_shifted_r', 'turbo_r', 'Blues_r', 'BrBG_r', 'BuGn_r', 'BuPu_r', 'CMRmap_r', 
        'GnBu_r', 'Greens_r', 'Greys_r', 'OrRd_r', 'Oranges_r', 'PRGn_r', 'PiYG_r', 'PuBu_r', 'PuBuGn_r', 'PuOr_r', 'PuRd_r', 
        'Purples_r', 'RdBu_r', 'RdGy_r', 'RdPu_r', 'RdYlBu_r', 'RdYlGn_r', 'Reds_r', 'Spectral_r', 'Wistia_r', 'YlGn_r', 
        'YlGnBu_r', 'YlOrBr_r', 'YlOrRd_r', 'afmhot_r', 'autumn_r', 'binary_r', 'bone_r', 'brg_r', 'bwr_r', 'cool_r', 
        'coolwarm_r', 'copper_r', 'cubehelix_r', 'flag_r', 'gist_earth_r', 'gist_gray_r', 'gist_heat_r', 'gist_ncar_r', 
        'gist_rainbow_r', 'gist_stern_r', 'gist_yarg_r', 'gnuplot_r', 'gnuplot2_r', 'gray_r', 'hot_r', 'hsv_r', 'jet_r', 
        'nipy_spectral_r', 'ocean_r', 'pink_r', 'prism_r', 'rainbow_r', 'seismic_r', 'spring_r', 'summer_r', 'terrain_r', 
        'winter_r', 'Accent_r', 'Dark2_r', 'Paired_r', 'Pastel1_r', 'Pastel2_r', 'Set1_r', 'Set2_r', 'Set3_r', 'tab10_r', 
        'tab20_r', 'tab20b_r', 'tab20c_r']
    '''
    values = np.asarray(values).reshape(-1)
    if vmin is None:
        vmin = np.min(values)
    if vmax is None:
        vmax = np.max(values)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(norm(values))
    rgb = (rgba[:, :3] * 255).astype(np.uint8)

    return rgb
    
def pcd_projection(img_shape:tuple,intran:np.ndarray,pcd:np.ndarray,range:np.ndarray):
    """project pcd into depth img

    Args:
        img_shape (tuple): (H,W)
        intran (np.ndarray): (3x3)
        pcd (np.ndarray): (3xN)
        range (np.ndarray): (N,)

    Returns:
        u,v,r,rev: u,v,r (with rev) and rev
    """
    H,W = img_shape
    proj_pcd = intran @ pcd
    u,v,w = proj_pcd[0,:], proj_pcd[1,:], proj_pcd[2,:]
    u = np.asarray(u/w,dtype=np.int32)
    v = np.asarray(v/w,dtype=np.int32)
    rev = (0<=u)*(u<W)*(0<=v)*(v<H)*(w>0)
    u = u[rev]
    v = v[rev]
    r = range[rev]
    return u,v,r,rev


def binary_projection(img_shape:tuple,intran:np.ndarray,pcd:np.ndarray):
    """project pcd on img (binary mode)

    Args:
        img_shape (tuple): (H,W)
        intran (np.ndarray): (3x3)
        pcd (np.ndarray): (3,N)

    Returns:
        u,v,rev: u,v (without rev filter) and rev
    """
    H,W = img_shape
    proj_pcd = intran @ pcd
    u,v,w = proj_pcd[0,:], proj_pcd[1,:], proj_pcd[2,:]
    
    u = np.asarray(u/w,dtype=np.int64)
    v = np.asarray(v/w,dtype=np.int64)
    rev = (0<=u)*(u<W)*(0<=v)*(v<H)*(w>0)
    return u,v,rev

def projectPCloud2DepthMap(xyz,K,T,projection_shape = (800,1200),depth_arr_generate_method = "grayDirect"):
    
    points = xyz
    
    R = T[:3,:3]
    t = T[:3,3]
    transformed_points = (R @ points.T).T + t

    xyz_camera = transformed_points.T  # [3, N]
    xyz_valid = xyz_camera[2] > 0
    xyz_camera = xyz_camera[:,xyz_valid]
    uv_homogeneous = K @ xyz_camera  # [3, N]
    uv = uv_homogeneous[:2] / uv_homogeneous[2]  # [2, N]
    uv = uv.T  # [N, 2]

    uv_valid = uv
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        uv_valid = np.round(uv_valid).astype(int)
    
    depth_camera = xyz_camera[2] # [N]
    depth_valid = depth_camera
    
    image_shape = projection_shape
    depth_arr = np.zeros(image_shape,dtype=np.float32)
    
    in_bounds = (uv_valid[:, 0] >= 0) & (uv_valid[:, 0] < image_shape[1]) & \
                (uv_valid[:, 1] >= 0) & (uv_valid[:, 1] < image_shape[0])
    uv_valid = uv_valid[in_bounds]

    depth_valid = depth_valid[in_bounds]
    
    depth_arr[uv_valid[:, 1], uv_valid[:, 0]] = depth_valid.reshape(-1) # (H,W)
    
    return depth_arr

def projectPCloud2ValueMap(xyz,values, K,T,projection_shape = (800,1200),depth_arr_generate_method = "grayDirect"):
    '''project corresponded values of a point cloud to a map

    Args:
        xyz (np.ndarray): (N,3)
        values (np.ndarray): (N,D)
    Returns:
        values_arr (np.ndarray): (H,W,D)
    '''
    assert values.shape[0] == xyz.shape[0], "the shape of values and xyz should be the same"
    
    points = xyz
    
    R = T[:3,:3]
    t = T[:3,3]
    transformed_points = (R @ points.T).T + t

    xyz_camera = transformed_points.T  # [3, N]
    uv_homogeneous = K @ xyz_camera  # [3, N]
    uv = uv_homogeneous[:2] / uv_homogeneous[2]  # [2, N]
    uv = uv.T  # [N, 2]

    # z > 0
    valid_indices = xyz_camera[2] > 0
    uv_valid = uv[valid_indices]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        uv_valid = np.round(uv_valid).astype(int)
    values_valid  = values[valid_indices]
    
    
    image_shape = (projection_shape[0],projection_shape[1],values.shape[1])
    D = values.shape[1]
    values_arr = np.zeros(image_shape,dtype=np.float32)

    
    in_bounds = (uv_valid[:, 0] >= 0) & (uv_valid[:, 0] < image_shape[1]) & \
                (uv_valid[:, 1] >= 0) & (uv_valid[:, 1] < image_shape[0])
    uv_valid = uv_valid[in_bounds]

    values_valid = values_valid[in_bounds]
    
    values_arr[uv_valid[:, 1], uv_valid[:, 0]] = values_valid.reshape(-1,D) # (H,W,D)
    
    return values_arr

def projectPCloud2ValueMap_tensor(cloud_xyz:torch.Tensor,values:torch.Tensor, K,T,proj_shape = (800,1200)):
    """transform pcd and project it to img

    Args:
        T (torch.Tensor): 4,4
        cloud_xyz (torch.Tensor): 3,N (XYZ)
        values: N,D
    Returns:
        tuple: depth_img (B,proj_channel,H,W), transformed_pcd (B,3,N)
    """
    
    H,W = proj_shape
    proj_channel = values.shape[1]
    
    pcd_trans = se3.transform(T,cloud_xyz)  # [4,4] x [3,N] -> [3,N]
    depth = pcd_trans[2,:]
    proj_pcd = torch.matmul(K,pcd_trans) # [B,3,3] x [B,3,N] -> [B,3,N]
    proj_x = (proj_pcd[0,:]/proj_pcd[2,:]).type(torch.long)
    proj_y = (proj_pcd[1,:]/proj_pcd[2,:]).type(torch.long)
    rev = ((proj_x>=0)*(proj_x<W)*(proj_y>=0)*(proj_y<H)*(proj_pcd[2,:]>0)).type(torch.bool)  # [B,N]
    batch_ldp_img = torch.zeros(proj_channel,H,W,dtype=torch.float32).to(cloud_xyz.device)  # [B,H,W]
    
    proj_xrev = proj_x[rev]
    proj_yrev = proj_y[rev]
    values_transpose = values.permute(1, 0)
    batch_ldp_img[:,proj_yrev,proj_xrev] = values_transpose[rev]
    
    return batch_ldp_img, pcd_trans   # (B,1,H,W), (B,3,N)

def nptrans(pcd:np.ndarray,G:np.ndarray)->np.ndarray:
    R,t = G[:3,:3], G[:3,[3]]  # (3,3), (3,1)
    return R @ pcd + t

def np_transPCloud(pcd:np.ndarray,T:np.ndarray)->np.ndarray:
    """transform point cloud with given extrinsic parameters

    Args:
        pcd (np.ndarray): (3,N)
        T (np.ndarray): (4x4)
    Returns:
        R @ pcd + t

    """
    R,t = T[:3,:3], T[:3,[3]]  # (3,3), (3,1)
    return R @ pcd + t

def np_transPCloud_T(pcd:np.ndarray,T:np.ndarray)->np.ndarray:
    """transform point cloud with given extrinsic parameters

    Args:
        pcd (np.ndarray): (N,3)
        T (np.ndarray): (4x4)
    Returns:
        R @ pcd + t

    """
    R,t = T[:3,:3], T[:3,3]  # (3,3), (3,1)
    return (R @ pcd.T).T + t

def np_mapCloudToFixProjSize(points_xyz, K, T, proj_size):
    '''
    map a cloud to the size the proj size (H*W, 3)
    '''
    ldp = projectPCloud2DepthMap(points_xyz,K,T,projection_shape=proj_size)
    mapped_xyz,_ = np_unproject_depth_to_pointcloud(ldp,K,np.linalg.inv(T),filt_zero_depth=False)
    return mapped_xyz

def np_renderCloudWithRGBImage(points_xyz:np.ndarray,RGB_img, K, T:np.ndarray = np.eye(4)):
    '''Render Point Cloud with RGB image

    Args:
        points_xyz (np.ndarray): (N,3)
        RGB_img (np.ndarray): (H,W,3)
        K: intrinsic matrix of projection virtual camera
        T: extrinsic matrix from point cloud to RGB iamge
    '''
    assert RGB_img.shape[2] == 3, "The channel of RGB img should be three, and the shape of RGB image should be (H,W,3)"
    
    proj_shape = (RGB_img.shape[0],RGB_img.shape[1])
    
    ldp = projectPCloud2DepthMap(points_xyz,K,T,projection_shape=proj_shape)
    
    valid_points_xyz,valid_mask = np_unproject_depth_to_pointcloud(ldp,K,np.eye(4))
    rgb = RGB_img.reshape(-1,3)
    rgb = zv.arrNormalizedMaxMin(rgb,0,1)
    valid_rgb = rgb[valid_mask]
    
    return valid_points_xyz,valid_rgb
    
def np_renderImageWithCloudIntensity(points_xyz:np.ndarray,
                                     intensity:np.ndarray,
                                     RGB_img:np.ndarray, 
                                     K:np.ndarray, 
                                     T:np.ndarray,
                                     z_cut_thres = -10,
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

    lip = projectPCloud2ValueMap(points_xyz_valid,intensity_valid,K,T,projection_shape=proj_shape)  # (H,W,1)
    valid_lip_mask = lip[:,:,0] > 0
    rendered_rgb_img = np.copy(RGB_img)
    #zv.visualNpArrHWC(rendered_rgb_img)


    valid_intensity = lip[valid_lip_mask]
    rendering_rgb = arr2RGB(valid_intensity,cmap_name = cmap_name)
    #zv.visualNpArrHWC(lip)
    rendered_rgb_img[valid_lip_mask] = rendering_rgb
    #print(np.max(valid_intensity),np.min(valid_intensity))

    return rendered_rgb_img

def np_renderImageWithCloudDepth(points_xyz:np.ndarray,
                                RGB_img:np.ndarray, 
                                K:np.ndarray, 
                                T:np.ndarray,
                                cmap_name = "hot"):
    '''Render RGB image with intensity values of a point cloud

    Args:
        points_xyz (np.ndarray): (N,3)
        RGB_img (np.ndarray): (H,W,3)
        K: intrinsic matrix of projection virtual camera
        T: extrinsic matrix from point cloud to RGB iamge
    '''

    assert RGB_img.shape[2] == 3, "The channel of RGB img should be three, and the shape of RGB image should be (H,W,3)"
    proj_shape = (RGB_img.shape[0],RGB_img.shape[1])
    ldp = projectPCloud2DepthMap(points_xyz,K,T,projection_shape=proj_shape)  # (H,W)
    valid_ldp_mask = ldp[:,:] > 0
    rendered_rgb_img = np.copy(RGB_img)
    
    valid_depth = ldp[valid_ldp_mask]
    rendering_rgb = arr2RGB(valid_depth,cmap_name = cmap_name)
    rendered_rgb_img[valid_ldp_mask] = rendering_rgb

    return rendered_rgb_img

def np_unproject_depth_to_pointcloud(depth_map: np.ndarray,
                                  K: np.ndarray,
                                  T: np.ndarray = np.eye(4),
                                  filt_zero_depth = True) -> np.ndarray:
    '''Back projecting depth to point cloud
    Args:
        depth_map: H x W numpy array
        K: 3 x 3 numpy array
        T: (optional) 4 x 4 numpy array, represents the transformation from camera to world coordinate system
    
    Returns:
        points: N x 3 3D point cloud
    '''
    assert depth_map.ndim == 2, "depth_map must have size H x W"
    H, W = depth_map.shape

    # construct pixel grid (u,v)
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)

    # [u, v, 1]
    ones = np.ones_like(uu)
    pixels_hom = np.stack((uu, vv, ones), axis=-1).reshape(-1, 3)  # (N,3)

    # inv_K
    invK = np.linalg.inv(K)

    dirs = (invK @ pixels_hom.T).T  # (N,3)

    depths = depth_map.reshape(-1)  # (N,)

    # obtain 3D points subject to camera coordinate system
    points_cam = dirs * depths[:, np.newaxis]  # (N,3)
    if filt_zero_depth:
        points_cam_valid_mask = points_cam[:,2] > 0 
        points_cam = points_cam[points_cam_valid_mask]
    else:
        points_cam_valid_mask = points_cam[:,2] > 0 
    
    # transform to world coordinate system using the given T
    assert T.shape == (4, 4)
    points_cam_hom = np.concatenate([points_cam, np.ones((points_cam.shape[0], 1))], axis=1)  # (N,4)
    points_world_hom = (T @ points_cam_hom.T).T  # (N,4)
    points = points_world_hom[:, :3]

    return points,points_cam_valid_mask

def np_projectPCloudXYZ_Depth(img_shape:tuple,intran:np.ndarray,pcd:np.ndarray):
    """project pcd into depth img

    Args:
        img_shape (tuple): (H,W)
        intran (np.ndarray): (3x3)
        pcd (np.ndarray): (3xN)
        range (np.ndarray): (N,)

    Returns:
        projection (H,W)
    """
    H,W = img_shape
    proj_pcd = intran @ pcd
    u,v,w = proj_pcd[0,:], proj_pcd[1,:], proj_pcd[2,:]
    u = np.asarray(u/w,dtype=np.int32)
    v = np.asarray(v/w,dtype=np.int32)
    rev = (0<=u)*(u<W)*(0<=v)*(v<H)*(w>0)
    projection = np.array((H,W), dtype = np.float32)
    u = u[rev]
    v = v[rev]
    depth = w[rev]
    projection[u,v] = depth

    return projection
    
def np_projectPCloudXYZI_Intensity(img_shape:tuple,intran:np.ndarray,pcd:np.ndarray):
    """project pcd into depth img

    Args:
        img_shape (tuple): (H,W)
        intran (np.ndarray): (3x3)
        pcd (np.ndarray): (3xN)
        range (np.ndarray): (N,)

    Returns:
        u,v,r,rev: u,v,r (with rev) and rev
    """
    H,W = img_shape
    pcd_xyz = pcd[:3,:]
    intensity = pcd[3,:]

    proj_pcd = intran @ pcd_xyz
    
    u,v,w = proj_pcd[0,:], proj_pcd[1,:], proj_pcd[2,:]
    u = np.asarray(u/w,dtype=np.int32)
    v = np.asarray(v/w,dtype=np.int32)
    rev = (0<=u)*(u<W)*(0<=v)*(v<H)*(w>0)
    projection = np.array((H,W), dtype = np.float32)
    u = u[rev]
    v = v[rev]
    intensity = intensity[rev]
    projection[u,v] = intensity
    
    return projection

def np_projectPCloudXYZ_Intensity(img_shape:tuple,intran:np.ndarray,pcd:np.ndarray,intensity:np.ndarray):
    """project pcd into depth img

    Args:
        img_shape (tuple): (H,W)
        intran (np.ndarray): (3x3)
        pcd (np.ndarray): (3xN)
        intensity (np.ndarray): (N,)

    Returns:
        u,v,r,rev: u,v,r (with rev) and rev
    """
    H,W = img_shape
    pcd_xyz = pcd[:3,:]

    proj_pcd = intran @ pcd_xyz
    
    u,v,w = proj_pcd[0,:], proj_pcd[1,:], proj_pcd[2,:]
    u = np.asarray(u/w,dtype=np.int32)
    v = np.asarray(v/w,dtype=np.int32)
    rev = (0<=u)*(u<W)*(0<=v)*(v<H)*(w>0)
    projection = np.array((H,W), dtype = np.float32)
    u = u[rev]
    v = v[rev]

    intensity = intensity[rev]
    projection[u,v] = intensity
    
    return projection

def tensor_unproject_2d_to_3d(pixel_coords, depth_values, K):
    fx = K[:, 0, 0].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
    fy = K[:, 1, 1].unsqueeze(-1).unsqueeze(-1)
    cx = K[:, 0, 2].unsqueeze(-1).unsqueeze(-1)
    cy = K[:, 1, 2].unsqueeze(-1).unsqueeze(-1)
    
    u = pixel_coords[..., 0:1]  # (B, N, 1)
    v = pixel_coords[..., 1:2]
    z = depth_values
    
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return torch.cat([x, y, z], dim=-1)

def batch_projectLiDARDepth(InTran:torch.Tensor,ExTran:torch.Tensor,
                            pcd:torch.Tensor,proj_shape:tuple,proj_channel = 1)->tuple:
    """transform pcd and project it to img
    Args:
        ExTran (torch.Tensor): B,4,4
        pcd (torch.Tensor): B,3,N (XYZ)
    Returns:
        tuple: depth_img (B,proj_channel,H,W), transformed_pcd (B,3,N)
    """
    
    H,W = proj_shape
    B = ExTran.size(0)
    
    pcd_trans = se3.transform(ExTran,pcd)  # [B,4,4] x [B,3,N] -> [B,3,N]
    depth = pcd_trans[:,2,:]
    proj_pcd = torch.bmm(InTran,pcd_trans) # [B,3,3] x [B,3,N] -> [B,3,N]
    proj_x = (proj_pcd[:,0,:]/proj_pcd[:,2,:]).type(torch.long)
    proj_y = (proj_pcd[:,1,:]/proj_pcd[:,2,:]).type(torch.long)
    rev = ((proj_x>=0)*(proj_x<W)*(proj_y>=0)*(proj_y<H)*(proj_pcd[:,2,:]>0)).type(torch.bool)  # [B,N]
    batch_ldp_img = torch.zeros(B,proj_channel,H,W,dtype=torch.float32).to(pcd.device)  # [B,H,W]
    # size of rev_i is not constant so that a batch-formed operdation cannot be applied
    
    for bi in range(B):
        rev_i = rev[bi,:]  # (N,)
        proj_xrev = proj_x[bi,rev_i]
        proj_yrev = proj_y[bi,rev_i]
        batch_ldp_img[bi,:,proj_yrev,proj_xrev] = depth[bi,rev_i]
    
    return batch_ldp_img, pcd_trans   # (B,1,H,W), (B,3,N)
