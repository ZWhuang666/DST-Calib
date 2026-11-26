from utils import*
import torch
import torch.nn.functional as F
from torch import nn
from math import pi as PI
from collections.abc import Iterable
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class VCamera():
    def __init__(self,
                 img_shape:Iterable,
                 intrinsic:np.ndarray,
                 extrinsic:np.ndarray = np.eye(4)):
        # InTran (3,4) or (4,4)
        self.img_shape = img_shape
        
        self.extrinsic_T = extrinsic
        self.intrinsic_K = intrinsic

        if type(intrinsic) == np.ndarray:
            self.InTran = zv.arr2Tensor(intrinsic)
        elif type(intrinsic) == torch.Tensor:
            self.InTran = torch.clone(intrinsic)
        else:
            self.InTran = zv.arr2Tensor(intrinsic)
        
    
    def setImgShape(self,img_shape:Iterable):
        self.img_shape = self.img_shape

    def batch_projectLiDARDepth_withScale(self,ExTran:torch.Tensor,
                                           pcd:torch.Tensor, 
                                           scale: torch.Tensor,
                                           proj_channel = 3)->tuple:
        """transform pcd and project it to img

        Args:
            ExTran (torch.Tensor): B,4,4
            pcd (torch.Tensor): B,3,N (XYZ)
            scale (torch.Tensor): B,N (S)
        Returns:
            tuple: depth_img (B,C,H,W), transformed_pcd (B,3,N)
        """
    
        H,W = self.img_shape
        B = ExTran.size(0)
        self.InTran = self.InTran.to(pcd.device)
        
        pcd_trans = se3.transform(ExTran,pcd)  # [B,4,4] x [B,3,N] -> [B,3,N]
        # scale
        pcd_trans = pcd_trans*(scale.unsqueeze(1))
        

        depth = pcd_trans[:,2,:]
        proj_pcd = torch.bmm(self.InTran.repeat(B,1,1),pcd_trans) # [B,3,3] x [B,3,N] -> [B,3,N]
        proj_x = (proj_pcd[:,0,:]/proj_pcd[:,2,:]).type(torch.long)
        proj_y = (proj_pcd[:,1,:]/proj_pcd[:,2,:]).type(torch.long)
        rev = ((proj_x>=0)*(proj_x<W)*(proj_y>=0)*(proj_y<H)*(proj_pcd[:,2,:]>3)).type(torch.bool)  # [B,N]
        batch_ldp_img = torch.zeros(B,proj_channel,H,W,dtype=torch.float32).to(pcd.device)  # [B,H,W]
        
        for bi in range(B):
            rev_i = rev[bi,:]  # (N,)
            proj_xrev = proj_x[bi,rev_i]
            proj_yrev = proj_y[bi,rev_i]
            batch_ldp_img[bi,:,proj_yrev,proj_xrev] = depth[bi,rev_i]
        
        return batch_ldp_img, pcd_trans   # (B,1,H,W), (B,3,N)
    
    def batch_projectLiDARIntensity_Normal(self,ExTran:torch.Tensor,
                                           pcd:torch.Tensor, 
                                           intensity: torch.Tensor,
                                           proj_channel = 3)->tuple:
        """transform pcd and project it to img

        Args:
            ExTran (torch.Tensor): B,4,4
            pcd (torch.Tensor): B,3,N (XYZ)
            intensity (torch.Tensor): B,N (I)
        Returns:
            tuple: depth_img (B,C,H,W), transformed_pcd (B,3,N)
        """

        H,W = self.img_shape
        B = ExTran.size(0)
        
        self.InTran = self.InTran.to(pcd.device)
        pcd_trans = se3.transform(ExTran.to(pcd.device),pcd)  # [B,4,4] x [B,3,N] -> [B,3,N]
        proj_pcd = torch.bmm(self.InTran.repeat(B,1,1),pcd_trans) # [B,3,3] x [B,3,N] -> [B,3,N]
        proj_x = (proj_pcd[:,0,:]/proj_pcd[:,2,:]).type(torch.long)
        proj_y = (proj_pcd[:,1,:]/proj_pcd[:,2,:]).type(torch.long)
        rev = ((proj_x>=0)*(proj_x<W)*(proj_y>=0)*(proj_y<H)*(proj_pcd[:,2,:]>0)).type(torch.bool)  # [B,N]
        batch_lip_img = torch.zeros(B,proj_channel,H,W,dtype=torch.float32).to(pcd.device)  # [B,C,H,W]
        
        for bi in range(B):
            rev_i = rev[bi,:]  # (N,)
            proj_xrev = proj_x[bi,rev_i]
            proj_yrev = proj_y[bi,rev_i]

            batch_lip_img[bi,:,proj_yrev,proj_xrev] = intensity[bi,rev_i]
        
        return batch_lip_img, pcd_trans   # (B,C,H,W), (B,3,N)
    
    
    def batch_projectLiDARIntensity(self, ExTran: torch.Tensor, 
                                        pcd: torch.Tensor, intensity: torch.Tensor,
                                        proj_channel = 3,
                                        method = "normal"):
        return self.batch_projectLiDARIntensity_Normal(ExTran,pcd,intensity,proj_channel=proj_channel)


    def batch_projectLiDARDepth(self,ExTran:torch.Tensor,pcd:torch.Tensor,proj_channel = 1,max_z = 20)->tuple:
        """transform pcd and project it to img
        Args:
            ExTran (torch.Tensor): B,4,4
            pcd (torch.Tensor): B,3,N (XYZ)
        Returns:
            tuple: depth_img (B,proj_channel,H,W), transformed_pcd (B,3,N)
        """
        
        H,W = self.img_shape
        B = ExTran.size(0)
        self.InTran = self.InTran.to(pcd.device)
        
        pcd_trans = se3.transform(ExTran,pcd)  # [B,4,4] x [B,3,N] -> [B,3,N]
        depth = pcd_trans[:,2,:]
        proj_pcd = torch.bmm(self.InTran.repeat(B,1,1),pcd_trans) # [B,3,3] x [B,3,N] -> [B,3,N]
        proj_x = (proj_pcd[:,0,:]/proj_pcd[:,2,:]).type(torch.long)
        proj_y = (proj_pcd[:,1,:]/proj_pcd[:,2,:]).type(torch.long)
        rev = ((proj_x>=0)*(proj_x<W)*(proj_y>=0)*(proj_y<H)*(proj_pcd[:,2,:]>3)*(proj_pcd[:,2,:]<max_z)).type(torch.bool)  # [B,N]
        batch_ldp_img = torch.zeros(B,proj_channel,H,W,dtype=torch.float32).to(pcd.device)  # [B,H,W]
        
        for bi in range(B):
            rev_i = rev[bi,:]  # (N,)
            proj_xrev = proj_x[bi,rev_i]
            proj_yrev = proj_y[bi,rev_i]
            batch_ldp_img[bi,:,proj_yrev,proj_xrev] = depth[bi,rev_i]
        
        return batch_ldp_img, pcd_trans   # (B,1,H,W), (B,3,N)

    def batch_projectLiDARXYZ(self,ExTran:torch.Tensor,pcd:torch.Tensor)->tuple:
        """transform pcd and project it to img

        Args:
            ExTran (torch.Tensor): B,4,4
            pcd (torch.Tensor): B,3,N (XYZ)
        Returns:
            tuple: depth_img (B,3,H,W), transformed_pcd (B,3,N)
        """
        
        H,W = self.img_shape
        B = ExTran.size(0)
        self.InTran = self.InTran.to(pcd.device)
        pcd_trans = se3.transform(ExTran,pcd)  # [B,4,4] x [B,3,N] -> [B,3,N]
        xyz = pcd_trans[:,:,:]
        proj_pcd = torch.bmm(self.InTran.repeat(B,1,1),pcd_trans) # [B,3,3] x [B,3,N] -> [B,3,N]
        proj_x = (proj_pcd[:,0,:]/proj_pcd[:,2,:]).type(torch.long)
        proj_y = (proj_pcd[:,1,:]/proj_pcd[:,2,:]).type(torch.long)
        rev = ((proj_x>=0)*(proj_x<W)*(proj_y>=0)*(proj_y<H)*(proj_pcd[:,2,:]>0)).type(torch.bool)  # [B,N]
        batch_xyz_img = torch.zeros(B,3,H,W,dtype=torch.float32).to(pcd.device)  # [B,H,W]
        
        for bi in range(B):
            rev_i = rev[bi,:]  # (N,)
            proj_xrev = proj_x[bi,rev_i]
            proj_yrev = proj_y[bi,rev_i]
            batch_xyz_img[bi,:,proj_yrev,proj_xrev] = xyz[bi,:,rev_i]
        
        return batch_xyz_img, pcd_trans   # (B,3,H,W), (B,3,N)
    
    def renderLiDARIntensityOnImageTensor(self,ExTran:torch.Tensor,pcd:torch.Tensor,
                                    intensity:torch.Tensor, camera_image:torch.Tensor, z_thr = 1):
        """render a LiDAR intensity on camera image
        Args:
            ExTran (torch.Tensor): 4,4
            pcd (torch.Tensor): 3,N (XYZ)
            intensity (torch.Tensor): N (I)
            camera_image (torch.Tensor): 1,H,W
        Returns:
            tuple: rendered_img (H,W,1) np.ndarray
        """
        cam_image = camera_image.clone()
        C,H,W = cam_image.shape
        
        z_values = pcd[2, :]
        pcd_cut = pcd[:, z_values > z_thr]
        intensity_cut = intensity[z_values > z_thr]
        self.InTran = self.InTran.to(pcd.device)
        pcd_trans = se3.transform(ExTran.to(pcd.device),pcd_cut)  # [B,4,4] x [B,3,N] -> [B,3,N]
        proj_pcd = torch.matmul(self.InTran,pcd_trans) # [B,3,3] x [B,3,N] -> [B,3,N]
        proj_x = (proj_pcd[0,:]/proj_pcd[2,:]).type(torch.long)
        proj_y = (proj_pcd[1,:]/proj_pcd[2,:]).type(torch.long)
        rev = ((proj_x>=0)*(proj_x<W)*(proj_y>=0)*(proj_y<H)*(proj_pcd[2,:]>0)).type(torch.bool)  # [B,N]
        rendered_img = cam_image.to(pcd.device)  # [B,C,H,W]
        
        proj_xrev = proj_x[rev]
        proj_yrev = proj_y[rev]
        
        rendered_img[:,proj_yrev,proj_xrev] = intensity_cut[rev]*255
        
        rendered_img_arr = zv.convertTensor2ImageArr(rendered_img)
        
        return rendered_img_arr   # (H,W,1) np.ndarray
    
    def renderLiDARDepthOnImageTensorColor(self, ExTran: torch.Tensor, pcd: torch.Tensor,
                                camera_image: torch.Tensor, z_thr=1, point_size=1):
        """Render LiDAR depth on camera image with color mapping (blue-green-red) and adjustable point size.
        Args:
            ExTran (torch.Tensor): 4x4 extrinsic matrix
            pcd (torch.Tensor): 3xN point cloud (XYZ)
            camera_image (torch.Tensor): 1xHxW camera image tensor
            z_thr (float): Depth threshold for filtering points
            point_size (int): Size of each rendered point in pixels (square side length)
        Returns:
            np.ndarray: Rendered image (H, W, 3) with colorized depth
        """
        cam_image = camera_image.clone()
        C, H, W = cam_image.shape

        # Filter points by depth threshold
        z_values = pcd[2, :]
        pcd_cut = pcd[:, z_values > z_thr]
        depth_cut = pcd[2, z_values > z_thr]

        # Transform points to camera coordinates
        self.InTran = self.InTran.to(pcd.device)
        pcd_trans = se3.transform(ExTran.to(pcd.device), pcd_cut)  # [3, N]
        proj_pcd = torch.matmul(self.InTran, pcd_trans)  # [3, N]

        # Project points to image plane
        proj_x = (proj_pcd[0, :] / proj_pcd[2, :]).type(torch.long)
        proj_y = (proj_pcd[1, :] / proj_pcd[2, :]).type(torch.long)
        valid_mask = ((proj_x >= 0) & (proj_x < W) & 
                    (proj_y >= 0) & (proj_y < H) & 
                    (proj_pcd[2, :] > 0))

        proj_x_valid = proj_x[valid_mask]
        proj_y_valid = proj_y[valid_mask]
        depth_valid = depth_cut[valid_mask]

        # Normalize depth to [0, 1] for color mapping
        depth_min, depth_max = depth_valid.min(), depth_valid.max()
        depth_normalized = (depth_valid - depth_min) / (depth_max - depth_min + 1e-6)  # Avoid division by zero

        # Use matplotlib colormap (blue-green-red)
        cmap = plt.get_cmap('jet')  # or 'viridis', 'plasma', etc.
        colors = cmap(depth_normalized.cpu().numpy())[:, :3]  # RGBA -> RGB (ignore alpha)
        colors = torch.from_numpy(colors).to(torch.float32).to(pcd.device)  # [N, 3]
        
        # generate points' coordinates
        half = point_size // 2
        x_offsets = torch.arange(-half, half + 1, device=pcd.device)
        y_offsets = torch.arange(-half, half + 1, device=pcd.device)
        
        # calculate pixel coordinates [N, point_size^2]
        x_all = (proj_x_valid.unsqueeze(1) + x_offsets.unsqueeze(0)).clamp(0, W-1)
        y_all = (proj_y_valid.unsqueeze(1) + y_offsets.unsqueeze(0)).clamp(0, H-1)
        
        # Flatten the x and y pixel-coordinate arrays to [N * point_size^2].
        x_flat = x_all.reshape(-1)
        y_flat = y_all.reshape(-1)
        
        # expand color value corresponding to pixel number
        colors_expanded = colors.unsqueeze(1).expand(-1, point_size**2, -1).reshape(-1, 3)
        
        # Create output image
        rendered_img = cam_image.to(pcd.device)  # [B,C,H,W]  # 3-channel output
        rendered_img[:, y_flat, x_flat] = colors_expanded.T
        

        rendered_img = (rendered_img * 255)
        rendered_img_arr = zv.convertTensor2ImageArr(rendered_img)
        
        return rendered_img_arr

    def renderLiDARDepthOnImageTensor(self, ExTran: torch.Tensor, pcd: torch.Tensor,
                                camera_image: torch.Tensor, z_thr=1, point_size=1):
        """Render LiDAR depth on camera image with color mapping (blue-green-red) and adjustable point size.
        Args:
            ExTran (torch.Tensor): 4x4 extrinsic matrix
            pcd (torch.Tensor): 3xN point cloud (XYZ)
            camera_image (torch.Tensor): 1xHxW camera image tensor
            z_thr (float): Depth threshold for filtering points
            point_size (int): Size of each rendered point in pixels (square side length)
        Returns:
            np.ndarray: Rendered image (H, W, 3) with colorized depth
        """
        
        cam_image = camera_image.clone()
        cam_image = zv.tensorNormalizedMaxMin_perChannel(cam_image,0,255)
        

        C,H,W = cam_image.shape
        
        z_values = pcd[2, :]
        
        pcd_cut = pcd[:, z_values > z_thr]
        
        self.InTran = self.InTran.to(pcd.device)
        pcd_trans = se3.transform(ExTran.to(pcd.device),pcd_cut)  # [B,4,4] x [B,3,N] -> [B,3,N]
        depth = pcd_trans[2,:]
        depth_cut = depth
        
        proj_pcd = torch.matmul(self.InTran,pcd_trans) # [B,3,3] x [B,3,N] -> [B,3,N]
        proj_x = (proj_pcd[0,:]/proj_pcd[2,:]).type(torch.long)
        proj_y = (proj_pcd[1,:]/proj_pcd[2,:]).type(torch.long)
        rev = ((proj_x>=0)*(proj_x<W)*(proj_y>=0)*(proj_y<H)*(proj_pcd[2,:]>0)).type(torch.bool)  # [B,N]
        rendered_img = cam_image.to(pcd.device)  # [B,C,H,W]
        
        proj_xrev = proj_x[rev]
        proj_yrev = proj_y[rev]
        
        
        if depth_cut[rev].shape[0] > 0:
            rendered_img[2,proj_yrev,proj_xrev] = zv.tensorNormalizedMaxMin(depth_cut[rev])*255
            pass
        
        rendered_img_arr = zv.convertTensor2ImageArr(rendered_img)
        
        return rendered_img_arr   # (H,W,1) np.ndarray
    
    def __call__(self,ExTran:torch.Tensor,pcd:torch.Tensor):
        """transform pcd and project it to img

        Args:
            ExTran (torch.Tensor): B,4,4
            pcd (torch.Tensor): B,3,N

        Returns:
            tuple: depth_img (B,H,W), transformed_pcd (B,3,N)
        """
        assert len(ExTran.size()) == 3, 'ExTran size must be (B,4,4)'
        assert len(pcd.size()) == 3, 'pcd size must be (B,3,N)'
        return self.transform(ExTran,pcd)


if __name__ == "__main__":
    pass
    
