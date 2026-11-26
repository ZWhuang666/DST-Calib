#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
from PIL import Image
import torch
from torchvision import transforms
import json
import numpy as np
import cv2
import random
from scipy.ndimage import median_filter
import shutil
import csv
import re

R_LVINI_NP = np.array([[0,-1,0],[0,0,-1],[1,0,0]],dtype=np.float64)
T_LVINI_NP = np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]],dtype=np.float64)
T_EYE_NP = np.eye(4,dtype=np.float64)

def compactJsonArrays(json_str, keys=[]):
    """Format JSON arrays under selected keys in compact form."""
    for key in keys:
        # Locate the start of the keyed array.
        start_str = f'"{key}": ['
        key_start = json_str.find(start_str)
        if key_start == -1:
            continue
            
        # Locate the matching closing bracket.
        bracket_count = 1
        pos = key_start + len(start_str)
        key_end = pos
        
        while bracket_count > 0 and pos < len(json_str):
            if json_str[pos] == '[':
                bracket_count += 1
            elif json_str[pos] == ']':
                bracket_count -= 1
            pos += 1
            if bracket_count == 0:
                key_end = pos
                break
        
        # Extract the array section.
        array_section = json_str[key_start:key_end]
        
        # Remove whitespace from the array section.
        compact_array = re.sub(r'\s+', '', array_section)
        
        # Replace the original section with its compact form.
        json_str = json_str[:key_start] + compact_array + json_str[key_end:]
    
    return json_str

def copyFile(source, destination):
    """
    Copy a file with ``shutil.copy``.

    Args:
        source: Path to the source file.
        destination: Destination directory or complete destination filename.
    """
    try:
        # Preserve the source filename when the destination is a directory.
        if os.path.isdir(destination):
            dest_path = shutil.copy(source, destination)
        else:
            # Use the supplied filename when the destination is a file path.
            dest_path = shutil.copy(source, destination)
        
        print(f"file is successfully copied to: {dest_path}")
        return dest_path
    except Exception as e:
        print(f"copy fail: {e}")
        return None

def copyFiles_Gap(src_folder,dst_folder, start_ind = 0, end_ind = -1, gap = 1,file_extension = "*",print_info = False):
    '''
    copy files from src folder to dst folder, by a setted gap
    '''
    all_files_name_list = os.listdir(src_folder)
    #print(all_files_name_list)
    # get initial files name with selected file extension
    files_name_list = []
    for filename in all_files_name_list:
        if file_extension == "*":
            files_name_list.append(filename)
        if filename.endswith(file_extension):
            files_name_list.append(filename)
    
    initial_file_num = len(files_name_list)
    
    if end_ind == -1:
        stop_ind = initial_file_num - 1
    else:
        if end_ind > initial_file_num - 1:
            stop_ind =  initial_file_num - 1
        else:
            stop_ind = end_ind
    
    if print_info:
        #print("initial files num: ",initial_file_num)
        #print("start_ind : ", start_ind)
        #print("stop_ind : ",stop_ind)
        pass

    #print(files_name_list)
    
    src_file_path = []
    dst_file_path = []
    for i in range(start_ind, stop_ind + 1):
        #print(i)
        if (i-start_ind)%gap == 0:
            src_file_path.append(src_folder + "/" + files_name_list[i])
            dst_file_path.append(dst_folder + "/" + files_name_list[i])
    
    if print_info:
        print(src_file_path)
        print(dst_file_path)

    if os.path.exists(dst_folder) == False:
        os.mkdir(dst_folder)        
    
    for i in range(len(src_file_path)):
        shutil.copy2(src_file_path[i], dst_file_path[i])     

def record2CSV_singleLine(line_data_list,csv_path):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(line_data_list)  # Write one row from the list.

def record2CSV(records_list,title,csv_path):
    needTitle=False
    if not os.path.exists(csv_path):
        needTitle=True
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if needTitle:
            writer.writerow(title)
        writer.writerow(records_list)  # Write one row from the list.

def record2CSV_byDict(in_dict,key_name_list,csv_path):
    records_list = []
    for key in key_name_list:
        records_list.append(in_dict[key])
    record2CSV(records_list,key_name_list,csv_path=csv_path)
    
def vnorm(v):
    return np.linalg.norm(v)
    pass

def FlattenFeatureMap(feature_map):
    """
    Flatten a feature map into a sequence.

    Args:
        feature_map: Tensor shaped ``(batch_size, channels, height, width)``.
    Returns:
        Tensor shaped ``(batch_size, sequence_length, feature_dim)``.
    """
    batch_size, channels, height, width = feature_map.shape
    feature_map = feature_map.view(batch_size, channels, -1)  # (batch_size, channels, height * width)
    feature_map = feature_map.permute(0, 2, 1)  # (batch_size, height * width, channels)
    return feature_map

def cropToDivisible(feature_map, grid_size):
    """
    Crop a feature map so its height and width are divisible by ``grid_size``.
    Args:
        feature_map: Input feature map shaped ``(B, C, H, W)``.
        grid_size: Grid or patch size.
    Returns:
        Cropped feature map whose spatial dimensions are divisible by ``grid_size``.
    """
    _, _, h, w = feature_map.shape
    # Determine how many pixels must be removed.
    crop_h = h % grid_size
    crop_w = w % grid_size
    if crop_h == 0 and crop_w == 0:
        return feature_map  # No cropping is required.
    
    # Distribute the crop symmetrically across opposite sides.
    crop_h_top = crop_h // 2
    crop_h_bottom = crop_h - crop_h_top
    crop_w_left = crop_w // 2
    crop_w_right = crop_w - crop_w_left
    
    # Apply the crop.
    cropped = feature_map[
        :, 
        :, 
        crop_h_top : h - crop_h_bottom, 
        crop_w_left : w - crop_w_right
    ]
    return cropped

def cropToWindowSize(feature_map, window_size):
    """
    Crop a feature map to the requested spatial window size.
    Args:
        feature_map: Input feature map shaped ``(B, C, H, W)``.
        grid_size: Target window size.
    Returns:
        Cropped feature map with the requested spatial dimensions.
    """
    _, _, h, w = feature_map.shape
    # Determine how many pixels must be removed.
    crop_h = (h - window_size[0])
    crop_w = (w - window_size[1])
    if crop_h == 0 and crop_w == 0:
        return feature_map  # No cropping is required.
    
    # Distribute the crop symmetrically across opposite sides.
    crop_h_top = crop_h // 2
    crop_h_bottom = crop_h - crop_h_top
    crop_w_left = crop_w // 2
    crop_w_right = crop_w - crop_w_left
    
    # Apply the crop.
    cropped = feature_map[
        :, 
        :, 
        crop_h_top : h - crop_h_bottom, 
        crop_w_left : w - crop_w_right
    ]
    return cropped


    

def selectEleFromListWithIndice(src_list,indice):
    '''select and generate a new list from src list given indice
    Args:
        src_list: [x,x,x]
        indice:   [[x,x],x,[x,x]]  0 < x < n
    '''
    dst_list = []
    for ind in indice:
        tmp_list = []
        if type(ind) == list:
            tmp_list = src_list[int(ind[0]-1):int(ind[1])]
        else:
            tmp_list = [src_list[int(ind-1)]]
        dst_list += tmp_list
        
    return dst_list

def selectEleFromListWithSampleMode(src_list:list,sample_mode:list):
    '''select and generate a new list from src list given indice
    Args:
        src_list: [x,x,x]
        sample_mode:   [[start_ind, gap, end_ind],...]
    '''
    src_len = len(src_list)

    dst_list = []
    start_ind = sample_mode[0]-1
    gap = sample_mode[1]
    end_ind = sample_mode[2]
    p_ind = start_ind

    if end_ind == -1:
        samp_num = int((src_len - start_ind)/gap)
    else:
        samp_num = int((end_ind - start_ind)/gap)

    assert samp_num > 0, "samp_num should larger than 0"

    #print(src_list)
    for i in range(samp_num):
        dst_list.append(src_list[int(p_ind)])
        p_ind += gap
    return dst_list
    
def listFilesByExtension(folder_path, file_extension = "*"):
    """Return absolute paths of files with the requested extension.

    Args:
        folder_path: Absolute path to the directory.
        file_extension: Extension such as ``'png'``; ``'*'`` matches all files.

    Returns:
        A sorted list of matching absolute file paths.
    """
    # Check that the path exists.
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"folder path does not exist: {folder_path}")

    # Check that the path is a directory.
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"the given path is not a folder: {folder_path}")

    # Initialize the result list.
    file_paths = []

    # Traverse the directory tree.
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Match every file when the extension is '*'.
            if file_extension == '*':
                file_paths.append(os.path.join(root, file))
            # Otherwise match the requested extension.
            elif file.lower().endswith(f".{file_extension.lower()}"):
                file_paths.append(os.path.join(root, file))
    
    return sorted(file_paths)

def listAllFoldersInfo(path):
    """Return the names and paths of all direct subdirectories."""
    folders_info = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            folders_info.append({
                'name': item,
                'path': item_path
            })
    return folders_info

def getFileNameFromPath(path,with_extenstion = True):
    
    if with_extenstion:
        """Return the filename with its extension."""
        return os.path.basename(path)
    else:
        filename = os.path.basename(path)
        return os.path.splitext(filename)[0]

def listAllFolders(folder_path):
    """
    Return absolute paths of all nested subdirectories.

    Args:
        folder_path: Absolute path to the parent directory.

    Returns:
        A sorted list of absolute subdirectory paths.
    """
    # Check that the path exists.
    if not os.path.exists(folder_path):
        # A missing path yields an empty result.
        print(f"folder path does not exist: {folder_path}")
        return []
    # Check that the path is a directory.
    if not os.path.isdir(folder_path):
        # A non-directory path yields an empty result.
        print(f"the given path is not a folder: {folder_path}")
        return []
    # Initialize the result list.
    folder_paths = []

    # Traverse the directory tree.
    for root, dirs, _ in os.walk(folder_path):
        for dir_name in dirs:
            folder_paths.append(os.path.join(root, dir_name))

    return sorted(folder_paths)

def listAllFolders_firstRank(folder_path):
    """
    Return absolute paths of direct subdirectories.

    Args:
        folder_path: Absolute path to the parent directory.

    Returns:
        A sorted list of direct subdirectory paths.
    """
    # Check that the path exists.
    if not os.path.exists(folder_path):
        # A missing path yields an empty result.
        print(f"folder path does not exist: {folder_path}")
        return []
    # Check that the path is a directory.
    if not os.path.isdir(folder_path):
        # A non-directory path yields an empty result.
        print(f"the given path is not a folder: {folder_path}")
        return []
    
    # Inspect the directory's immediate children.
    file_name_list = sorted(os.listdir(folder_path))
    folder_path_list = []
    for file_name in file_name_list:
        tmp_path = os.path.join(folder_path,file_name)
        print(tmp_path)
        if os.path.isdir(tmp_path):
            folder_path_list.append(tmp_path)

    return folder_path_list

def selectFilesFromFolder(files_name,folder_path):
    if os.path.exists(folder_path) == False:
        print("folder path not exists")
        return []
    sel_files_list = []
    files_name_list = os.listdir(folder_path)
    
    for file_name in files_name:
        if file_name in files_name_list:
            sel_files_list.append(folder_path + "/" + file_name)
    return sel_files_list

def selectFoldersFromPath(folders_name, path):
    if os.path.exists(path) == False:
        print("folder path not exists")
        return []
    sel_folders_list = []
    folders_name_list = os.listdir(path)
    
    for folder_name in folders_name:
        if folder_name in folders_name_list:
            sel_folders_list.append(path + "/" + folder_name)
    return sel_folders_list
    

def loadImage2Tensor(image_path):
    """
    Read an image file and convert it to a PyTorch tensor.

    Args:
        image_path: Path to the image file.

    Returns:
        The converted image tensor.
    """
    # Check that the image file exists.
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    # Open the image file.
    image = Image.open(image_path).convert('RGB')

    # Define the image transformation.
    transform = transforms.ToTensor()

    # Convert the image to a tensor.
    tensor = transform(image)

    return tensor

def arr2BatchArr(arr,batch_size):
    arr_list = [arr for _ in range(batch_size)]  # Replicate the 4x4 matrix across the batch.
    batch_arr = np.stack(arr_list, axis=0)
    return batch_arr

def convertDepthMap2RGB(depth_map:np.array):
    '''covert a (H,W,1) or (H,W) depth map to (H,W,3) rgb image
    '''
    depth_rgb = np.expand_dims(depth_map, axis=2)  # Add a channel dimension.
    depth_rgb_extend = np.repeat(depth_rgb, 3, axis=2).astype(np.uint8)
    
    return depth_rgb_extend
    

def loadImage(image_path, format = "RGB",channel_type = "CHW",gray_channel = 1):
    # Check that the image file exists.
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"image path does not exist: {image_path}")

    # Open the image file.
    #image = Image.open(image_path).convert(format)
    image_array_transposed = np.array([])
    #image_array_transposed = np.transpose(image_rgb, (2, 0, 1))
    image_rgb = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if format == "RGB":
        #image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        if (channel_type == "CHW"):
            image_array_transposed = np.transpose(image_rgb,(2, 0, 1))
        else:
            image_array_transposed = image_rgb
    elif format == "GRAY":
        gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
        if gray_channel == 3:
            # Replicate the grayscale channel three times.
            gray_image = np.expand_dims(gray_image, axis=2)  # Add a channel dimension.
            image_gray_extend= np.repeat(gray_image, 3, axis=2)
        else:
            image_gray_extend = np.expand_dims(gray_image, axis=2)  # Add a channel dimension.

        if (channel_type == "CHW"):
            image_array_transposed = np.transpose(image_gray_extend,(2, 0, 1))
        else:
            image_array_transposed = image_gray_extend

    assert image_array_transposed.shape[0] != 0, "image size == 0"

    return image_array_transposed

def loadImagesFromPaths(paths_list, format = "RGB", channel_type = "CHW", gray_channel = 1):
    imgs_list = []
    for path in paths_list:
        imgs_list.append(loadImage(path,format,
                                   channel_type = channel_type,
                                   gray_channel = gray_channel))
    return imgs_list

def loadNpyArrFromPaths(paths_list):
    arr_list = []
    for path in paths_list:
        arr_list.append(np.load(path))
    return arr_list

def loadJsonsFromPaths(paths_list):
    json_list = []
    for path in paths_list:
        file = open(path, 'r')
        json_list.append(json.load(file))
    return json_list
    
def arr2Tensor(arr, dtype = torch.float):
    '''
    convert numpy arr to tensor
    '''
    return torch.from_numpy(arr).type(dtype)
    
def arrList2TensorList(arr_list):
    tensor_list = []
    for arr in arr_list:
        tensor_list.append(arr2Tensor(arr))
    return tensor_list

def arr2TensorNormalized(arr:np.ndarray):
    '''
    convert np ndarray to normalized torch tensor
    '''
    preprocess = transforms.Compose([
            transforms.ToTensor(),  # Convert to a tensor and scale pixels to [0, 1].
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Standardize channels.
        ])
    image_tensor = preprocess(arr)  # (C, H, W)
    #print(image_tensor.shape)
    return image_tensor

def tensor2Arr(tensor_x):
    return tensor_x.cpu().detach().numpy()


def stackTensors(tensor_list):
    """
    Stack a list of ``n`` tensors shaped ``(x, y, z)`` into one tensor.

    Args:
        tensor_list: List of ``n`` tensors shaped ``(x, y, z)``.

    Returns:
        A tensor shaped ``(n, x, y, z)``.
    """
    if not tensor_list:
        raise ValueError("the input list cannot be empty")

    # Verify that all tensors have the same shape.
    shapes = [tensor.shape for tensor in tensor_list]
    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError("tensors in the list should have the same shape")

    # Stack tensors along a new leading dimension.
    result = torch.stack(tensor_list, dim=0)
    return result

def tensorStandardized(x):
    '''tensor standardized to 0-1 using std
    Args:
        x: (N)
    '''
    
    # Use a small epsilon for numerical stability.
    epsilon = 1e-12

    # Standardize without risking division by zero.
    x_mean = x.mean()
    x_std = x.std()

    # Clamp the standard deviation away from zero.
    x_standardized = (x - x_mean) / torch.clamp(x_std, min=epsilon)
    return x_standardized

def tensorNormalizedMaxMin(x, min = 0, max = 1):
    '''tensor nomalized to 0-1 using max and min value 
    Args:
        x: (*)
    '''
    # Use a small epsilon for numerical stability.
    epsilon = 1e-12

    # Normalize without risking division by zero.
    x_min = x.min()
    x_max = x.max()
    x_normalized = (x - x_min) / torch.maximum(x_max - x_min, torch.tensor(epsilon))
    x_normalized =  x_normalized*(max-min) + min
    return x_normalized

def tensorNormalizedMaxMin_perChannel(tensor, min = 0, max = 1):
    """Apply min-max normalization independently to each channel of a ``(C, H, W)`` tensor."""
    # Compute the minimum and maximum of each channel.
    # Dimensions 1 and 2 correspond to H and W.
    min_vals = tensor.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    max_vals = tensor.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]

    # Avoid division by zero for constant inputs.
    epsilon = 1e-12  # Small positive constant.
    normalized_tensor = (tensor - min_vals) / torch.maximum(max_vals - min_vals, torch.tensor(epsilon))
    normalized_tensor = normalized_tensor*(max-min) + min

    return normalized_tensor

def tensorNormalizedMaxMin_batch_perChannel(tensor, min_val=0, max_val=1):
    """Apply min-max normalization per batch item and channel to a ``(B, C, H, W)`` tensor."""
    # Compute per-channel extrema for each batch item.
    # Dimensions 2 and 3 correspond to H and W.
    min_vals = tensor.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    max_vals = tensor.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]

    # Avoid division by zero for constant inputs.
    epsilon = 1e-12  # Small positive constant.
    normalized_tensor = (tensor - min_vals) / torch.maximum(max_vals - min_vals, torch.tensor(epsilon))
    normalized_tensor = normalized_tensor * (max_val - min_val) + min_val

    return normalized_tensor

def tensorNormalizedMaxMin_perBatch(tensor, min_val=0, max_val=1):
    """Normalize each batch item using extrema shared across all channels and spatial dimensions."""
    # Compute global extrema over C, H, and W for each batch item.
    min_vals = tensor.amin(dim=(1, 2, 3), keepdim=True)  # shape=(B, 1, 1, 1)
    max_vals = tensor.amax(dim=(1, 2, 3), keepdim=True)  # shape=(B, 1, 1, 1)

    # Normalize and scale to the requested output range.
    epsilon = 1e-12
    normalized_tensor = (tensor - min_vals) / torch.clamp(max_vals - min_vals, min=epsilon)
    normalized_tensor = normalized_tensor * (max_val - min_val) + min_val

    return normalized_tensor

def arrNormalizedMaxMin(x, min = 0, max = 1):
    '''tensor nomalized to 0-1 using max and min value 
    Args:
        x: (*)
    '''
    # Use a small epsilon for numerical stability.
    epsilon = 1e-12

    # Normalize without risking division by zero.
    x_min = x.min()
    x_max = x.max()
    x_normalized = (x - x_min) / np.maximum(x_max - x_min, (epsilon))
    x_normalized =  x_normalized*(max-min) + min
    return x_normalized

def arrNormalizedMaxMin_perChannel(x, min=0, max=1, channel_axis=-1):
    """Normalize a NumPy array independently along its specified channel axis."""
    epsilon = 1e-12
    
    # Compute per-channel extrema while retaining dimensions for broadcasting.
    min_vals = np.min(x, axis=tuple([i for i in range(x.ndim) if i != channel_axis]), keepdims=True)
    max_vals = np.max(x, axis=tuple([i for i in range(x.ndim) if i != channel_axis]), keepdims=True)
    
    # Normalize each channel.
    normalized = (x - min_vals) / np.maximum(max_vals - min_vals, epsilon)
    normalized = normalized * (max - min) + min
    
    return normalized



def tensor_filterZ2Values(points: np.ndarray, z_threshold: float,dst_value = 0.0) -> np.ndarray:
    """Filter a ``(B, 3, N)`` point tensor by its Z-coordinate range."""
    z_mask = points[:, 2, :] < z_threshold  # (B, N)
    
    # Expand the mask to (B, 3, N) and fill invalid entries with zero.
    points = points.masked_fill(z_mask.unsqueeze(1), dst_value)
    
    return points

def arrMedianFilter(x,size = 3,mode = 'reflect'):
    '''Median Filter
    '''
    dst_arr = median_filter(x, size=size, mode=mode)
    return dst_arr

def visualizeTensor(tensor_x, window_name = "tmp"):
    assert len(tensor_x.size()) == 3, "visualization of tensor should be size of 3 (x, x, x)"
    assert tensor_x.size()[0] == 1 or tensor_x.size()[0] == 3, "tensor channel should be 1 or 3"

    arr_x = tensor2Arr(tensor_x)
    
    image_arr = arrNormalizedMaxMin(arr_x, 0 , 255)
    image_arr = image_arr.astype(np.uint8)
    image_arr = np.transpose(image_arr, (1, 2, 0))
    print(image_arr.shape)
    cv2.imshow(window_name,image_arr)
    cv2.waitKey()
    #cv2.destroyAllWindows()
    return 

def imshow(window_name, img_arr, channel_type = "CHW"):
    '''Visualize an image array
    Default data channel type is CxHxW
    '''
    if channel_type == "CHW":
        cv2.imshow(window_name, np.transpose(img_arr, (1, 2, 0)))
    else:
        cv2.imshow(window_name, img_arr)
    cv2.waitKey()

def convertTensor2ImageArr(tensor_x):
    assert len(tensor_x.size()) == 3, "visualization of tensor should be size of 3 (x, x, x)"
    assert tensor_x.size()[0] == 1 or tensor_x.size()[0] == 3, "tensor channel should be 1 or 3"

    arr_x = tensor2Arr(tensor_x)
    
    image_arr = arrNormalizedMaxMin(arr_x, 0 , 255)
    image_arr = image_arr.astype(np.uint8)
    image_arr = np.transpose(image_arr, (1, 2, 0))
    
    return image_arr

def calRemapScaledSize(src_K: np.ndarray, dst_K: np.ndarray, src_size: tuple):
    """Compute a target size that supports lossless intrinsic-based image remapping before cropping."""
    src_width = src_size[1]
    src_height = src_size[0]

    src_K_fx = src_K[0][0]
    src_K_fy = src_K[1][1]
    dst_K_fx = dst_K[0][0]
    dst_K_fy = dst_K[1][1]

    width_scale = dst_K_fx/src_K_fx
    height_scale = dst_K_fy/src_K_fy
    # calculate resize height and width
    resize_width = int(src_width*width_scale)
    resize_height = int(src_height*height_scale)

    return (resize_height,resize_width)

def scaleImage(src_image: np.ndarray, dst_img_size: tuple):
    dst_image = cv2.resize(src_image,(dst_img_size[1],dst_img_size[0]))
    return dst_image

def remapImage(src_image: np.ndarray, src_K: np.ndarray, dst_K: np.ndarray, dst_img_size: tuple,channel_type = "CHW") -> np.ndarray:
    """Remap an image from source camera intrinsics to target camera intrinsics."""
    if channel_type == "CHW":
        src_image = np.transpose(src_image, (1, 2, 0))
    
    # Read the target image dimensions.
    dst_height, dst_width = dst_img_size
    src_width = src_image.shape[1]
    src_height = src_image.shape[0]
    src_channel = src_image.shape[2]

    src_K_fx = src_K[0][0]
    src_K_fy = src_K[1][1]
    dst_K_fx = dst_K[0][0]
    dst_K_fy = dst_K[1][1]
    src_K_dx = src_K[0][2]
    src_K_dy = src_K[1][2]
    dst_K_dx = dst_K[0][2]
    dst_K_dy = dst_K[1][2]

    width_scale = dst_K_fx/src_K_fx
    height_scale = dst_K_fy/src_K_fy
    # calculate resize height and width
    resize_width = int(src_width*width_scale)
    resize_height = int(src_height*height_scale)
    src_image_scale = cv2.resize(src_image,(resize_width,resize_height))

    # the offset of pixel coordinate of the scaled src image
    origin_x = -src_K_dx/src_K_fx*dst_K_fx + dst_K_dx
    origin_y = -src_K_dy/src_K_fy*dst_K_fy + dst_K_dy
    
    
    # Generate the target-image pixel grid.
    x, y = np.meshgrid(np.arange(resize_width), np.arange(resize_height))
    dst_x = x + origin_x
    dst_y = y + origin_y
    #ones = np.ones_like(x)
    dst_coords = np.stack([dst_x, dst_y], axis=-1)
    dst_coords = dst_coords.reshape((resize_height*resize_width),2) # (HW, 2)
    
    rev_index = (dst_coords[:,0] > 0) & (dst_coords[:,1] > 0) & (dst_coords[:,0] < dst_width) & (dst_coords[:,1] < dst_height)
    dst_coords = dst_coords[rev_index].astype(np.int32)
    
    if channel_type == "CHW":
        src_image_colors = src_image_scale.reshape(resize_height*resize_width,src_channel)
        dst_colors = src_image_colors[rev_index,:]
        remapped_image = np.zeros((src_channel,dst_img_size[0],dst_img_size[1]),dtype=np.uint8)
        remapped_image[:,dst_coords[:,1],dst_coords[:,0]] = np.transpose(dst_colors,(1,0))
    else:
        src_image_colors = src_image_scale.reshape(resize_height*resize_width,src_channel)
        dst_colors = src_image_colors[rev_index,:]
        remapped_image = np.zeros((dst_img_size[0],dst_img_size[1],src_channel),dtype=np.uint8)
        remapped_image[dst_coords[:,1],dst_coords[:,0],:] = dst_colors

    #print(remapped_image.shape)
    return remapped_image

def remapArr2D_uint8(src_arr: np.ndarray, src_K: np.ndarray, dst_K: np.ndarray, dst_img_size: tuple) -> np.ndarray:
    """Remap a 2D uint8 array from source camera intrinsics to target camera intrinsics."""
    
    # Read the target image dimensions.
    dst_height, dst_width = dst_img_size
    src_width = src_arr.shape[1]
    src_height = src_arr.shape[0]

    src_K_fx = src_K[0][0]
    src_K_fy = src_K[1][1]
    dst_K_fx = dst_K[0][0]
    dst_K_fy = dst_K[1][1]
    src_K_dx = src_K[0][2]
    src_K_dy = src_K[1][2]
    dst_K_dx = dst_K[0][2]
    dst_K_dy = dst_K[1][2]

    width_scale = dst_K_fx/src_K_fx
    height_scale = dst_K_fy/src_K_fy
    # calculate resize height and width
    resize_width = int(src_width*width_scale)
    resize_height = int(src_height*height_scale)
    src_image_scale = cv2.resize(src_arr,(resize_width,resize_height))

    # the offset of pixel coordinate of the scaled src image
    origin_x = -src_K_dx/src_K_fx*dst_K_fx + dst_K_dx
    origin_y = -src_K_dy/src_K_fy*dst_K_fy + dst_K_dy
    
    
    # Generate the target-image pixel grid.
    x, y = np.meshgrid(np.arange(resize_width), np.arange(resize_height))
    dst_x = x + origin_x
    dst_y = y + origin_y
    #ones = np.ones_like(x)
    dst_coords = np.stack([dst_x, dst_y], axis=-1)
    dst_coords = dst_coords.reshape((resize_height*resize_width),2) # (HW, 2)
    
    rev_index = (dst_coords[:,0] > 0) & (dst_coords[:,1] > 0) & (dst_coords[:,0] < dst_width) & (dst_coords[:,1] < dst_height)
    dst_coords = dst_coords[rev_index].astype(np.int32)
    
    
    src_image_colors = src_image_scale.reshape(resize_height*resize_width)
    dst_colors = src_image_colors[rev_index]
    remapped_image = np.zeros((dst_img_size[0],dst_img_size[1]),dtype=np.uint8)
    remapped_image[dst_coords[:,1],dst_coords[:,0]] = dst_colors
        
    #print(remapped_image.shape)
    return remapped_image # (H,W)

def remapArr2D_Float(src_arr: np.ndarray, src_K: np.ndarray, dst_K: np.ndarray, dst_img_size: tuple) -> np.ndarray:
    """Remap a floating-point 2D array from source camera intrinsics to target camera intrinsics."""
    
    # Read the target image dimensions.
    dst_height, dst_width = dst_img_size
    src_width = src_arr.shape[1]
    src_height = src_arr.shape[0]

    src_K_fx = src_K[0][0]
    src_K_fy = src_K[1][1]
    dst_K_fx = dst_K[0][0]
    dst_K_fy = dst_K[1][1]
    src_K_dx = src_K[0][2]
    src_K_dy = src_K[1][2]
    dst_K_dx = dst_K[0][2]
    dst_K_dy = dst_K[1][2]

    width_scale = dst_K_fx/src_K_fx
    height_scale = dst_K_fy/src_K_fy
    # calculate resize height and width
    resize_width = int(src_width*width_scale)
    resize_height = int(src_height*height_scale)
    src_image_scale = cv2.resize(src_arr,(resize_width,resize_height), interpolation=cv2.INTER_NEAREST)

    # the offset of pixel coordinate of the scaled src image
    origin_x = -src_K_dx/src_K_fx*dst_K_fx + dst_K_dx
    origin_y = -src_K_dy/src_K_fy*dst_K_fy + dst_K_dy
    
    
    # Generate the target-image pixel grid.
    x, y = np.meshgrid(np.arange(resize_width), np.arange(resize_height))
    dst_x = x + origin_x
    dst_y = y + origin_y
    #ones = np.ones_like(x)
    dst_coords = np.stack([dst_x, dst_y], axis=-1)
    dst_coords = dst_coords.reshape((resize_height*resize_width),2) # (HW, 2)
    
    rev_index = (dst_coords[:,0] > 0) & (dst_coords[:,1] > 0) & (dst_coords[:,0] < dst_width) & (dst_coords[:,1] < dst_height)
    dst_coords = dst_coords[rev_index].astype(np.int32)
    
    
    src_image_colors = src_image_scale.reshape(resize_height*resize_width)
    dst_colors = src_image_colors[rev_index]
    remapped_image = np.zeros((dst_img_size[0],dst_img_size[1]),dtype=np.float32)
    remapped_image[dst_coords[:,1],dst_coords[:,0]] = dst_colors
        
    #print(remapped_image.shape)
    return remapped_image # (H,W)

def resizeImageByWidth(image, target_width):
    """Resize an image to a target width while preserving its aspect ratio."""
    h, w = image.shape[:2]  # Read the original height and width.
    scale = target_width / w  # Compute the scale factor.
    target_height = int(h * scale)  # Compute the resized height.
    
    resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized_image

def concatImage(img1, img2, 
                title1 = "1", title2 = "2", 
                title_location = (50,50),
                gap_width = 50,
                total_width = 1200
                ):
    concat_img_height = img1.shape[0]
    
    # Ensure that the image array is writable.
    if not img1.flags.writeable:
        img1.flags.writeable = True
    # Ensure that the image is stored contiguously.
    if not img1.flags.c_contiguous:
        img1 = np.ascontiguousarray(img1)

    if not img2.flags.writeable:
        img2.flags.writeable = True
    # Ensure that the image is stored contiguously.
    if not img2.flags.c_contiguous:
        img2 = np.ascontiguousarray(img2)    

    # Add the title in the upper-left corner.
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Create an example image shaped (800, 1200, 3).
    cv2.putText(img1, title1, title_location, font, 1.5, (0, 0, 255), 3)  # Title for the left image.
    cv2.putText(img2, title2, title_location, font, 1.5, (0, 0, 255), 3)  # Title for the right image.

    # Set the gap width, for example 50 pixels.
    gap = np.ones((concat_img_height, gap_width, 3), dtype=np.uint8) * 255  # Create a white separator.

    # Concatenate the images and gap horizontally.
    concatenated_img = np.hstack((img1, gap, img2))  # cv2.hconcat can be used equivalently.
    concatenated_img = resizeImageByWidth(concatenated_img,total_width)
    return concatenated_img

def concatImageGrid(image_list,
                    h, w,
                    titles=None,
                    title_location=(30, 40),
                    gap_width=20,
                    total_width=1200):
    """Arrange a list of images in an ``h``-by-``w`` grid with optional titles and gaps."""

    num_images = len(image_list)
    grid_size = h * w

    # Pad the image list to the requested grid size.
    if num_images < grid_size:
        # Create blank images using the first image dimensions.
        ref = image_list[0]
        blank = np.ones_like(ref) * 255
        image_list = image_list + [blank] * (grid_size - num_images)

    # Prepare each image.
    processed_imgs = []
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, img in enumerate(image_list[:grid_size]):
        # Ensure that the array is writable and contiguous.
        if not img.flags.writeable:
            img = img.copy()
        if not img.flags.c_contiguous:
            img = np.ascontiguousarray(img)

        # Add the title.
        if titles is not None and i < len(titles):
            cv2.putText(img, titles[i], title_location,
                        font, 1.0, (0, 0, 255), 2)

        processed_imgs.append(img)

    # Build the image grid.
    rows = []
    for i in range(h):
        row_imgs = processed_imgs[i * w:(i + 1) * w]

        # Insert horizontal gaps.
        gap_v = np.ones((row_imgs[0].shape[0], gap_width, 3), dtype=np.uint8) * 255

        row = row_imgs[0]
        for img in row_imgs[1:]:
            row = np.hstack((row, gap_v, img))

        rows.append(row)

    # Concatenate grid rows vertically.
    gap_h = np.ones((gap_width, rows[0].shape[1], 3), dtype=np.uint8) * 255

    grid_img = rows[0]
    for row in rows[1:]:
        grid_img = np.vstack((grid_img, gap_h, row))

    # ====== 5. resize ======
    grid_img = resizeImageByWidth(grid_img, total_width)

    return grid_img


def np_VisualizePixelMatchingResults(kpts1:np.ndarray, kpts2:np.ndarray, img1:np.ndarray, img2:np.ndarray):
    """Visualize pixel correspondences between two NumPy images."""
    # Ensure that both images have matching dimensions.
    H = max(img1.shape[0], img2.shape[0])
    W = max(img1.shape[1], img2.shape[1])

    img1_pad = np.zeros((H, W, 3), dtype=np.uint8)
    img2_pad = np.zeros((H, W, 3), dtype=np.uint8)
    img1_pad[:img1.shape[0], :img1.shape[1], :] = img1
    img2_pad[:img2.shape[0], :img2.shape[1], :] = img2

    # Concatenate the images.
    vis_img = np.concatenate((img1_pad, img2_pad), axis=1)

    # Draw lines between matched points.
    for pt1, pt2 in zip(kpts1, kpts2):
        color = tuple([random.randint(0, 255) for _ in range(3)])
        pt1_int = tuple(map(int, pt1))
        pt2_int = (int(pt2[0]) + W, int(pt2[1]))  # Offset the x coordinate of pt2 by the image width.
        cv2.line(vis_img, pt1_int, pt2_int, color, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(vis_img, pt1_int, 3, color, -1)
        cv2.circle(vis_img, pt2_int, 3, color, -1)

    return vis_img

def tensor_VisualizePixelMatchingResults(kpts1:torch.Tensor, kpts2:torch.Tensor, img1:torch.Tensor, img2:torch.Tensor):
    """Visualize pixel correspondences between two tensor images."""
    # Convert tensors to NumPy arrays.
    kpts1 = np.transpose(tensor2Arr(kpts1), (1, 0))
    kpts2 = np.transpose(tensor2Arr(kpts2), (1, 0))
    
    
    img1 = np.transpose(tensor2Arr(img1), (1, 2, 0))
    img1 = arrNormalizedMaxMin(img1, 0 , 255)
    img2 = np.transpose(tensor2Arr(img2), (1, 2, 0))
    
    # Ensure that both images have matching dimensions.
    H = max(img1.shape[0], img2.shape[0])
    W = max(img1.shape[1], img2.shape[1])

    img1_pad = np.zeros((H, W, 3), dtype=np.uint8)
    img2_pad = np.zeros((H, W, 3), dtype=np.uint8)
    img1_pad[:img1.shape[0], :img1.shape[1], :] = img1
    img2_pad[:img2.shape[0], :img2.shape[1], :] = img2

    # Concatenate the images.
    vis_img = np.concatenate((img1_pad, img2_pad), axis=1)

    # Draw lines between matched points.
    for pt1, pt2 in zip(kpts1, kpts2):
        color = tuple([random.randint(0, 255) for _ in range(3)])
        pt1_int = tuple(map(int, pt1))
        pt2_int = (int(pt2[0]) + W, int(pt2[1]))  # Offset the x coordinate of pt2 by the image width.
        cv2.line(vis_img, pt1_int, pt2_int, color, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(vis_img, pt1_int, 3, color, -1)
        cv2.circle(vis_img, pt2_int, 3, color, -1)

    return vis_img

def visualNpArrHWC(arr,title = "arr_img",wait = True,show_max_min = False, per_channel = False):
    if per_channel:
        arr_img = arrNormalizedMaxMin_perChannel(arr,0,255).astype(np.uint8)
    else:
        arr_img = arrNormalizedMaxMin(arr,0,255).astype(np.uint8)
   
    if show_max_min:
        print("arr_min", arr.min())
        print("arr_max",arr.max())
        
    cv2.imshow(title,arr_img)
    if wait:
        cv2.waitKey(0)
    else:
        cv2.waitKey(1)

def saveNpArrHWC_toImg(arr,dst_path,wait = True,show_max_min = False,per_channel = False):
    if per_channel:
        arr_img = arrNormalizedMaxMin_perChannel(arr,0,255).astype(np.uint8)
    else:
        arr_img = arrNormalizedMaxMin(arr,0,255).astype(np.uint8)

    if show_max_min:
        print("arr_min", arr.min())
        print("arr_max",arr.max())
    #print(dst_path)
    cv2.imwrite(dst_path,arr_img)    

def convert_RT():
    R = np.array([[0,-1,0],[0,0,-1],[1,0,0]],dtype=np.float64)
    #R = np.array([[0,-1,0],[1,0,0],[0,0,1]],dtype=np.float64)
    rvec, _ = cv2.Rodrigues(R)
    print(rvec)
    
def dgGetIndCombinations(ind_list,comb_inds,start_ind,end_ind,num_in_combine,depth):

    '''
    brief 
    ------
    dg generate combinations
    '''
    if (num_in_combine == depth + 1):

        # loop to complete the adding process to get all combinations
        for i in range(start_ind,end_ind+1):
            comb_inds[depth] = i
            for k in range(0,num_in_combine):
                ind_list.append(comb_inds[k])
    else:
        for i in range(start_ind,end_ind - (num_in_combine-depth) + 2):
            comb_inds[depth] = i
            dgGetIndCombinations(ind_list,comb_inds,i+1,end_ind,num_in_combine,depth + 1)

def GetIndCombinations(ind_list,start_ind,end_ind,num_in_combine):
    comb_inds = []
    for i in range(num_in_combine):
        comb_inds.append(0)
    dgGetIndCombinations(ind_list,comb_inds,start_ind,end_ind,num_in_combine,0)
    


def batch_getImageBlockCenters(B, H, W, n_blocks, m_blocks, device='cpu'):
    '''
    B is the batch size
    '''
    h, w = H // n_blocks, W // m_blocks
    centers = []
    for i in range(n_blocks):
        for j in range(m_blocks):
            u = j * w + w / 2
            v = i * h + h / 2
            centers.append([u, v])
    return torch.tensor(centers, dtype=torch.float32, device=device).repeat(B, 1, 1)  # shape: (N, 2)


def getImageBlockCenters(H, W, n_blocks, m_blocks, device='cpu'):
    h, w = H // n_blocks, W // m_blocks
    centers = []
    for i in range(n_blocks):
        for j in range(m_blocks):
            u = j * w + w / 2
            v = i * h + h / 2
            centers.append([u, v])
    return torch.tensor(centers, dtype=torch.float32, device=device)  # shape: (N, 2)

def sortAndSamplebyZ(points_xyz, target_num_points):
    """Sort points by ascending Z coordinate and sample them at uniform index intervals."""
    n = points_xyz.shape[0]
    
    # Step 1: Sort points by ascending Z coordinate.
    z_values = points_xyz[:, 2]  # Extract the Z coordinates.
    sort_indices = np.argsort(z_values)  # Obtain the sorting indices.
    sorted_points = points_xyz[sort_indices]  # Apply the ordering to the point cloud.
    
    # Step 2: Sample at uniform index intervals.
    if target_num_points >= n:
        # Return all points when the target count is at least the input count.
        sampled_indices = np.arange(n)
        print(f"warning: target points number ({target_num_points}) is larger than the initial number({n})，returning all points")
    else:
        # Generate uniformly spaced sample indices.
        sampled_indices = np.linspace(0, n-1, target_num_points, dtype=int)
    
    # Return sampled points and their original indices.
    sampled_points = sorted_points[sampled_indices]
    original_indices = sort_indices[sampled_indices]
    
    return sampled_points, original_indices

def isProjSizeSim(proj_size_a, proj_size_b, thres = 100.0):
    diff = np.array(proj_size_a) - np.array(proj_size_b)
    diff_norm = np.linalg.norm(diff)
    #print(proj_size_a)
    #print(proj_size_b)
    #print(f"Projection size difference: {diff_norm}")
    if diff_norm < thres:
        return True
    else:
        return False
    
