#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import sys
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
from depthInfer import*
from tqdm import tqdm
import random
from operator import itemgetter
from pcloud import*

# __________________ Dataset Generation (KITTI) ________________________#
def Dataset_batchDepthFolderGeneration(root_path, 
                                       key_word = "*",
                                       depth_solution = "Moge2+DAR", 
                                       generate_method = "LIDAR_INI",
                                       T_ini = zv.T_LVINI_NP,
                                       Depth_folder_name = "Depth", 
                                       rank = 1):
    '''
    Args:
        generate_method: "LIDAR_INI" | "INPUT_INI" | "AUTO_INI"
    '''
    print(f"Depth Solution: [{GREEN}{depth_solution}{RESET}]")

    Moge2_model = createMogeModel()
    if generate_method == "LIDAR_INI":
        gen_T_pred = zv.T_LVINI_NP
        t_per_ragne = 0.0
        r_per_range = 0.0
        depth_config_json = {
            "SOLUTION": depth_solution,
            "GEN_METHOD": generate_method,
            "DAR_Range": [r_per_range, t_per_ragne],
            "numpy_version": np.__version__
        }
        print(f"Depth Generate Method: {GREEN}{generate_method}{RESET}")
    elif generate_method == "AUTO_INI":
        gen_T_pred = np.array([None])
        t_per_ragne = 0.5
        r_per_range = 5.0
        depth_config_json = {
            "SOLUTION": depth_solution,
            "GEN_METHOD": generate_method,
            "DAR_Range": [r_per_range, t_per_ragne],
            "numpy_version": np.__version__
        }
        print(f"Depth Generate Method: {GREEN}{generate_method}{RESET}")
    elif generate_method == "INPUT_INI":
        gen_T_pred = np.array(T_ini)
        t_per_ragne = 0.0
        r_per_range = 0.0
        depth_config_json = {
            "SOLUTION": depth_solution,
            "GEN_METHOD": generate_method,
            "DAR_Range": [r_per_range, t_per_ragne],
            "numpy_version": np.__version__
        }
        print(f"Depth Generate Method: {GREEN}{generate_method}{RESET}")
    else:
        gen_T_pred = np.array([None])
        t_per_ragne = 0.5
        r_per_range = 5.0
        depth_config_json = {
            "SOLUTION": depth_solution,
            "GEN_METHOD": "AUTO_INI",
            "DAR_Range": [r_per_range, t_per_ragne],
            "numpy_version": np.__version__
        }
        print(f"{YELLOW}[Warning]{RESET}Depth Generate Method is not provided correctly, fall back to: {YELLOW}Camera_Rand{RESET}")
        
    if rank == 1:
        # if only has one rank son folders
        folder_list = zv.listAllFolders_firstRank(root_path)

        deal_subset_list = []
        notdeal_subset_list = []
        for folder_path in folder_list:

            folder_name = zv.getFileNameFromPath(folder_path)
            
            # depth generation for subset
            subset_name = folder_name
            if key_word != "*":
                if key_word in subset_name:
                    pass
                else:
                    # if key word not in the subset name, then skip it
                    continue
            
            print("Depth Generation Begin, Subset Name is: ",subset_name)
            subset_folder_list = zv.listAllFolders_firstRank(folder_path)
            depth_folder_existed = False
            for subset_folder_path in subset_folder_list:
                subset_folder_name = zv.getFileNameFromPath(subset_folder_path)
                if subset_folder_name  == "calib":
                    #calib_json_path = zv.listFilesByExtension(folder_path=subset_folder_path, file_extension="json")[0]
                    calib_json_folder = subset_folder_path
                elif subset_folder_name == "pointclouds":
                    pcd_folder_path = subset_folder_path
                elif subset_folder_name == "RGB":
                    rgb_folder_path = subset_folder_path
                elif subset_folder_name == Depth_folder_name:
                    depth_folder_existed = True
            if depth_folder_existed:
                notdeal_subset_list.append(subset_name)
                continue

            depth_folder_path = os.path.join(folder_path, Depth_folder_name)
            json_path = os.path.join(depth_folder_path,"config.json")
            os.makedirs(depth_folder_path,exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump(depth_config_json, f)

            if depth_solution == "Moge2+DAR":
                generatePCDFromImg_ByFolder_AutoInitialGuess(rgb_folder_path,
                                                        pcd_folder_path,
                                                        depth_folder_path,
                                                        calib_json_folder,
                                                        Moge2_model,
                                                        T_pred = gen_T_pred,
                                                        t_perturb_range = t_per_ragne,
                                                        euler_perturb_range= r_per_range,
                                                        visual_process = False)
            elif depth_solution == "Moge2":
                inferImageFolder_Moge(rgb_folder_path,
                                      depth_folder_path)
                

            print("Depth Generation Done")
            deal_subset_list.append(subset_name)
    elif rank == 2:
        root_folder_list = zv.listAllFolders_firstRank(root_path)
        for root_folder in root_folder_list:
            folder_list = zv.listAllFolders_firstRank(root_folder)
            deal_subset_list = []
            notdeal_subset_list = []
            for folder_path in folder_list:
                folder_name = zv.getFileNameFromPath(folder_path)
                
                # depth generation for subset
                subset_name = folder_name
                if key_word != "*":
                    if key_word in subset_name:
                        pass
                    else:
                        # if key word not in the subset name, then skip it
                        continue

                print("Depth Generation Begin, Subset Name is: ",subset_name)
                subset_folder_list = zv.listAllFolders_firstRank(folder_path)
                depth_folder_existed = False
                for subset_folder_path in subset_folder_list:
                    subset_folder_name = zv.getFileNameFromPath(subset_folder_path)
                    if subset_folder_name  == "calib":
                        #calib_json_path = zv.listFilesByExtension(folder_path=subset_folder_path, file_extension="json")[0]
                        calib_json_folder = subset_folder_path
                    elif subset_folder_name == "pointclouds":
                        pcd_folder_path = subset_folder_path
                    elif subset_folder_name == "RGB":
                        rgb_folder_path = subset_folder_path
                    elif subset_folder_name == Depth_folder_name:
                        depth_folder_existed = True
                if depth_folder_existed:
                    notdeal_subset_list.append(subset_name)
                    continue

                depth_folder_path = os.path.join(folder_path,Depth_folder_name)
                json_path = os.path.join(depth_folder_path,"config.json")
                os.makedirs(depth_folder_path,exist_ok=True)
                with open(json_path, 'w') as f:
                    json.dump(depth_config_json, f)
                    
                if depth_solution == "Moge2+DAR":
                    generatePCDFromImg_ByFolder_AutoInitialGuess(rgb_folder_path,
                                                            pcd_folder_path,
                                                            depth_folder_path,
                                                            calib_json_folder,
                                                            Moge2_model,
                                                            T_pred = gen_T_pred,
                                                            t_perturb_range = t_per_ragne,
                                                            euler_perturb_range= r_per_range,
                                                            visual_process = False)
                elif depth_solution == "Moge2":
                    inferImageFolder_Moge(rgb_folder_path,
                                            depth_folder_path)
                                
                print("Depth Generation Done")
                deal_subset_list.append(subset_name)

    print(f"{BLUE}Depth Generation All Done. The generated subsets are: {deal_subset_list}\n The following subsets's Depth folder has already existed, so they are not included in the Depth generation:\n{notdeal_subset_list}{RESET}")


def parse_dataset_generate_args(args=None):
    parser = argparse.ArgumentParser(
        description="Generate depth folders for datasets in batch mode."
    )
    parser.add_argument(
        "--root-path",
        required=True,
        help="Root directory containing the dataset subsets.",
    )
    parser.add_argument(
        "--key-word",
        default="*",
        help="Only process subset names containing this keyword (default: *).",
    )
    parser.add_argument(
        "--depth-solution",
        choices=("Moge2", "Moge2+DAR"),
        default="Moge2+DAR",
        help="Depth generation solution (default: Moge2+DAR).",
    )
    parser.add_argument(
        "--generate-method",
        choices=("LIDAR_INI", "INPUT_INI", "AUTO_INI"),
        default="LIDAR_INI",
        help="Initial-pose generation method (default: LIDAR_INI).",
    )
    parser.add_argument(
        "--depth-folder-name",
        default="Depth",
        help="Name of the generated depth folder (default: Depth).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        choices=(1, 2),
        default=1,
        help="Dataset directory hierarchy rank (default: 1).",
    )
    return parser.parse_args(args)


def main():
    
    rewrite_menu = True
    
    while(1):
        # Display the menu.
        if rewrite_menu:
            print("----------------------------------")
            print("1-> convert pcd to ascii format")
            print("2-> down sample PCD by Folder")
            print("3-> generate Depth folder (Moge2)")
            print("4-> generate Depth folder (AutoInitial Guess)")
            print("5-> batch Depth Generation (Any Dataset)")
            print("q-> exit")
            
        order = input()
 
        rewrite_menu = True
        if (order == "1"):
            pcdConvert2ASCII_byFolder(r"./src_pointclouds",r"./dst_pointclouds")
        elif (order == "2"):
            downsamplePCD_byFolder(r"./Datasets/subset/pointclouds_bin",
                                   r"./Datasets/subset/pointclouds_downsampled",
                                   downsample_ratio=0.2,
                                   ascii = True)
            pass
        elif (order == "3"):
            '''
            Infer images by Folder (API)
            '''
            dataset_root = r"./Datasets/KITTI_00"
            depth_config_json = {
                "SOLUTION": "Moge2",
                "GEN_METHOD": "CAMERA",
                "DAR_Range": [0.0, 0.0],
                "numpy_version": np.__version__
            }

            os.makedirs(os.path.join(dataset_root,"Depth"),exist_ok=True)
            json_path = os.path.join(dataset_root,"Depth", "config.json")
            with open(json_path, 'w') as f:
                json.dump(depth_config_json, f)
            
            inferImageFolder_Moge(dataset_root+r"/RGB",
                                    dataset_root+r"/Depth")
        elif (order == "4"):
            Moge2_model = createMogeModel()
            dataset_root = r"./Datasets/KITTI_00"
            depth_config_json = {
                "SOLUTION": "Moge2+DAR",
                "GEN_METHOD": "AUTO_INI",
                "DAR_Range": [5.0, 0.5],
                "numpy_version": np.__version__
            }
            depth_folder_name = "Depth"
            json_path = os.path.join(dataset_root,depth_folder_name,"config.json")
            os.makedirs(os.path.join(dataset_root,depth_folder_name),exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump(depth_config_json, f)

            generatePCDFromImg_ByFolder_AutoInitialGuess(dataset_root+r"/RGB",
                        dataset_root+r"/pointclouds",
                        os.path.join(dataset_root,depth_folder_name),
                        dataset_root+r"/calib",
                        Moge2_model,
                        visual_process = False)
        elif (order == "5"):
            Dataset_batchDepthFolderGeneration(r"./Datasets",
                                               key_word = "KITTI",
                                               depth_solution = "Moge2+DAR",
                                               generate_method = "LIDAR_INI",
                                               Depth_folder_name = "Depth", 
                                               rank = 1)
        
        elif (order == "q"):
            return
        else:
            print("input error\n")
            rewrite_menu = False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        args = parse_dataset_generate_args()
        Dataset_batchDepthFolderGeneration(
            root_path=args.root_path,
            key_word=args.key_word,
            depth_solution=args.depth_solution,
            generate_method=args.generate_method,
            Depth_folder_name=args.depth_folder_name,
            rank=args.rank,
        )
    else:
        main()
