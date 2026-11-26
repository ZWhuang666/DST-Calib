from train_core import *
import argparse
import gc
import json
import os
import platform
import sys
import time
import cv2


class User_Runtime_Controller(ZV_Runtime_Controller):
    """Extension point for project-specific runtime behavior."""
    pass


def ConfigRuntimeController(config_json_path, test_ver="DST_Calib"):
    with open(config_json_path, "r") as config_file:
        config_loader = json.load(config_file)

    zvdataloader = ZV_Dataset_Loader()
    zvdataloader.projection_channel = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setDSTCalibDatasetMode()
    model = DST_Calib(
        grid_size=config_loader.get("grid_size", 5),
        dropout_rate=config_loader.get("dropout_rate", 0.3),    
        dropout_rate2=config_loader.get("dropout_rate2", 0.05)
    ).to(device)

    rtime_ctrller = ZV_Runtime_Controller(test_ver, model, dataloader=zvdataloader)
    rtime_ctrller.dataset_name = config_loader.get("dataset_name", "KITTI")
    rtime_ctrller.config_loader = config_loader
    rtime_ctrller.pretrained = config_loader.get("pretrained", False)
    rtime_ctrller.sel_subsets_Func = GetSubsets_All
    rtime_ctrller.scheduler_batch = True
    rtime_ctrller.scheduler_setFunc = OneCycleScheduler_Func_Calib

    r_loss = loss_utils.RotLoss()
    t_loss = loss_utils.TransLoss()
    p_loss = loss_utils.PointCloudDisLoss()
    rtime_ctrller.configLoss([r_loss, t_loss, p_loss], [[30, 0.1, 2.0]])  
    
    return rtime_ctrller


def train_byTestMode_Json(test_ver, config_json_path):

    rtime_ctrller = ConfigRuntimeController(config_json_path, test_ver=test_ver)
    rtime_ctrller.dataloader.loadFromJsonFile(config_json_path)
    rtime_ctrller.dataloader.train_subset.loadDataFromPaths(sameTperBatch=False)
    rtime_ctrller.dataloader.val_subset.loadDataFromPaths(sameTperBatch=False)
    rtime_ctrller.dataloader.createTrainDataLoader(num_workers=8, pre_loading=True, shuffle=True,
                                                   persistend_workers=True, pre_fetch_batch=2, pin_memory=True, timeout=0)
    rtime_ctrller.dataloader.createValDataLoader(num_workers=8, pre_loading=True, shuffle=True,
                                                 persistend_workers=True, pre_fetch_batch=2, pin_memory=True, timeout=0)
    train_DST_Calib(
        rtime_ctrller,
        visual_process=rtime_ctrller.config_loader.get("visual_process", False),
        mode="eva",
    )

def batchTrainWithJsonPathList(json_path_list, test_mode):
    for json_path in json_path_list:
        train_byTestMode_Json(test_mode, json_path)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        time.sleep(15)


def batchCalibByAutoFunc():
    
    test_mode = "DST_Calib"
    
    json_path_list = [
        r"/path/to/train.json"  # place your json path
    ]
    print("begin")
    batchTrainWithJsonPathList(json_path_list, test_mode=test_mode)


def parse_args():
    parser = argparse.ArgumentParser(description="Train or calibrate DSTCalib models from a JSON runtime config.")
    parser.add_argument(
        "--config",
        help="Path to the runtime JSON config",
    )
    parser.add_argument(
        "--test-ver",
        default="DST_Calib",
        help="Training/calibration version name .",
    )
    return parser.parse_args()


if __name__ == "__main__":
    SYSTYPE = platform.system()
    print(f"{MAGENTA}OS:{SYSTYPE}{RESET}")
    args = parse_args()
    if args.config:
        train_byTestMode_Json(args.test_ver, args.config)
    else:
        batchCalibByAutoFunc()
