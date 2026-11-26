from train_core import *
import train_core as train_core_utils
import torch.nn as nn

GUI_Available = train_core_utils.GUI_Available

def _eval_err_dict(enable_extra_metric=False):
    err = {
        "gt_er": 0, "gt_et": 0,
        "batch_er": 0, "batch_et": 0,
        "total_er": 0, "total_et": 0,
        "batch_err_xyz": np.zeros(3), "batch_err_ypr": np.zeros(3),
        "total_err_xyz": np.zeros(3), "total_err_ypr": np.zeros(3),
    }
    if enable_extra_metric:
        err.update({"batch_extra": 0, "total_extra": 0})
    return err

def _get_testRecords_header():
    '''Record Header for CSV file
    
    [
        "test_ver", 
        "infer_mode", 
        "weight_epoch", 
        "r_range1",
        "t_range1",
        "r_range2",
        "t_range2",
        "subset_name",
        "scene_num",
        "batch_size",
        "calib_turns",
        "er",
        "et",
        "ypr",
        "xyz"
    ]
    '''
    header = [
        "test_ver", 
        "infer_mode", 
        "weight_epoch", 
        "r_range1",
        "t_range1",
        "r_range2",
        "t_range2",
        "subset_name",
        "scene_num",
        "batch_size",
        "calib_turns",
        "er",
        "et",
        "ypr",
        "xyz"
    ]
    return header

def exportTestResFromBatch(batch, output, test_dataset):
    frame_calib_res_list = []
    scene_idx_batch = batch["scene_idx_batch"]
    T_ini_batch = batch["T_lidar2vir_batch"]
    T_vir2_batch = batch["T_lidar2vir2_batch"]
    T_lidar2cam_batch = batch["T_lidar2cam_batch"]
    K_ini_batch = batch["K_ini_batch"]

    batch_num = int(scene_idx_batch.shape[0]) if hasattr(scene_idx_batch, "shape") else len(scene_idx_batch)
    for i in range(batch_num):
        T_lidar2cam = zv.tensor2Arr(T_lidar2cam_batch[i])
        scene_idx = int(scene_idx_batch[i])
        T_ini = zv.tensor2Arr(T_ini_batch[i])
        T_vir2 = zv.tensor2Arr(T_vir2_batch[i])
        infer_rvec = zv.tensor2Arr(output["rvec"][i])
        infer_tvec = zv.tensor2Arr(output["tvec"][i])
        infer_T = pose.rt2T(infer_rvec, infer_tvec)
        calib_T = np.matmul(infer_T, T_ini)
        K = zv.tensor2Arr(K_ini_batch[i])
        pcd_path = test_dataset.pcd_path_list[scene_idx]
        scene_name = zv.getFileNameFromPath(pcd_path).split(".")[0]
        T_err = np.matmul(T_lidar2cam, np.linalg.inv(calib_T))
        calib_err = pose.CalPoseErr_From_TRel(T_err)
        frame_calib_res_list.append({
            "scene_idx": scene_idx,
            "sim": 0,
            "scene_name": scene_name,
            "calib_T": calib_T,
            "T": calib_T,
            "calib_err": calib_err,
            "T_ini": T_ini,
            "T_vir2": T_vir2,
            "T_lidar2cam": T_lidar2cam,
            "K": K,
        })
    return frame_calib_res_list


def saveTestResToJson(test_res, save_dir=os.path.join("output", "TestResJson")):
    K = test_res["K"]
    T_gt = test_res["T_lidar2cam"]
    R_gt, t_gt = pose.T2rt(T_gt)
    T_cal = test_res["calib_T"]
    R_cal, t_cal = pose.T2rt(T_cal)
    T_ini = test_res["T_ini"]
    R_ini, t_ini = pose.T2rt(T_ini)
    scene_name = test_res["scene_name"]
    save_json_dict = {
        "cloudCamera": {
            "Intrinsic": {"K": K.tolist(), "D": [[0], [0], [0], [0], [0]]},
            "Extrinsic": {"R": R_gt.tolist(), "T": t_gt.reshape(3, 1).tolist()},
        },
        "calCamera": {
            "Intrinsic": {"K": K.tolist(), "D": [[0], [0], [0], [0], [0]]},
            "Extrinsic": {"R": R_cal.tolist(), "T": t_cal.reshape(3, 1).tolist()},
        },
        "iniCamera": {
            "Intrinsic": {"K": K.tolist(), "D": [[0], [0], [0], [0], [0]]},
            "Extrinsic": {"R": R_ini.tolist(), "T": t_ini.reshape(3, 1).tolist()},
        },
    }
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, scene_name + ".json")
    with open(save_path, "w") as f:
        json_str = json.dumps(save_json_dict, indent=4)
        json_str = zv.compactJsonArrays(json_str, keys=["guess_wVec", "guess_eulerVec", "K", "D", "R", "T"])
        f.write(json_str)


def test_DSTCalib_SB(rtime_ctrller: ZV_Runtime_Controller):
    # Runtime Configuration
    global GUI_Available
    config_loader = rtime_ctrller.config_loader
    model: DST_Calib = rtime_ctrller.model
    zvdataloader = rtime_ctrller.dataloader
    iter_calib = config_loader.get("iter_calib", 1)
    multi_frame_opt = config_loader.get("multi_frame_opt", False)
    test_ver = rtime_ctrller.version
    test_mode = config_loader.get("test_mode", "eva")
    dataset_name = zvdataloader.test_subset.name
    visual_process = config_loader.get("visual_process", False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if "GUI_Available" in config_loader:
        GUI_Available = bool(config_loader["GUI_Available"]) and train_core_utils.cv2 is not None
    train_core_utils.GUI_Available = GUI_Available

    # Load Checkpoint
    checkpoint_folder = config_loader["weight_folder"]
    checkpoint_name = config_loader["weight_name"]
    if os.path.exists(checkpoint_folder) == False:
        os.mkdir(checkpoint_folder)
    checkpoint_path = checkpoint_folder + "/" + checkpoint_name
    if os.path.exists(checkpoint_path) == False:
        checkpoint_path = checkpoint_folder + "/" + checkpoint_name
    if checkpoint_path:
        weight_epoch, _ ,_ = LoadCheckPoint(checkpoint_path, model)

    # Preparation
    test_loader = zvdataloader.test_dataloader
    test_subset = zvdataloader.test_subset
    scene_num = zvdataloader.getTestSceneNum()
    test_batch_num = 0
    
    projection_type = config_loader.get("projection_type", "Depth")
    result = _eval_err_dict()
    total_calib_res_list = []
    sim_calib_frame_list = []

    if multi_frame_opt:
        T_gt_arr = test_subset.generateTArrForProjections(
            batch=None,
            method="same_all_batch",
        )[0]
        test_subset.set_T_relative(T_gt_arr)
        test_subset.T_rel_persistent = True
        
    model.eval()
    
    with torch.no_grad():
        tqdm_console = tqdm(total=zvdataloader.getTestBatchLen(), desc="Test")
        with tqdm_console:
            for i, batch in enumerate(test_loader):
                scene_idx = test_subset.getSceneIdxFromBatchIdx(i)
                batch = test_subset.batchDeal(batch)
                label_batch = batch["label_batch"]
                est_T = batch["T_lidar2vir_batch"].clone()
                outputs = None

                for _iter in range(iter_calib):
                    batch["T_lidar2vir_batch"] = est_T
                    outputs = train_core_utils._forward_model(model, batch, test_mode if test_mode in ["eva", "calib"] else "eva")
                    outputs = train_core_utils._update_outputs_for_loss(batch, outputs, test_subset, projection_type)
                    train_core_utils._visualize_calibration_process(
                        batch, outputs, test_subset, scene_idx, i, projection_type,
                        visual_process=visual_process, title_after=f"After"
                    )
                    est_T = outputs.get("est_T", est_T)

                iter_est_xvec = outputs["xvec"]
                if multi_frame_opt:
                    sim_batch = dict(batch)
                    sim_batch["T_lidar2vir_batch"] = est_T
                    sim_batch["lip_batch"] = outputs["est_lip_batch"]
                    sim_nxt_output = train_core_utils._forward_model(
                        model,
                        sim_batch,
                        "eva",
                    )
                    sim_nxt_xvec = sim_nxt_output["xvec"]
                    sim_score = pose.CalPoseSimBatch_tensor_Indiv(
                        sim_nxt_xvec,
                        torch.zeros_like(sim_nxt_xvec),
                        print_err=False,
                    ) # (B,)
                    sim_calib_batch = {
                        "sim": sim_score,
                        "calib_xvec": iter_est_xvec,
                    }

                    sim_calib_frame = loss_utils.getSimCalibList_fromSimCalibBatch_Indiv(sim_calib_batch)
                    sim_calib_frame_list.extend(sim_calib_frame)

                
                iter_er, iter_et, iter_ypr, iter_xyz = pose.CalPoseErrBatch(
                    iter_est_xvec,
                    label_batch,
                    print_err=False,
                )
                
                batch_gt_er, batch_gt_et, _, _ = pose.CalPoseErrBatch(
                    label_batch,
                    torch.zeros_like(label_batch),
                    print_err=False,
                )

                result["batch_er"] = iter_er
                result["batch_et"] = iter_et
                result["batch_err_xyz"] = iter_xyz
                result["batch_err_ypr"] = iter_ypr
                result["total_er"] += iter_er
                result["total_et"] += iter_et
                result["gt_er"] += batch_gt_er
                result["gt_et"] += batch_gt_et
                result["total_err_xyz"] += iter_xyz
                result["total_err_ypr"] += iter_ypr


                # Add to total calibration results list
                try:
                    batch_calib_res_list = exportTestResFromBatch(batch, outputs, test_subset)

                    if multi_frame_opt:
                        for sim_i in range(len(sim_calib_frame)):
                            batch_calib_res_list[sim_i]["sim"] = sim_calib_frame[sim_i]["sim"]

                    total_calib_res_list += batch_calib_res_list      
                except KeyError:
                    pass
                    
                tqdm_console.set_postfix_str(
                    ("Scene({:03d}|{:03d}), er:{:.3f}, et:{:.3f}").format(
                        scene_idx + 1, scene_num, iter_er, iter_et
                    )
                )
                tqdm_console.update(1)
                test_batch_num += 1

    test_batch_num = max(test_batch_num, 1)
    for key in ["total_er", "total_et", "gt_er", "gt_et", "total_extra"]:
        if key in result:
            result[key] /= test_batch_num
    result["total_err_xyz"] /= test_batch_num
    result["total_err_ypr"] /= test_batch_num

    if multi_frame_opt:
        if len(sim_calib_frame_list) < 5:
            raise ValueError(
                "multi_frame_opt requires at least 5 calibration frames "
            )
        raw_er = result["total_er"]
        raw_et = result["total_et"]
        multiframe_opt_res = loss_utils.MFOptimization(
            sim_calib_frame_list,
            ratio=0.2,
        )
        opt_T = multiframe_opt_res["opt_T"]
        opt_err = pose.CalPoseErr_T(opt_T,T_gt_arr)
        result["raw_total_er"] = raw_er
        result["raw_total_et"] = raw_et
        result["total_er"] = opt_err["er"]
        result["total_et"] = opt_err["et"]
        result["total_err_ypr"] = opt_err["ypr"]
        result["total_err_xyz"] = opt_err["xyz"]

        gt_err_dict = pose.CalPoseErr_From_TRel(T_gt_arr)
        print("================== Multi-Frame Optimization Results ==================")
        print(
            f"{RED}CalibRes (GT) , er:{gt_err_dict['er']:.4f},"
            f"et:{gt_err_dict['et']:.4f}{RESET}"
        )
        print(f"{RED}CalibRes (RAW), er:{raw_er:.4f},et:{raw_et:.4f}.{RESET}")
        print(
            f"{RED}CalibRes (Opt), er:{result['total_er']:.4f},"
            f"et:{result['total_et']:.4f}.{RESET}"
        )

    logger = get_logger(test_ver, "output/test.log")
    logger.info(
        ("Test avg err:|gt_er:{:.3f}, gt_et:{:.3f}|er:{:.3f},et:{:.3f}").format(
            result["gt_er"], result["gt_et"], result["total_er"], result["total_et"]
        )
    )
    print(f"{BLUE}Test avg err:{RESET} er:{result['total_er']:.3f}, et:{result['total_et']:.3f}")

    if 0:
        for test_res in total_calib_res_list:
            saveTestResToJson(test_res, save_dir=os.path.join(OUTPUT_DEFAULT_PATH, "TestResJson"))
    
    record_dict = {
            "test_ver":test_ver,
            "infer_mode":"SB",
            "weight_epoch":weight_epoch,
            "r_range1":config_loader["r_Euler_range"],
            "t_range1":config_loader["t_range"],
            "r_range2":config_loader["r_Euler_range2"],
            "t_range2":config_loader["t_range2"],
            "subset_name":zvdataloader.test_subsets_path,
            "scene_num":scene_num,
            "batch_size":zvdataloader.batch_size,
            "calib_turns":"-",
            "er":result["total_er"],
            "et":result["total_et"],
            "ypr":result["total_err_ypr"],
            "xyz":result["total_err_xyz"]
        }
        
    # write to CSV
    record_header = _get_testRecords_header()
    zv.record2CSV_byDict(record_dict,record_header,csv_path=os.path.join(OUTPUT_DEFAULT_PATH, "test.csv"))
    
    return result["total_er"], result["total_et"], result


def test_DSTCalib_Joint(rtime_ctrller: ZV_Runtime_Controller, min_iter_batch=60, mode="SB*"):
    
    config_loader = rtime_ctrller.config_loader
    joint_mode = {
        "SB*": "SB",
        "PE": "PE",
        "PE+SB*": "PE+SB",
    }.get(mode)
    if joint_mode is None:
        raise ValueError(
            f"Unsupported joint calibration mode: {mode}. "
            "Expected one of: SB*, PE, PE+SB*."
        )
    
    # create dataloader
    zvdataloader = rtime_ctrller.dataloader
    test_ver = rtime_ctrller.version

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # Load Runtime Controller Info 
    test_ver = rtime_ctrller.version
    dataset = rtime_ctrller.dataset_name
    
    model: DST_Calib = rtime_ctrller.model
    model.to(device)
    print(f"{BLUE}Calibration model: {model.__class__.__name__}{RESET}")
    
    # Training process
    total_epochs = zvdataloader.epoch_num
    scene_num = zvdataloader.getTestSceneNum()
    batch_size = zvdataloader.batch_size
    total_batch_num=zvdataloader.getTestBatchLen()
    if total_epochs <= 0:
        raise ValueError("The configured number of calibration epochs must be positive.")
    if total_batch_num <= 0:
        raise ValueError("The test dataloader is empty; joint calibration cannot run.")
    if min_iter_batch < 5:
        raise ValueError(
            "min_iter_batch must be at least 5"
        )
    
    # load checkpoint 
    checkpoint_folder = config_loader["weight_folder"]
    checkpoint_name = config_loader["weight_name"]
    if os.path.exists(checkpoint_folder) == False:
        os.makedirs(checkpoint_folder, exist_ok=True)
    checkpoint_path = checkpoint_folder + "/" + checkpoint_name
    if os.path.exists(checkpoint_path) == False:
        checkpoint_path = (checkpoint_folder + "/" + checkpoint_name)[:-4]+"_"+dataset+test_ver+"_BS"+str(batch_size)+"_scene"+str(scene_num)+"_batch"+str(total_batch_num)+".pth"
    print(f"{MAGENTA}rtime_ctrller.pretrained:{rtime_ctrller.pretrained}{RESET}")
    start_epoch, best_model_eva_score, checkpoint = rtime_ctrller.loadCheckPoint(checkpoint_path, model, rtime_ctrller.optimizer,scheduler = rtime_ctrller.scheduler,pretrained = rtime_ctrller.pretrained)
    
    
    # dataset 
    print(f"{BLUE}VERSION:{RESET} {GREEN}{test_ver}{RESET}")
    print("calib scene num : ", zvdataloader.getTestSceneNum())
    print("batch size: ", zvdataloader.batch_size)

    test_records=[] #testinfo

    total_calib_er = 0
    total_calib_et = 0
    total_calib_err_ypr = np.array([0.0,0.0,0.0])    
    total_calib_err_xyz = np.array([0.0,0.0,0.0])

    test_subset = zvdataloader.test_subset
    
    for epoch in range(start_epoch,start_epoch + total_epochs):
        test_records.clear() #clear
        test_records.append(test_ver)   
        test_records.append(dataset)
        test_records.append(scene_num)
        test_records.append(batch_size) 
        test_records.append(total_batch_num)    
        test_records.append(epoch+1)
        
        
        print("\n----------------------------------------------------")    
        
        
        # Reset Pose Estimator and its optimizer and Scheduler
        model.reset_PoseEstimator()
        
        # -------------------------- Calibration using Pose Estimation Module --------------------------- #
        test_subset = zvdataloader.test_subset
        
        T_vir1_to_vir2_arr_batch = test_subset.generateTArrForProjections(batch = None,method = "same_all_batch")   # same T per batch
        T_vir1_to_vir2_forCalib_0 = T_vir1_to_vir2_arr_batch[0]
        T_vir1_to_vir2_after_calibed = T_vir1_to_vir2_forCalib_0

        gt_err = pose.CalPoseErr_From_TRel(T_vir1_to_vir2_forCalib_0)
        gt_er = gt_err["er"]
        gt_et = gt_err["et"]
        print(f"{RED}gt_er:{gt_er:.4f},gt_et:{gt_et:.4f}{RESET}")
        visual_process = config_loader.get("visual_process", False)
        
        T_vir1_to_vir2_arr_input = T_vir1_to_vir2_after_calibed.copy()
        test_subset.set_T_relative(T_vir1_to_vir2_arr_input)
        test_subset.T_rel_persistent = True
        
        if joint_mode == "SB":
            print("T_vir1_to_vir2_arr_input:\n", T_vir1_to_vir2_arr_input)
            fin_calib_T,fin_avg_err = jointCalib(rtime_ctrller,model,epoch,T_vir1_2_vir2_arr=T_vir1_to_vir2_arr_input,
                                                                                            min_iter_batch=min_iter_batch,method = "SB",visual_process=visual_process)
        elif joint_mode == "PE":
            print("T_vir1_to_vir2_arr_input:\n", T_vir1_to_vir2_arr_input)
            fin_calib_T,fin_avg_err = jointCalib(rtime_ctrller,model,epoch,T_vir1_2_vir2_arr=T_vir1_to_vir2_arr_input,
                                                                                            min_iter_batch=min_iter_batch,method = "PE",visual_process=visual_process)
            
        elif joint_mode == "PE+SB":
            print("T_vir1_to_vir2_arr_input:\n", T_vir1_to_vir2_arr_input)
            fin_calib_T,fin_avg_err = jointCalib(rtime_ctrller,model,epoch,T_vir1_2_vir2_arr=T_vir1_to_vir2_arr_input,
                                                                                            min_iter_batch=min_iter_batch,method = "PE",visual_process=visual_process)
            fin_calib_rvec,fin_calib_tvec = pose.T2rt(fin_calib_T)
            T_SB = pose.rt2T(fin_calib_rvec,fin_calib_tvec)

            T_vir1_to_vir2_after_calibed = np.matmul(T_vir1_to_vir2_after_calibed,np.linalg.inv(T_SB))

            T_vir1_to_vir2_arr_input = T_vir1_to_vir2_after_calibed.copy()
            print("T_vir1_to_vir2_arr_input:\n", T_vir1_to_vir2_arr_input)
            test_subset.set_T_relative(T_vir1_to_vir2_arr_input)
            test_subset.T_rel_persistent = True
            test_subset.updateTmpJson()
            fin_calib_T,fin_avg_err = jointCalib(rtime_ctrller,model,epoch,T_vir1_2_vir2_arr=T_vir1_to_vir2_arr_input,
                                                                                            min_iter_batch=min_iter_batch,method = "SB",visual_process=visual_process)

        total_calib_er += fin_avg_err["er"]
        total_calib_et += fin_avg_err["et"]
        total_calib_err_ypr += fin_avg_err["ypr"]
        total_calib_err_xyz += fin_avg_err["xyz"]
        

    total_calib_er /= total_epochs
    total_calib_et /= total_epochs
    total_calib_err_ypr /= total_epochs
    total_calib_err_xyz /= total_epochs

    # Print Optimization Result
    print(f"{MAGENTA}Sequence Optimization Results{RESET}| er:{total_calib_er:.4f},et:{total_calib_et:.4f} | err_ypr:{total_calib_err_ypr},err_xyz:{total_calib_err_xyz}")
    
    #record evaluation results
    test_records.append(model.__class__.__name__)
    test_records.append(ZV_Dataset._RGB_TENSOR_TYPE)#increased 250719
    test_records.append(config_loader["r_Euler_range"])
    test_records.append(config_loader["t_range"])
    test_records.append(config_loader["r_Euler_range2"])
    test_records.append(config_loader["t_range2"])
    test_records.append(zvdataloader.test_subsets_path)
    test_records.append(batch_size)
    record_dict = {
        "test_ver":test_ver,
        "infer_mode":mode,
        "weight_epoch":start_epoch,
        "r_range1":config_loader["r_Euler_range"],
        "t_range1":config_loader["t_range"],
        "r_range2":config_loader["r_Euler_range2"],
        "t_range2":config_loader["t_range2"],
        "subset_name":zvdataloader.test_subsets_path,
        "scene_num":scene_num,
        "batch_size":batch_size,
        "calib_turns":total_epochs,
        "er":total_calib_er,
        "et":total_calib_et,
        "ypr":total_calib_err_ypr,
        "xyz":total_calib_err_xyz
    }
    
    # write to CSV
    record_metrics_key = _get_testRecords_header()
    zv.record2CSV_byDict(record_dict,record_metrics_key,csv_path=os.path.join(OUTPUT_DEFAULT_PATH, "test.csv"))

def jointCalib(rtime_ctrller:ZV_Runtime_Controller,model:DST_Calib,epoch, T_vir1_2_vir2_arr = np.array([None]),
                min_iter_batch = 60,
                method = "SB",visual_process = True):      
    
    # create dataloader
    zvdataloader = rtime_ctrller.dataloader
    
    test_loader = zvdataloader.test_dataloader
    test_subset = zvdataloader.test_subset

    total_epochs = zvdataloader.epoch_num
    scene_num = zvdataloader.getTestSceneNum()

    
    test_loader_batch_num = zvdataloader.getTestBatchLen()
    if test_loader_batch_num <= 0:
        raise ValueError("The test dataloader is empty; joint calibration cannot run.")
    if min_iter_batch > test_loader_batch_num:
        total_batch_num = int((min_iter_batch // test_loader_batch_num + 1) * test_loader_batch_num)
    else:
        total_batch_num = min_iter_batch

    
    if 1:
        # loss configuration
        cloud_sim_loss = loss_utils.ChamferDis_Loss_PE()
        ini_guess_tvec_loss = loss_utils.TransLoss_SelfSuper()
        eva_loss = loss_utils.CalibrationErr_SelfSuper_EvaGuided()

    if method == "SB":
        model.eval()
    else:
        model.train()
    
    rtime_ctrller.optimizer_calib, rtime_ctrller.scheduler_calib = OneCycleScheduler_Func_Calib(model.pose_estimator, scene_num, max(total_batch_num, 1), epoch)
    
    if 1:
        if (
            T_vir1_2_vir2_arr is None
            or (
                isinstance(T_vir1_2_vir2_arr, np.ndarray)
                and T_vir1_2_vir2_arr.dtype == object
                and T_vir1_2_vir2_arr.item() is None
            )
        ):
            test_subset = zvdataloader.test_subset
            T_vir1_2_vir2_arr = test_subset.generateTArrForProjections(batch = None,method = "same_all_batch")[0]   # relative T 

        gt_err = pose.CalPoseErr_From_TRel(T_vir1_2_vir2_arr)
        gt_er = gt_err["er"]
        gt_et = gt_err["et"]
        

    # calibraiton result
    total_er = 0
    total_et = 0

    test_batch_num = 0  

    sim_calib_frame_list = []

    tqdm_console = tqdm(total=total_batch_num,desc='Calib')

    with tqdm_console:
        tqdm_console.set_description_str('Calib SS: {:03d}|{:03d}'.format(epoch+1,total_epochs))
        while (test_batch_num < total_batch_num):
            for i,batch_load in enumerate(test_loader):
                if test_batch_num >= total_batch_num:
                    break
                
                scene_idx = test_subset.getSceneIdxFromBatchIdx(i)
                batch = test_subset.batchDeal(batch_load)
                
                label_batch = batch["label_batch"]
                pcd_batch = batch["pcd_batch"]
                T_lidar2vir_batch = batch["T_lidar2vir_batch"]
                rgb_tensor = batch["rgb_tensor"]
                vir_camera = test_subset.getPVirCamera()
                
                
                uncalibed_rendred_image = vir_camera.renderLiDARDepthOnImageTensor(T_lidar2vir_batch[0],pcd_batch[0],
                                                                                    rgb_tensor,z_thr=-10)
                            
                infer_iter = 0

                # set optimizer grad
                if rtime_ctrller.optimizer is not None:
                    rtime_ctrller.optimizer.zero_grad()
                rtime_ctrller.optimizer_calib.zero_grad()
                
                outputs = train_core_utils._forward_model(model, batch, "calib")
                
                # Calculate Iterative Calibration Result in this iteration
                iter_est_xvec = outputs["xvec"]
                est_T = outputs["est_T"]
                
                # ss calib
                calib_er,calib_et,_,_ = pose.CalPoseErrBatch(iter_est_xvec,label_batch,print_err=False)
                
                

                # sim calib
                sim_iter_calib_xvec = outputs["sim_calib_xvec"]
                sim_iter_calib_er, sim_iter_calib_et,_,_ = pose.CalPoseErrBatch(sim_iter_calib_xvec,label_batch,print_err=False)
                
                
                if (visual_process):
                    icalibed_rendered_image = vir_camera.renderLiDARDepthOnImageTensor(est_T[0],pcd_batch[0],
                                                                                        rgb_tensor,z_thr=-10)
                    
                    concat_render_image = zv.concatImage(uncalibed_rendred_image,icalibed_rendered_image,
                                    "[ Scene {:03d} Batch {:03d} ] Before".format(scene_idx+1,i+1),"After (Rank {:01d})".format(0),total_width=2000)
                    cv2.imshow("render", concat_render_image)
                    cv2.waitKey(1)

                
                # ----------------- Calculate Loss and Update Optimizer ------------ #
                joint_loss = torch.tensor(0)
                if method == "SB":
                    sim_score = outputs["sim_sim"] # (B,)

                    sim_calib_batch = {
                        "sim": sim_score,
                        "calib_xvec": sim_iter_calib_xvec,
                    }

                    sim_calib_frame = loss_utils.getSimCalibList_fromSimCalibBatch_Indiv(sim_calib_batch)
                    sim_calib_frame_list.extend(sim_calib_frame)
                    batch_er = sim_iter_calib_er
                    batch_et = sim_iter_calib_et
                elif method == "PE":
                    cloud_sim, cd_loss_list = cloud_sim_loss(batch, outputs)
                    joint_loss = 1*cloud_sim + 1*ini_guess_tvec_loss(batch, outputs) + 0*eva_loss(batch, outputs)
                    
                    sim_calib_batch = {
                        "sim": cd_loss_list, 
                        "calib_xvec": iter_est_xvec,
                    }

                    sim_calib_frame = loss_utils.getSimCalibList_fromSimCalibBatch_Indiv(sim_calib_batch)
                    sim_calib_frame_list.extend(sim_calib_frame)

                    batch_er = calib_er
                    batch_et = calib_et
                    
                    joint_loss.backward()
                    
                    rtime_ctrller.optimizer_calib.step()
                    if rtime_ctrller.scheduler_batch:
                        rtime_ctrller.scheduler_calib.step()

                total_er += batch_er
                total_et += batch_et
                
                infer_iter += 1
                
                batch_loss = joint_loss.item()
                tqdm_console.set_postfix_str(
                "Scene({:03d}|{:03d}),loss:{:.3f}|er:{:.3f}, et:{:.3f}".format(scene_idx+1,
                                                                                scene_num,batch_loss, 
                                                                                batch_er,batch_et))
                
                tqdm_console.update()
                test_batch_num += 1
                

    if rtime_ctrller.scheduler_batch == False:
        rtime_ctrller.scheduler_calib.step()

    tqdm_console.close()    

    test_subset.T_rel_persistent = False
    test_subset.updateTmpJson()

    # --------------- Calibration Error Settlement ------------- #
    total_er /= test_batch_num
    total_et /= test_batch_num
    
    # Multi-Frame Optimization
    multiframe_opt_res = loss_utils.MFOptimization(sim_calib_frame_list, ratio = 0.2)
    
    fin_calib_T = multiframe_opt_res["opt_T"]
    fin_err = pose.CalPoseErr_T(fin_calib_T, T_vir1_2_vir2_arr)
    fin_er = fin_err["er"]
    fin_et = fin_err["et"]
    
    print(f"{RED}CalibRes (GT) , er:{gt_er:.4f},et:{gt_et:.4f}{RESET}")
    print(f"{RED}CalibRes (RAW), er:{total_er:.4f},et:{total_et:.4f}.{RESET}")
    print(f"{BLUE}CalibRes (Opt), er:{fin_er:.4f},et:{fin_et:.4f}.{RESET}")
    
    return fin_calib_T,fin_err
