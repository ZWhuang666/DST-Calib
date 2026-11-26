from DST_Calib import *
from tqdm import tqdm
import loss as loss_utils
import csv
import json
import os
import numpy as np
import torch
import cv2
import logging
import os
import sys

GUI_Available = cv2 is not None
OUTPUT_DEFAULT_PATH = "output/"
if os.path.exists(OUTPUT_DEFAULT_PATH) == False:
    os.mkdir(OUTPUT_DEFAULT_PATH)

def get_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def isExtraMetricEnabled(rtime_ctrller):
    return callable(getattr(rtime_ctrller, "extra_metric_func", None))


def calExtraMetric(rtime_ctrller, pred_xvec, label_xvec):
    return rtime_ctrller.extra_metric_func(pred_xvec, label_xvec)


def CalModelEvaScore(train_er, train_et, val_er, val_et):
    train_score = train_er + 10 * train_et
    val_score = val_er + 10 * val_et
    return val_score + 0.4 * train_score + 0.4 * abs(train_score - val_score)

def GetSubsets_Routine(input_subsets, epoch):
    ptr_ind = (epoch % len(input_subsets)) // 2
    return [input_subsets[int(ptr_ind * 2)], input_subsets[int(ptr_ind * 2 + 1)]]

def GetSubsets_All(input_subsets, epoch):
    return input_subsets

logging.basicConfig(level=logging.INFO)
def get_logger(name,filepath,level=logging.DEBUG,format='[%(levelname)s]:%(name)s, %(asctime)s, %(message)s',mode='a'):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if filepath:
        if os.path.exists(os.path.dirname(filepath)):
            fh = logging.FileHandler(filename=filepath,mode=mode)
            fh.setFormatter(logging.Formatter(format))
            fh.setLevel(level)
            logger.addHandler(fh)
        else:
            logger.warning("path of logfile {} is not a valid file, considered as no logfile.".format(os.path.abspath(filepath)))
    return logger


class ZV_Runtime_Controller(object):
    def __init__(self, version="DST_Calib", model: DST_Calib = None, dataloader: ZV_Dataset_Loader = None):
        self.version = version
        self.pretrained = False
        self.max_epoch = 200
        self.params = {}
        self.config_loader = {}
        self.model = model
        self.dataloader = dataloader
        self.dataset_name = "Untitled"
        self.optimizer = None
        self.scheduler = None
        self.optimizer_calib = None
        self.scheduler_calib = None
        self.loss_List = []
        self.loss_Weights = []
        self.scheduler_setFunc = None
        self.sel_subsets_Func = GetSubsets_All

    def setDataLoader(self, dataloader: ZV_Dataset_Loader, dataset_name="Untitled"):
        self.dataloader = dataloader
        self.dataset_name = dataset_name

    def configLoss(self, loss_list: list, loss_weights: list, switch_epoch: list = None, epoch_dist=0):
        self.loss_List = loss_list
        self.loss_Weights = loss_weights
        self.switchEpoch = switch_epoch
        self.epochDist = epoch_dist

    def setScheduler(self, scene_num, batch_num, epoch):
        assert self.scheduler_setFunc is not None, "scheduler_setFunc is None, please check your configuration"
        self.optimizer, self.scheduler = self.scheduler_setFunc(self.model, scene_num, batch_num, epoch)

    def calModelEvaScore(self, train_er, train_et, val_er, val_et):
        return CalModelEvaScore(train_er, train_et, val_er, val_et)

    def loadCheckPoint(self, checkpoint_path, model, optimizer=None, scheduler=None, pretrained=False):
        return LoadCheckPoint(checkpoint_path, model, optimizer, scheduler, pretrained=pretrained)

    def saveCheckPoint(self, model, optimizer, scheduler, epoch, best_loss, best_model_eva_score, checkpoint_path,
                       rand_range_dict={}, save_info_color=BLUE):
        SaveCheckPoint(model, optimizer, scheduler, epoch, best_loss, best_model_eva_score, checkpoint_path,
                       rand_range_dict=rand_range_dict, save_info_color=save_info_color)


def OneCycleScheduler_Func_Calib(model: DST_Calib, scene_num, batch_num, epoch):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=8e-4,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,
        epochs=210,
        steps_per_epoch=max(batch_num, 1),
        pct_start=0.25,
        anneal_strategy="cos",
        div_factor=5,
        final_div_factor=2,
    )
    return optimizer, scheduler

def SaveCheckPoint(model, optimizer, scheduler, epoch, loss, model_eva_score, filepath,
                   rand_range_dict={}, save_info_color=BLUE, **_):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "model_eva_score": model_eva_score,
        "loss": loss,
        "rand_range_dict": rand_range_dict,
    }
    torch.save(checkpoint, filepath)
    print(f"{save_info_color}Checkpoint saved at {filepath}{RESET}")


def LoadCheckPoint(filepath, model, optimizer=None, scheduler=None, pretrained=False):
    if not os.path.exists(filepath):
        print(f"{YELLOW}No checkpoint found at {filepath}. Starting from scratch.{RESET}")
        return 0, float("inf"), None

    checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if not pretrained:
        if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", -1) + 1
        model_eva_score = checkpoint.get("model_eva_score", float("inf"))
        print(f"{GREEN}Checkpoint loaded from {filepath}. Resuming from epoch {start_epoch}.{RESET}")
        return start_epoch, model_eva_score, checkpoint

    print(f"{GREEN}Checkpoint loaded from {filepath}. Restart from epoch 0.{RESET}")
    return 0, float("inf"), checkpoint


def _init_err(enable_extra_metric=False, include_gt=False):
    err = {"batch_er": 0, "batch_et": 0, "total_er": 0, "total_et": 0}
    if include_gt:
        err.update({"gt_er": 0, "gt_et": 0})
    if enable_extra_metric:
        err.update({"batch_extra": 0, "total_extra": 0})
    return err


def _forward_model(model, batch, mode):
    try:
        return model(batch, mode=mode)
    except TypeError:
        return model(batch)


def _cv2_visual_enabled(visual_process):
    return bool(visual_process and GUI_Available and cv2 is not None)


def _show_cv2_window(window_name, image):
    global GUI_Available
    if not _cv2_visual_enabled(True):
        return
    try:
        cv2.imshow(window_name, image)
        cv2.waitKey(1)
    except cv2.error as err:
        GUI_Available = False
        print(f"{YELLOW}OpenCV GUI is unavailable, visual_process has been disabled. {err}{RESET}")


def _visualize_calibration_process(batch, outputs, dataset, scene_idx, batch_idx,
                                   projection_type="Depth", visual_process=False,
                                   title_after="After"):
    if not _cv2_visual_enabled(visual_process):
        return
    if "pcd_batch" not in batch or "T_lidar2vir_batch" not in batch or "est_T" not in outputs:
        return
    try:
        vir_camera = dataset.getPVirCamera()
        pcd_batch = batch["pcd_batch"]
        rgb_tensor = batch.get("rgb_tensor", batch.get("rgb_batch", None))
        if rgb_tensor is None:
            return
        if projection_type == "Depth":
            uncalibed_rendered_image = vir_camera.renderLiDARDepthOnImageTensor(
                batch["T_lidar2vir_batch"][0], pcd_batch[0], rgb_tensor, z_thr=-10
            )
            calibed_rendered_image = vir_camera.renderLiDARDepthOnImageTensor(
                outputs["est_T"][0], pcd_batch[0], rgb_tensor, z_thr=-10
            )
        elif projection_type == "Intensity" and "intensity_batch" in batch:
            uncalibed_rendered_image = vir_camera.renderLiDARIntensityOnImageTensor(
                batch["T_lidar2vir_batch"][0], pcd_batch[0], batch["intensity_batch"][0], rgb_tensor, z_thr=-10
            )
            calibed_rendered_image = vir_camera.renderLiDARIntensityOnImageTensor(
                outputs["est_T"][0], pcd_batch[0], batch["intensity_batch"][0], rgb_tensor, z_thr=-10
            )
        else:
            return
        concat_render_image = zv.concatImage(
            uncalibed_rendered_image,
            calibed_rendered_image,
            "Before",
            title_after,
            total_width=1200,
        )
        _show_cv2_window("render", concat_render_image)
        
    except Exception as err:
        print(f"{YELLOW}Calibration visualization skipped: {err}{RESET}")


def _update_outputs_for_loss(batch, outputs, train_subset, projection_type="Depth"):
    if "est_pcd" in outputs and "est_lip_batch" in outputs:
        return outputs

    pcd_batch = batch["pcd_batch"]
    pcd_trans_batch = batch["pcd_trans_batch"]
    label_batch = batch["label_batch"]
    vir_camera = train_subset.getPVirCamera()
    iter_est_T = pose.batch_rtTensor2TmatTensor(outputs["rvec"], outputs["tvec"])
    est_T = iter_est_T.bmm(batch["T_lidar2vir_batch"].detach())
    est_lip_batch = batch["lip_batch"]
    est_pcd = pcd_trans_batch
    if projection_type == "Depth":
        est_lip_batch, est_pcd = vir_camera.batch_projectLiDARDepth(est_T, pcd_batch)
    outputs["iter_est_T"] = iter_est_T
    outputs["est_T"] = est_T
    outputs["est_pcd"] = est_pcd
    outputs["est_lip_batch"] = est_lip_batch
    return outputs


def _select_loss_triplet(rtime_ctrller):
    loss_list = rtime_ctrller.loss_List
    loss_weights = rtime_ctrller.loss_Weights
    loss_weights = loss_weights[0] if len(loss_weights) > 0 and isinstance(loss_weights[0], list) else loss_weights
    return loss_list[0], loss_list[1], loss_list[2], loss_weights[:3]


def _write_train_csv_record(rtime_ctrller, zvdataloader, epoch, record):
    csv_path = os.path.join(OUTPUT_DEFAULT_PATH, "train.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    enable_extra_metric = isExtraMetricEnabled(rtime_ctrller)
    header = [
        "test_Ver", "Dataset", "Scene_Num", "Batch_Size", "Batches", "Epoch",
        "Loss", "Train_Er", "Train_Et",
        "Val_Er", "Val_Et"
    ]
    row = [
        rtime_ctrller.version,
        rtime_ctrller.dataset_name,
        zvdataloader.getTrainSceneNum(),
        zvdataloader.batch_size,
        zvdataloader.getTrainBatchLen(),
        epoch + 1,
        record["loss"],
        record["train_er"],
        record["train_et"],
        record["val_er"],
        record["val_et"]
    ]
    
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

def train_DST_Calib(rtime_ctrller: ZV_Runtime_Controller, 
                    visual_process=False, mode="eva"):
    global GUI_Available
    config_loader = rtime_ctrller.config_loader
    if "GUI_Available" in config_loader:
        GUI_Available = bool(config_loader["GUI_Available"]) and cv2 is not None
    projection_type = config_loader.get("projection_type", "Depth")
    zvdataloader = rtime_ctrller.dataloader
    train_loader = zvdataloader.train_dataloader
    train_subset = zvdataloader.train_subset
    val_subset = zvdataloader.val_subset
    model: DST_Calib = rtime_ctrller.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_epochs = zvdataloader.epoch_num
    batch_size = zvdataloader.batch_size
    scene_num = zvdataloader.getTrainSceneNum()
    total_batch_num = zvdataloader.getTrainBatchLen()

    # load checkpoint 
    checkpoint_folder = config_loader["weight_folder"]
    checkpoint_name = config_loader["weight_name"]
    if os.path.exists(checkpoint_folder) == False:
        os.mkdir(checkpoint_folder)
    checkpoint_path = checkpoint_folder + "/" + checkpoint_name
    if os.path.exists(checkpoint_path) == False:
        checkpoint_path = checkpoint_folder + "/" + checkpoint_name
    os.makedirs(checkpoint_folder, exist_ok=True)

    rtime_ctrller.setScheduler(scene_num, max(total_batch_num, 1), 0)
    start_epoch, best_model_train_score, checkpoint = rtime_ctrller.loadCheckPoint(
        checkpoint_path, model, rtime_ctrller.optimizer, scheduler=rtime_ctrller.scheduler, pretrained=rtime_ctrller.pretrained
    )
    
    best_model_train_score = float('inf') 
    best_model_calib_score = float("inf")
    
    r_loss, t_loss, p_loss, loss_weights = _select_loss_triplet(rtime_ctrller)
    alpha, beta, gama = loss_weights[:3]
    
    logger = get_logger(rtime_ctrller.version, "output/train.log")
    current_lr_history = []
    grad_norm_history = []
    train_loss_history = []
    Et_history = []
    Er_history = []
    calib_Et_history = []
    calib_Er_history = []

    for epoch in range(start_epoch, start_epoch + total_epochs):
        print("\n----------------------------------------------------")    
        print(f"{YELLOW}Epoch{RESET} [{epoch+1}] start")

        model.train()
        train_err = _init_err(include_gt=True)
        total_loss = 0
        total_r_loss = 0
        total_t_loss = 0
        total_p_loss = 0
        train_batch_num = 0

        tqdm_console = tqdm(total=total_batch_num, desc="Train")
        with tqdm_console:
            for i, batch in enumerate(train_loader):
                batch = train_subset.batchDeal(batch)
                scene_idx = train_subset.getSceneIdxFromBatchIdx(i)
                label_batch = batch["label_batch"]
                outputs = _forward_model(model, batch, mode)
                outputs = _update_outputs_for_loss(batch, outputs, train_subset, projection_type)
                _visualize_calibration_process(
                    batch, outputs, train_subset, scene_idx, i, projection_type,
                    visual_process=visual_process, title_after="After"
                )

                iter_est_xvec = outputs["xvec"]
                iter_er, iter_et,_,_ = pose.CalPoseErrBatch(iter_est_xvec, label_batch, print_err=False)
                gt_er, gt_et,_,_ = pose.CalPoseErrBatch(label_batch, torch.zeros((batch_size, 6)).to(label_batch),print_err=False)
                train_err["batch_er"] = iter_er
                train_err["batch_et"] = iter_et
                train_err["gt_er"] += gt_er
                train_err["gt_et"] += gt_et
                
                loss_r = r_loss(batch, outputs)
                loss_t = t_loss(batch, outputs)
                loss_p = p_loss(batch, outputs)
                loss = alpha * loss_r + beta * loss_t + gama * loss_p

                rtime_ctrller.optimizer.zero_grad()
                loss.backward()
                grad_norm_history.append(get_gradient_norm(model))
                rtime_ctrller.optimizer.step()
                if len(rtime_ctrller.optimizer.param_groups) > 0:
                    current_lr_history.append(rtime_ctrller.optimizer.param_groups[0]["lr"])
                if getattr(rtime_ctrller, "scheduler_batch", False) and rtime_ctrller.scheduler is not None:
                    rtime_ctrller.scheduler.step()

                total_loss += loss.item()
                total_r_loss += float(loss_r.detach())
                total_t_loss += float(loss_t.detach())
                total_p_loss += float(loss_p.detach())
                train_err["total_er"] += train_err["batch_er"]
                train_err["total_et"] += train_err["batch_et"]
                
                train_batch_num += 1
                tqdm_console.set_postfix_str(
                    ("loss:{:.3f}, er:{:.3f}, et:{:.3f}").format(
                        float(loss.detach()), train_err["batch_er"], train_err["batch_et"]
                    )
                )
                tqdm_console.update(1)

        train_batch_num = max(train_batch_num, 1)
        for key in ["total_er", "total_et", "gt_er", "gt_et", "total_extra"]:
            if key in train_err:
                train_err[key] /= train_batch_num
        avg_loss = total_loss / train_batch_num
        train_loss_history.append(avg_loss)

        fin_epoch_er = train_err["total_er"]
        fin_epoch_et = train_err["total_et"]
        fin_gt_er = train_err["gt_er"]
        fin_gt_et = train_err["gt_et"]

        Er_history.append(train_err["total_er"])
        Et_history.append(train_err["total_et"])
        if not getattr(rtime_ctrller, "scheduler_batch", False) and rtime_ctrller.scheduler is not None:
            rtime_ctrller.scheduler.step()
            if len(rtime_ctrller.optimizer.param_groups) > 0:
                current_lr_history.append(rtime_ctrller.optimizer.param_groups[0]["lr"])

        logger.info('Epoch {:03d}|{:03d}, total loss:{:.4f}|gt er:{:.4f},gt et:{:.4f}|fin er:{:.4f},fin et:{:.4f}'.format(epoch+1,total_epochs,
                                                                                                                        total_loss,
                                                                                                                        fin_gt_er,
                                                                                                                        fin_gt_et,
                                                                                                                        fin_epoch_er,
                                                                                                                        fin_epoch_et
                                                                                                                        ))

        calib_err = val_ZW(
            rtime_ctrller, model, zvdataloader, epoch=epoch, mode=mode, data_mode="val",
            projection_type=projection_type, visual_process=visual_process
        )
        calib_fin_er = calib_err["total_er"]
        calib_fin_et = calib_err["total_et"]
        
        calib_Er_history.append(calib_err["total_er"])
        calib_Et_history.append(calib_err["total_et"])
        logger.info(
            ("Val|"
             "gt_er:{:.4f},gt_et:{:.4f}|Fin_Val_Er:{:.4f},Fin_Val_Et:{:.4f}").format(
                calib_err["gt_er"], calib_err["gt_et"], calib_err["total_er"], calib_err["total_et"]
            )
        )
        
        _write_train_csv_record(rtime_ctrller, zvdataloader, epoch, {
            "loss": avg_loss,
            "train_er": train_err["total_er"],
            "train_et": train_err["total_et"],
            "val_er": calib_err["total_er"],
            "val_et": calib_err["total_et"]
        })
        output_figure = config_loader.get("output_figure", rtime_ctrller.dataset_name + rtime_ctrller.version)
        output_figure = str(output_figure) + "_" + str(rtime_ctrller.version)
        pic_name = rtime_ctrller.version + "  ,   model: " + str(rtime_ctrller.model.__class__.__name__)
        key_param = "loss_list: [" + ",".join([str(loss_item)[:5] for loss_item in rtime_ctrller.loss_List]) + "]"
        key_param += " , lossWeights: " + str(rtime_ctrller.loss_Weights)
        key_param += " , input_mode: " + str(getattr(model, "input_mode", "unknown"))
        key_param += " , dropout_rate: " + str(getattr(model, "dropout_rate2", "unknown"))
        key_param += " , r1=" + str(config_loader.get("r_Euler_range", ""))
        key_param += " , r2=" + str(config_loader.get("r_Euler_range2", ""))
        key_param += " , t1=" + str(config_loader.get("t_range", ""))
        key_param += " , t2=" + str(config_loader.get("t_range2", ""))
        key_param += " , Scene=" + str(scene_num) + " , BS=" + str(batch_size)
        plot_records = [
            rtime_ctrller.version,
            rtime_ctrller.dataset_name,
            scene_num,
            batch_size,
            total_batch_num,
            epoch + 1,
            avg_loss,
            train_err["total_et"],
            train_err["total_er"],
            calib_err["total_et"],
            calib_err["total_er"],
        ]
        
        outputPlot(
            current_lr_history, grad_norm_history, output_figure,
            train_loss_history, Et_history, Er_history,
            calib_Et_history, calib_Er_history,
            plot_records, pic_name=pic_name, key_param=key_param,
        )
        
        model_calib_score = rtime_ctrller.calModelEvaScore(fin_epoch_er,fin_epoch_et, calib_fin_er, calib_fin_et)
        model_train_score = rtime_ctrller.calModelEvaScore(fin_epoch_er,fin_epoch_et, fin_epoch_er,fin_epoch_et)
        
        if model_train_score < best_model_train_score and 0:
            best_model_train_score = model_train_score
            checkpoint_path2=stepWeights(checkpoint_path,epoch,tag = "_train")
            # save checkpoint
            rtime_ctrller.saveCheckPoint(model, rtime_ctrller.optimizer, rtime_ctrller.scheduler, epoch,
                                        avg_loss, model_train_score, checkpoint_path2, save_info_color=GREEN)
        
        if model_calib_score < best_model_calib_score:
            best_model_calib_score = model_calib_score
            checkpoint_path2=stepWeights(checkpoint_path,epoch,tag = "_calib")
            # save checkpoint
            rtime_ctrller.saveCheckPoint(model, rtime_ctrller.optimizer, rtime_ctrller.scheduler, epoch,
                                         avg_loss, model_calib_score, checkpoint_path2, save_info_color=MAGENTA)



@torch.no_grad()
def val_ZW(rtime_ctrller: ZV_Runtime_Controller, val_model: DST_Calib, val_loader: ZV_Dataset_Loader,
           epoch=0, mode="eva", data_mode="val", projection_type="Depth",
           visual_process=False):
    model = val_model
    original_training = model.training
   
    model.eval()
    
    if data_mode == "train":
        val_div = 8
        val_subset = val_loader.train_subset
        val_dataloader = val_loader.train_dataloader
        tqdm_console = tqdm(total=int(val_subset.getBatchNum()//val_div),desc='Val')
    else:
        val_div = 8
        val_subset = val_loader.val_subset
        val_dataloader = val_loader.val_dataloader
        tqdm_console = tqdm(total=int(val_subset.getBatchNum()//val_div),desc='Val')
        
    scene_num = val_loader.getValSceneNum()
    batch_size = val_loader.batch_size
    
    val_err = _init_err(include_gt=True)
    val_batch_num = 0
    
    with tqdm_console:
        tqdm_console.set_description_str('Val:')
        
        for i, batch in enumerate(val_dataloader):
            if i >= int(val_subset.getBatchNum()//val_div):
                break
            scene_idx = val_subset.getSceneIdxFromBatchIdx(i)
            batch = val_subset.batchDeal(batch)
            label_batch = batch["label_batch"]
            outputs = _forward_model(model, batch, mode)
            outputs = _update_outputs_for_loss(batch, outputs, val_subset, projection_type)
            _visualize_calibration_process(
                batch, outputs, val_subset, scene_idx, i, projection_type,
                visual_process=visual_process, title_after="After"
            )
            iter_est_xvec = outputs["xvec"]
            iter_er, iter_et, _, _ = pose.CalPoseErrBatch(iter_est_xvec, label_batch, print_err=False)
            gt_er, gt_et, _, _ = pose.CalPoseErrBatch(label_batch, torch.zeros((label_batch.shape[0], 6)).to(label_batch),print_err=False)

            val_err["total_er"] += iter_er
            val_err["total_et"] += iter_et
            val_err["gt_er"] += gt_er
            val_err["gt_et"] += gt_et
            
            
            tqdm_console.set_postfix_str(
            ("er:{:.3f}, et:{:.3f}" + ", val_batch_num:{:.3f}").format(iter_er,iter_et,val_batch_num))
            
            tqdm_console.update()
            
            val_batch_num += 1

    val_batch_num = max(val_batch_num, 1)
    for key in ["total_er", "total_et", "gt_er", "gt_et", "total_extra"]:
        if key in val_err:
            val_err[key] /= val_batch_num
    model.train(original_training)
    return val_err
