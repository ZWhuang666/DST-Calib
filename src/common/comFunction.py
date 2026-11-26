import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import os
import json
import torch
import numpy as np
import sys

def setRemapKAndSize(remap_proj_f,
                    remap_proj_size,
                    size_add_bias = (64, 600)
                    ):
    
    lidar_fov_proj_K = np.array([[800,0,768],[0,800,256],[0,0,1]],dtype=np.float32)
    lidar_fov_proj_size = (512,1536)
    lidar_fov_proj_f = lidar_fov_proj_K[0][0]

    remap_proj_K = np.eye(3)
    remap_proj_K[0][0] = remap_proj_f
    remap_proj_K[0][2] = remap_proj_size[1] / 2
    remap_proj_K[1][1] = remap_proj_f
    remap_proj_K[1][2] = remap_proj_size[0] / 2
    remap_proj_f = remap_proj_K[0][0]

    f_ratio_lidar2remap = lidar_fov_proj_f / remap_proj_f

    if f_ratio_lidar2remap * remap_proj_size[0] > lidar_fov_proj_size[0] or f_ratio_lidar2remap * remap_proj_size[1] > lidar_fov_proj_size[1]:
        if f_ratio_lidar2remap * remap_proj_size[0] > lidar_fov_proj_size[0]:
            lidar_fov_proj_h = f_ratio_lidar2remap* remap_proj_size[0]
        else:
            lidar_fov_proj_h = lidar_fov_proj_size[0]
        
        if f_ratio_lidar2remap * remap_proj_size[1] > lidar_fov_proj_size[1]:
            lidar_fov_proj_w = f_ratio_lidar2remap* remap_proj_size[1]
        else:
            lidar_fov_proj_w = lidar_fov_proj_size[1] 
        
        lidar_fov_proj_K[0][2] = lidar_fov_proj_w/2
        lidar_fov_proj_K[1][2] = lidar_fov_proj_h/2
        lidar_fov_proj_size = (int(lidar_fov_proj_h),int(lidar_fov_proj_w))
    
    
    dcloud_proj_f = remap_proj_f + (lidar_fov_proj_f -  remap_proj_f) / 2

    f_ratio_dcloud2remap = (dcloud_proj_f / remap_proj_f)
    dcloud_proj_size = [f_ratio_dcloud2remap*remap_proj_size[0],f_ratio_dcloud2remap*remap_proj_size[1]]

    if (dcloud_proj_size[0] + size_add_bias[0] <= lidar_fov_proj_size[0]):
        dcloud_proj_size[0] = (dcloud_proj_size[0] + size_add_bias[0])
    
    if (dcloud_proj_size[1] + size_add_bias[1] <= lidar_fov_proj_size[1]):
        dcloud_proj_size[1] = (dcloud_proj_size[1] + size_add_bias[1])

    dcloud_proj_K = np.eye(3)
    dcloud_proj_K[0][0] = dcloud_proj_f
    dcloud_proj_K[0][2] = dcloud_proj_size[1] / 2
    dcloud_proj_K[1][1] = dcloud_proj_f
    dcloud_proj_K[1][2] = dcloud_proj_size[0] / 2
    dcloud_proj_size = (int(dcloud_proj_size[0]),int(dcloud_proj_size[1]))
    
    print(f"LiDAR FOV PROJ PARAMETERS:\n lidar_fov_proj_K:{lidar_fov_proj_K}\nlidar_fov_proj_size:{lidar_fov_proj_size}")
    print(f"DCLOUD PROJ PARAMETERS:\n dcloud_proj_K:{dcloud_proj_K}\ndcloud_proj_size:{dcloud_proj_size}")
    print(f"REMAP PROJ PARAMETERS:\n remap_proj_K:{remap_proj_K}\nremap_proj_size:{remap_proj_size}")
    
    return lidar_fov_proj_K,lidar_fov_proj_size,dcloud_proj_K,dcloud_proj_size, remap_proj_K,remap_proj_size

    
def outputPlot(current_lr_history, grad_norm_history, output_figure, 
               train_loss_history, Et_history, Er_history, 
               calib_Et_history,calib_Er_history,test_records, 
               csvName="trainRes.csv", pic_name="",key_param=""):
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        
        fig1 = plt.figure(figsize=(12, 6))
        
        # learning rate
        plt.subplot(1, 2, 1)
        plt.plot(current_lr_history, color='blue')
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        
        # gradient norm
        plt.subplot(1, 2, 2)
        plt.plot(grad_norm_history, color='red')
        plt.yscale('log')
        plt.xlabel('Step')
        plt.ylabel('Gradient Norm (log scale)')
        plt.title('Gradient Stability')
        
        
        fig1.text(0.5, 0.01, f"{current_time} | {pic_name}", 
                fontsize=12, color='black', alpha=0.7,
                ha='center', va='bottom', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        fig1.text(0.5, 0.06, f"{key_param}", 
                fontsize=12, color='black', alpha=0.7,
                ha='center', va='bottom', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))             
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        lrgrad_str = f"output/learning_grad_{output_figure}.png"
        plt.savefig(lrgrad_str)
        plt.close(fig1)

        # loss and error metrics
        fig2 = plt.figure(figsize=(24, 6))
        
        # loss
        plt.subplot(1, 3, 1)
        plt.xlabel("epochs")
        plt.ylabel('train Loss')    
        plt.plot(train_loss_history, color="blue", label="train Loss")
        
        # et
        plt.subplot(1, 3, 2)
        plt.xlabel("epochs")
        plt.ylabel('Et')  
        plt.plot(Et_history, color="green", label="Train Et")
        plt.plot(calib_Et_history, color="orange", label="Calib Et")  
        plt.legend(loc="upper right")
        
        # er
        plt.subplot(1, 3, 3)
        plt.xlabel("epochs")
        plt.ylabel('Er')  
        plt.plot(Er_history, color="green", label="Train Er")
        plt.plot(calib_Er_history, color="orange", label="Calib Er")   
        plt.legend(loc="upper right")
        
        fig2.text(0.5, 0.01, f"{current_time} | {pic_name}|", 
                fontsize=12, color='black', alpha=0.7,
                ha='center', va='bottom', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        fig2.text(0.5, 0.06, f"{key_param}", 
                fontsize=12, color='black', alpha=0.7,
                ha='center', va='bottom', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        loss_Er_str = f"output/loss_Er_{output_figure}.png"
        plt.savefig(loss_Er_str)
        plt.close(fig2)

        # write to CSV
        first_row=["test_Ver","Dataset","scene_num","Batch_Size","Batchs","Epoch","Train_Loss","Train_Et","Train_Er","Val_Et","Val_Er","outputFigure"]
        if len(test_records) == len(first_row) + 3:
            first_row = first_row[:-1] + ["Train_E", "Val_E"] + first_row[-1:]
        needTitle=False
        test_records.append(output_figure)
        if not os.path.exists("output/"+csvName):
            needTitle=True
        with open("output/"+csvName, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if needTitle:
                writer.writerow(first_row)

            writer.writerow(test_records)

def stepWeights(checkpoint_path,epoch,tag = "eva"):
            checkpoint_path1=checkpoint_path[:-4]+tag
            stepList=[60,80,100,120,140,160,180,200,225,250,275]
            findEpochStage=False
            for n in range(len(stepList)):
                 if epoch<=stepList[n] and findEpochStage==False:
                      checkpoint_path1+="_e"+str(stepList[n])+".pth"
                      findEpochStage=True
            if findEpochStage==False:
                checkpoint_path1+="_e300.pth" 
            return checkpoint_path1


if __name__ == "__main__":
    pass