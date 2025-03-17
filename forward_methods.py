# System / Python
import os
import argparse
import logging
import random
import shutil
import time

import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
import cv2
# PyTorch

import torch.fft
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
# Custom
from mri_tools import *
from utils import *

def forward_dccnn_boot_test_only(mode, rank, model, dataloader, criterion, optimizer, args):
    assert mode in ["val","test"], "This method can only be used in test!"
    if mode=="val":
        recorder = BootRecorder(os.path.join("./results", args.exp_name, f"boot_samples_val"))
    else:
        recorder = BootRecorder(os.path.join("./results", args.exp_name, f"boot_samples_test"))
    t = tqdm(dataloader, desc=mode + 'ing',
             total=int(len(dataloader))) if rank == 0 else dataloader
    for iter_num, data_batch in enumerate(t):
        full_kspace = data_batch[0].to(rank, non_blocking=True)
        csm = data_batch[1].to(rank, non_blocking=True)
        mask_under = data_batch[2][0].to(rank, non_blocking=True)
        fnames = data_batch[3]
        slice_id = data_batch[4]
        mask_prob=data_batch[5][0].to(rank, non_blocking=True)
        
        label = torch.sum(ifft2(full_kspace) * torch.conj(csm), dim=1)

        under_img = At(full_kspace, csm, mask_under)
        under_img = torch.view_as_real(
            under_img).permute(0, 3, 1, 2).contiguous()
        under_kspace = full_kspace * mask_under[:, None, ...]
        with torch.no_grad():
            output_under=model(under_kspace, csm, mask_under)
        output_under_img=torch.sum(ifft2(output_under) * torch.conj(csm), dim=1)
        for _ in range(args.inner_epochs):
        
            mask_net_up=generate_re_mask(mask_under,under_kspace)
            input_kspace = full_kspace * mask_net_up[:, None, ...]

            output = model(input_kspace, csm, mask_net_up)

            output_img=torch.sum(ifft2(output) * torch.conj(csm), dim=1)
            
            boot_list=[output_img]
            for _ in range(args.m-1):
                mask_net_up=generate_re_mask(mask_under,under_kspace)
                input_kspace = full_kspace * mask_net_up[:, None, ...]
                output = model(input_kspace, csm, mask_net_up)
                output_img=torch.sum(ifft2(output) * torch.conj(csm), dim=1)
                boot_list.append(output_img)
                
            mean_boot_img=torch.mean(torch.stack([torch.abs(output) for output in boot_list]),dim=0)
            boot_mse=torch.mean(torch.stack([(torch.abs(output)-torch.abs(output_under_img))**2 for output in boot_list]),dim=0)
            boot_var=torch.mean(torch.stack([(torch.abs(output)-torch.abs(mean_boot_img))**2 for output in boot_list]),dim=0)
            gt_mse=(torch.abs(output_under_img)-torch.abs(label))**2

            output_path = recorder.path
            os.makedirs(output_path, exist_ok=True)
            output_fnames = []
            for i, fname in enumerate(fnames):
                assert "rec" not in fname
                bname = os.path.basename(fname)
                if bname.endswith(".h5"):
                    bname = bname[:-3]+f"_{data_batch[4][i]}.png"
                output_fnames.append(os.path.join(
                    output_path, "rec_"+bname.replace("png", "npy")))
            recorder.submit([torch.abs(output_under_img),label,gt_mse,mean_boot_img,boot_mse,boot_var], output_fnames)
            break
    recorder.export()
    return
    
def forward_dccnn_boot_per(mode, rank, model, dataloader, criterion, optimizer, log, args):
    assert mode in ['train', 'val', 'test']
    if mode=="val" or mode=="test":
        m=args.m
        if hasattr(args,"m_t"):
            args.m=args.m_t
        with torch.no_grad():
            forward_dccnn_boot_test_only(mode, rank, model, dataloader, criterion, optimizer, args)
        args.m=m
            
    loss, psnr, ssim = 0.0, 0.0, 0.0
    recorder = Recorder(os.path.join("./results", args.exp_name))
    t = tqdm(dataloader, desc=mode + 'ing',
             total=int(len(dataloader))) if rank == 0 else dataloader
    for iter_num, data_batch in enumerate(t):
        full_kspace = data_batch[0].to(rank, non_blocking=True)
        csm = data_batch[1].to(rank, non_blocking=True)
        mask_under = data_batch[2][0].to(rank, non_blocking=True)
        fnames = data_batch[3]
        slice_id = data_batch[4]
        mask_prob=data_batch[5][0].to(rank, non_blocking=True)
        
        label = torch.sum(ifft2(full_kspace) * torch.conj(csm), dim=1)

        under_img = At(full_kspace, csm, mask_under)
        under_img = torch.view_as_real(
            under_img).permute(0, 3, 1, 2).contiguous()
        under_kspace = full_kspace * mask_under[:, None, ...]
        with torch.no_grad():
            output_under=model(under_kspace, csm, mask_under)
        output_under_img=torch.sum(ifft2(output_under) * torch.conj(csm), dim=1)
        
        for _ in range(args.inner_epochs):
            if mode == 'train':
                optimizer.zero_grad()
        
            mask_net_up=generate_re_mask(mask_under,under_kspace)

            if mode == 'test':
                net_img_up = net_img_down = under_img
                mask_net_up = mask_net_down = mask_under

            input_kspace = full_kspace * mask_net_up[:, None, ...]

            if mode == 'train':
                output = model(input_kspace, csm, mask_net_up)
                output=output*mask_under[:, None, ...]+output_under.detach()*(1-mask_under[:, None, ...])
            else:
                with torch.no_grad():
                    output = model(input_kspace, csm, mask_net_up)
            loss_boot = criterion(torch.view_as_real(output),torch.view_as_real(output_under))/args.m
            output_img=torch.sum(ifft2(output) * torch.conj(csm), dim=1)
            if mode=='train':
                loss_boot.backward()
            batch_loss=loss_boot.item()
            
            for _ in range(args.m-1):
                mask_net_up=generate_re_mask(mask_under,under_kspace)
                input_kspace = full_kspace * mask_net_up[:, None, ...]
                output = model(input_kspace, csm, mask_net_up)
                output=output*mask_under[:, None, ...]+output_under.detach()*(1-mask_under[:, None, ...])
                loss_boot = criterion(torch.view_as_real(output),torch.view_as_real(output_under))/args.m
                output_img=torch.sum(ifft2(output) * torch.conj(csm), dim=1)
                if mode=='train':
                    loss_boot.backward()
                batch_loss+=loss_boot.item()

            loss += batch_loss
            print(batch_loss,end="\r")

            if mode == 'train':
                optimizer.step()
            if mode!='train':
                f_output = output_under_img
                output_path = os.path.join("./results", args.exp_name, "samples")
                os.makedirs(output_path, exist_ok=True)
                output_fnames = []
                for i, fname in enumerate(fnames):
                    bname = os.path.basename(fname)
                    if bname.endswith(".h5"):
                        bname = bname[:-3]+f"_{data_batch[4][i]}.png"
                    output_fnames.append(os.path.join(
                        output_path, "rec_"+bname.replace("png", "npy")))
                psnr += psnr_slice(label, f_output)
                ssim += ssim_slice(label, f_output)
                recorder.submit(label, f_output, output_fnames)
                break
    if mode == 'train':
        loss /= (len(dataloader)*args.inner_epochs)
        log.append(loss)
        curr_lr = optimizer.param_groups[0]['lr']
        log.append(curr_lr)
    else:
        log.append(loss)
        psnr /= len(dataloader)
        ssim /= len(dataloader)
        log.append(psnr)
        log.append(ssim)
        recorder.export()
    return log
