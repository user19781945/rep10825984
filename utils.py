import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, normalized_root_mse
from sklearn.metrics import mean_squared_error
import pandas as pd
import json
import os
import shutil

class MixL1L2Loss(nn.Module):
    def __init__(self, eps=1e-6,scalar=1/2):
        super().__init__()
        #self.mse = nn.MSELoss()
        self.eps = eps
        self.scalar=scalar
    def forward(self, yhat, y):

        loss = self.scalar*(torch.norm(yhat-y) / torch.norm(y)) + self.scalar*(torch.norm(yhat-y,p=1) / torch.norm(y, p=1))
        
        return loss

def complex2real(data, axis=-1):
    assert type(data) is np.ndarray
    data = np.stack((data.real, data.imag), axis=axis)
    return data


def real2complex(data, axis=-1):
    assert type(data) is np.ndarray
    assert data.shape[axis] == 2
    mid = data.shape[axis] // 2
    data = data[..., 0:mid] + data[..., mid:] * 1j
    return data.squeeze(axis=axis)


def get_cos_similar(v1, v2):
    v1, v2 = torch.abs(v1), torch.abs(v2)
    batch_size = v1.shape[0]
    similar = torch.tensor(0.)
    for i in range(batch_size):
        num = torch.dot(v1[i], v2[i])
        denom = torch.linalg.norm(v1[i] * torch.linalg.norm(v2[i]))
        res = 0.5 + 0.5 * (num/denom) if denom != 0 else 0
        similar = torch.add(similar, res)
    return similar/batch_size


def get_cos_similar_matrix(v1, v2):
    assert type(v1) == type(v2)
    if type(v1) is torch.Tensor:
        v1, v2 = v1.detach().cpu().numpy(), v2.detach().cpu().numpy()
    v1, v2 = np.abs(v1), np.abs(v2)
    batch_size = v1.shape[0]
    similar = 0.0
    for i in range(batch_size):
        num = np.dot(v1[i], np.array(v2[i]).T)
        denom = np.linalg.norm(v1[i], axis=1).reshape(-1, 1)*np.linalg.norm(v2[i], axis=1)
        res = num / denom
        res[np.isneginf(res)] = 0
        res = np.mean(0.5 + 0.5 * res)
        similar += res
    return similar/batch_size


def mse_slice(gt, pred):
    assert type(gt) == type(pred)
    if type(pred) is torch.Tensor:
        gt, pred = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()
    gt, pred = np.abs(gt), np.abs(pred)
    batch_size = gt.shape[0]
    MSE = 0.0
    for i in range(batch_size):
        mse = mean_squared_error(gt[i], pred[i])
        MSE += mse
    return MSE/batch_size


def psnr_slice(gt, pred, maxval=None):
    assert type(gt) == type(pred)
    if type(pred) is torch.Tensor:
        gt, pred = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()
    gt, pred = np.abs(gt), np.abs(pred)
    batch_size = gt.shape[0]
    PSNR = 0.0
    for i in range(batch_size):
        max_val = gt[i].max() if maxval is None else maxval
        psnr = peak_signal_noise_ratio(gt[i], pred[i], data_range=max_val)
        PSNR += psnr

    return PSNR / batch_size


def ssim_slice(gt, pred, maxval=None):
    assert type(gt) == type(pred)
    if type(pred) is torch.Tensor:
        gt, pred = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()
    gt, pred = np.abs(gt), np.abs(pred)
    batch_size = gt.shape[0]
    SSIM = 0.0
    for i in range(batch_size):
        max_val = gt[i].max() if maxval is None else maxval
        ssim = structural_similarity(gt[i], pred[i], data_range=max_val)
        SSIM += ssim
    return SSIM / batch_size

class Recorder:
    def __init__(self,path):
        self.path=path
        self.psnr=[]
        self.ssim=[]
        self.mse=[]
        self.nmse=[]
        self.fnames=[]
        self.total_time=0

    def submit(self,gt, pred, fnames):
        assert type(gt) == type(pred)
        if type(pred) is torch.Tensor:
            gt, pred = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()
        gt, pred = np.abs(gt), np.abs(pred)
        batch_size = gt.shape[0]
        for i in range(batch_size):
            max_val = gt[i].max()
            self.ssim.append(structural_similarity(gt[i], pred[i], data_range=max_val))
            self.psnr.append(peak_signal_noise_ratio(gt[i], pred[i], data_range=max_val))
            self.mse.append(mean_squared_error(gt[i], pred[i]))
            self.nmse.append(normalized_root_mse(gt[i], pred[i]))
            self.fnames.append(fnames[i])
            np.save(fnames[i],pred[i])
            np.save(fnames[i].replace("rec","gt"),gt[i])
    
    def export(self):
        print("Total Time:{}".format(self.total_time))
        with open(os.path.join(self.path,"time.txt"),"a+") as fo:
            fo.writelines([str(self.total_time)+"\n"])
        exp_dict={"fnames":self.fnames,"psnr":self.psnr,"ssim":self.ssim,"mse":self.mse,"nmse":self.nmse}
        frame=pd.DataFrame(exp_dict)
        frame.to_csv(os.path.join(self.path,"_record.csv"))

class BootRecorder:
    def __init__(self,path):
        self.path=path
        self.gt_mses=[]
        self.boot_mses=[]
        self.boot_vars=[]
        self.fnames=[]
        
    def convert(self,data):
        if type(data) is torch.Tensor:
            data = data.detach().cpu().numpy()
        return np.abs(data)

    def submit(self, data_tuples, fnames):
        batch_size=len(data_tuples[0])
        assert batch_size==len(fnames)
        data_tuples=[self.convert(data) for data in data_tuples]
        for i in range(batch_size):
            output_under_img_abs,label,gt_mse,mean_boot_img,boot_mse,boot_var=[data_tuple[i] for data_tuple in data_tuples]
            self.gt_mses.append(np.mean(gt_mse))
            self.boot_mses.append(np.mean(boot_mse))
            self.boot_vars.append(np.mean(boot_var))
            self.fnames.append(fnames[i])
            np.save(fnames[i],output_under_img_abs)
            np.save(fnames[i].replace("rec","gt"),label)
            np.save(fnames[i].replace("rec","mean_boot"),mean_boot_img)
            np.save(fnames[i].replace("rec","gt_mse"),gt_mse)
            np.save(fnames[i].replace("rec","boot_mse"),boot_mse)
            np.save(fnames[i].replace("rec","boot_var"),boot_var)
    
    def export(self):
        exp_dict={"fnames":self.fnames,"gt_mses":self.gt_mses,"boot_mses":self.boot_mses,"boot_vars":self.boot_vars}
        frame=pd.DataFrame(exp_dict)
        frame.to_csv(os.path.join(self.path,"boot_record.csv"))

def generate_re_mask(mask, under_kspace):
    under_kspace=under_kspace.squeeze()
    if under_kspace.ndim==2:
        under_kspace=under_kspace.unsqueeze(0)
    assert under_kspace.ndim==3 ,"Must be (N-channel,N-row,N-col) k-space! Currently shape is {}".format(under_kspace.shape)
    center_x=under_kspace.norm(p='fro',dim=[0,1]).argmax().item()
    center_y=under_kspace.norm(p='fro',dim=[0,2]).argmax().item()
    remask_prob=torch.ones_like(mask)*(1-(1-1/1000)**1000)
    remask_prob[...,center_x-2:center_x+2,center_y-2:center_y+2]=1
    remask_prob=mask*remask_prob
    remask=(torch.rand_like(remask_prob)<remask_prob).float()
    return remask.to(mask.device)