import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, normalized_root_mse
from sklearn.metrics import mean_squared_error
import pandas as pd
import json
import os
import shutil


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
        
def generate_re_mask(lambda_mask,p_Lambda,args):
    n=args.n
    if n>1:
        assert not getattr(args,"prob_mask",None),"The specific prob mask {} in args is not working when n>1!".format(args.prob_mask)
        p_re_lambda_mask=torch.zeros_like(lambda_mask)
        p_re_lambda_mask[lambda_mask==1]=1-(1-1/n)**n
        p_re_lambda_mask[p_Lambda==1]=1
    else:
        if type(args.prob_mask) is str:
            args.prob_mask=torch.tensor(np.load(args.prob_mask)[None,...]).to(lambda_mask.device)
        p_re_lambda_mask=args.prob_mask
    if not getattr(args,"same_mask",False):
        return (torch.rand_like(lambda_mask)<p_re_lambda_mask).float()
    else:
        return (torch.rand_like(lambda_mask)[:1]<p_re_lambda_mask).float()