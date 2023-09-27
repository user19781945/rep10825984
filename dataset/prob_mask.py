import os
import numpy as np
import torch
import h5py
import scipy.io as sio

class ProbMask:
    def __init__(self, mask_arg):
        self.type=mask_arg["type"]
        assert os.path.exists(mask_arg["path"]),"static mask must specify a path"
        if mask_arg["path"].endswith("npy"):
            self.mask_prob=np.load(mask_arg["path"])
        else:
            try:
                self.mask_prob=np.array(sio.loadmat(mask_arg["path"])['mask'])
            except:
                mask = h5py.File(mask_arg["path"],"r")
                self.mask_prob=np.array(mask['mask_matrix'])
        self.fixed_mask={}
    def sample(self,item):
        prob=torch.from_numpy(self.mask_prob).float()
        if self.type=="dynamic":
            mask = (torch.rand(prob.shape)<prob).float()
        elif self.type=="fixed":
            if item not in self.fixed_mask:
                self.fixed_mask[item]=(torch.rand(prob.shape)<prob).float()
            mask = self.fixed_mask[item]
        else:
            raise Exception("Unkown mask type {}".format(self.type))
        return mask
    
class ProbMask_with_Resample:
    def __init__(self, mask_arg):
        self.probmask=ProbMask(mask_arg.prob_arg)
        self.n=mask_arg.n
    def sample(self,item):
        return self.probmask.sample(item)
    def resample(self,mask):
        p_re_lambda_mask=torch.zeros_like(mask)
        p_re_lambda_mask[mask==1]=1-(1-1/n)**n
        p_re_lambda_mask[self.probmask.mask_prob==1]=1
        return (torch.rand_like(p_re_lambda_mask)<p_re_lambda_mask).float()
    
class ProbMask_with_SubMask:
    def __init__(self, mask_arg):
        self.probmask=ProbMask(mask_arg.prob_arg)
        self.resubmask=ProbMask(mask_arg.resub_arg)
    def sample(self,item):
        return self.probmask.sample(item)
    def resample(self,mask):
        return mask*self.resubmask.sample(0)