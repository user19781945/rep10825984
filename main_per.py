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
from forward_methods import *
from net import *
from dataset import *
from mri_tools import *
from utils import psnr_slice, ssim_slice, get_cos_similar_matrix, get_cos_similar

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def load_cfg(cfg_path):
    with open(cfg_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    namespace=dict2namespace(config)
    return namespace
parser = argparse.ArgumentParser()
parser.add_argument('--cfg-path', type=str, help='path to the config')
parser.add_argument('--test', action="store_true")
parser.add_argument('--pretrain', action="store_true")
parser.add_argument('--dist', action="store_true")

def create_logger(args):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s:\t%(message)s')
    stream_formatter = logging.Formatter('%(levelname)s:\t%(message)s')

    file_handler = logging.FileHandler(filename=os.path.join("./results",args.exp_name,'logger.txt'), mode='a+', encoding='utf-8')
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(file_formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.INFO)
    stream_handler.setFormatter(stream_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

def read_batches(fnames):
    imgs=[cv2.imread(fname,cv2.IMREAD_GRAYSCALE) for fname in fnames]
    return np.stack(imgs,0)/255

def init_weights(net, init_type='xavier', gain=1.0):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('Initialization method {} is not implemented.'.format(init_type))
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)
    
def solvers_per(rank, ngpus_per_node, args):
    forward=globals()[args.forward_method]
    Dataset=globals()[args.dataset_type]
    
    if rank == 0:
        logger = create_logger(args)
        logger.info('Running distributed data parallel on {} gpus.'.format(args.world_size))
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl', init_method=args.init_method, world_size=args.world_size, rank=rank)
    
    
    train_set_all = Dataset(**vars(args.dataset_params.train))
    origin_exp_name=args.exp_name
    for indice in range(len(train_set_all.files)):
        train_set=torch.utils.data.Subset(train_set_all, train_set_all.indices[indice])
        data_name=train_set_all.files[indice].name.split(".")[0]
        args.exp_name=origin_exp_name+"/"+data_name
        logger.info("Running Experiment on data {}".format(data_name))
        args.loss_curve_path=os.path.join("./results",args.exp_name,"loss_curve")
        args.model_save_path=os.path.join("./results",args.exp_name,"checkpoints")
        os.makedirs(os.path.join("./results",args.exp_name),exist_ok=True)
        
        train_sampler = DistributedSampler(train_set)
        train_loader = DataLoader(
            dataset=train_set, batch_size=len(train_set), shuffle=False,
            pin_memory=True, sampler=train_sampler
        )
    
        # set initial value
        start_epoch = 0
        # model
        model = globals()[args.model_type](**vars(args.model_params))
        model.rank=rank
        # whether load checkpoint
        assert not args.pretrained,"Per training mode do not support continue training from checkpoint!"
        if args.pretrained or args.mode == 'test':
            model_path = os.path.join(args.model_save_path, 'best_checkpoint_dc_3x(1).pth.tar')
            assert os.path.isfile(model_path)
            checkpoint = torch.load(model_path, map_location='cuda:{}'.format(rank))
            start_epoch = checkpoint['epoch']
            lr = checkpoint['lr']
            args.lr = lr
            best_ssim = checkpoint['best_ssim']
            model.load_state_dict(checkpoint['model'])
            if rank == 0:
                logger.info('Load checkpoint at epoch {}.'.format(start_epoch))
                logger.info('Current learning rate is {}.'.format(lr))
                logger.info('Current ssim in train phase is {}.'.format(best_ssim))
                logger.info('The model is loaded.')
        elif args.use_init_weights:
            init_weights(model, init_type=args.init_type, gain=args.gain)
            if rank == 0:
                logger.info('Initialize model with {}.'.format(args.init_type))
        model = model.to(rank)
        model = DDP(model, device_ids=[rank])

        # criterion, optimizer, learning rate scheduler
        criterion = nn.MSELoss()
        optimizer = optim.Adam([{'params': model.parameters(), 'lr': args.lr}])
        if not args.pretrained:
            warm_up = lambda epoch: epoch / args.warmup_epochs if epoch <= args.warmup_epochs else 1
            scheduler_wu = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_up)

        if hasattr(args,"distillee_params"):
            buil_distillee_model(args.distillee_params)

        # test step
        if args.mode == 'test':
            if rank == 0:
                logger.info('The size of test dataset is {}.'.format(len(test_set)))
                logger.info('Now testing {}.'.format(args.exp_name))
            model.eval()
            with torch.no_grad():
                test_log = []
                start_time = time.time()
                test_log = forward('test', rank, model, train_loader, criterion, optimizer, test_log, args)
                test_time = time.time() - start_time
            # test information
            test_loss = test_log[0]
            test_psnr = test_log[1]
            test_ssim = test_log[2]
            if rank == 0:
                logger.info('time:{:.5f}s\ttest_loss:{:.7f}\ttest_psnr:{:.5f}\ttest_ssim:{:.5f}'.format(test_time, test_loss, test_psnr, test_ssim))
            continue

        # training step
        if rank == 0:
            logger.info('The size of training dataset and validation dataset is {}.'.format(len(train_set)))
            logger.info('Now training {}.'.format(args.exp_name))
            writer = SummaryWriter(args.loss_curve_path)
        # loss curve
        epochs = []
        trains = []
        vals = []
        psnr = []
        ssim = []
        for epoch in range(start_epoch + 1, args.num_epochs + 1):
            train_sampler.set_epoch(epoch)
            train_log = [epoch]
            epoch_start_time = time.time()
            model.train()

            train_log = forward('train', rank, model, train_loader, criterion, optimizer, train_log, args)
            model.eval()
            with torch.no_grad():
                train_log = forward('val', rank, model, train_loader, criterion, optimizer, train_log, args)
            epoch_time = time.time() - epoch_start_time
            # train information
            epoch = train_log[0]
            train_loss = train_log[1]
            lr = train_log[2]
            val_loss = train_log[3]
            val_psnr = train_log[4]
            val_ssim = train_log[5]

            # add loss
            epochs.append(epoch)
            trains.append(train_loss)
            vals.append(val_loss)
            psnr.append(val_psnr)
            ssim.append(val_ssim)

            if rank == 0:
                logger.info('epoch:{:<8d}time:{:.5f}s\tlr:{:.8f}\ttrain_loss:{:.7f}\tval_loss:{:.7f}\tval_psnr:{:.5f}\t'
                            'val_ssim:{:.5f}'.format(epoch, epoch_time, lr, train_loss, val_loss, val_psnr, val_ssim))
                writer.add_scalars('loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)
                # save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'lr': lr,
                    'best_ssim': val_ssim,
                    'model': model.module.state_dict()
                }
                if not os.path.exists(args.model_save_path):
                    os.makedirs(args.model_save_path)
                model_path = os.path.join(args.model_save_path, 'checkpoint_dc_3x(1).pth.tar')
                best_model_path = os.path.join(args.model_save_path, 'best_checkpoint_dc_3x(1).pth.tar')
                torch.save(checkpoint, model_path)
                shutil.copy(model_path, best_model_path)
            # scheduler
            if epoch <= args.warmup_epochs and not args.pretrained:
                scheduler_wu.step()
        if rank == 0:
            writer.close()
        np.savetxt(os.path.join("./results",args.exp_name,"dc_train_loss_3x(1).txt"), trains, fmt='%.5f', delimiter=" ")
        np.savetxt(os.path.join("./results",args.exp_name,"dc_val_loss_3x(1).txt"), vals, fmt='%.5f', delimiter=" ")
        np.savetxt(os.path.join("./results",args.exp_name,"dc_val_psnr_3x(1).txt"), psnr, fmt='%.5f', delimiter=" ")
        np.savetxt(os.path.join("./results",args.exp_name,"dc_val_ssim_3x(1).txt"), ssim, fmt='%.5f', delimiter=" ")

        # plot curve
        plt.ion()
        x = range(0, len(epochs))
        plt.subplot(1, 3, 1)
        plt.plot(x, trains)
        plt.plot(x, vals)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.subplot(1, 3, 2)
        plt.plot(x, psnr)
        plt.xlabel('epoch')
        plt.ylabel('psnr')
        plt.subplot(1, 3, 3)
        plt.plot(x, ssim)
        plt.xlabel('epoch')
        plt.ylabel('ssim')
        plt.show()
        name = os.path.join("./results",args.exp_name,str(time.time()) + '_dc_3x(1).jpg')
        plt.savefig(name)


def main():
    args0 = parser.parse_args()
    args=load_cfg(args0.cfg_path)
    
    args.world_size = args.nodes * args.gpus  # 1*2
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    args.loss_curve_path=os.path.join("./results",args.exp_name,"loss_curve")
    args.model_save_path=os.path.join("./results",args.exp_name,"checkpoints")
    os.makedirs(os.path.join("./results",args.exp_name),exist_ok=True)
    if args0.test:
        args.mode="test"
    if args0.pretrain:
        args.pretrained=True
    if hasattr(args,"solvers"):
        print(f"Using {args.solvers} solver!")
        solvers=globals()[args.solvers]
    else:
        print("Using supervised solver!")
        solvers=solvers_sup
    
    print('begin')
    solvers(0,1,args)
    print('end')


if __name__ == '__main__':
    main()
