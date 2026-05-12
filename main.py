### Train
# python main.py   --config config/RPI/mymodel.yaml
# torchrun --nproc_per_node=4 main.py   --config config/RPI/mymodel.yaml   --distributed True --device 0 1 2 3
# python multi_train.py
### Test
# python main.py --config config/RPI/mymodel.yaml --phase test --save-score True --weights work_dir/RPI-degcn_joint/epoch_31_155.pt --device 0


from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from time import strftime, localtime
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob
import warnings
import re

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

import thop
from copy import deepcopy

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
from torch.cuda.amp import autocast, GradScaler

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = True

def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0

def get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1

def is_main_process():
    return get_rank() == 0

def get_current_timestamp():
    ct = time.time()
    ms = int((ct - int(ct)) * 1000)
    return '[ {},{:0>3d} ] '.format(strftime('%Y-%m-%d %H:%M:%S', localtime(ct)), ms)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def get_parser():
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')

    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')
    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd120-cross-subject/default.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=True,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=3, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--save-epoch',
        type=int,
        default=0,
        help='the start epoch to save model (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=0,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument(
        '--distributed',
        type=str2bool,
        default=False,
        help='use DistributedDataParallel')
    parser.add_argument(
        '--local_rank',
        type=int,
        default=0,
        help='local rank for distributed training')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--warm_up_epoch', 
        type=int, 
        default=0)
    parser.add_argument(
        '--cosine_epoch', 
        type=int, 
        default=0)
    parser.add_argument(
        '--half', 
        type=str2bool, 
        default=True)
    return parser


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)

    
class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.is_distributed = getattr(arg, 'distributed', False) and dist.is_available() and dist.is_initialized()
        self.is_main_process = is_main_process()
        self.save_arg()
        self.ctime = ''.join(re.split('-|:|\[|\]', get_current_timestamp())).split(',')[0]
        self.savepath = self.arg.work_dir + '/' + self.ctime
        if self.is_main_process and not os.path.exists(self.savepath):
            os.makedirs(self.savepath)
        if self.is_distributed:
            dist.barrier()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if self.is_main_process:
                    if os.path.isdir(arg.model_saved_name):
                        print('log_dir: ', arg.model_saved_name, 'already exist')
                        # answer = input('delete it? y/n:')
                        answer = 'y'
                        if answer == 'y':
                            shutil.rmtree(arg.model_saved_name)
                            print('Dir removed: ', arg.model_saved_name)
                        else:
                            print('Dir not removed: ', arg.model_saved_name)
                    self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                    self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
                else:
                    self.train_writer = None
                    self.val_writer = None
            else:
                if self.is_main_process:
                    self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
                else:
                    self.train_writer = None
                    self.val_writer = None
        self.global_step = 0
        self.load_model()
        
        self.model = self.model.cuda(self.output_device)

        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.output_device],
                output_device=self.output_device)
        elif type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)

        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
            self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

        if self.arg.half:
            self.print_log('Use PyTorch AMP (fp16) Training!')
            self.scaler = GradScaler()
        else:
            self.scaler = None
        

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        self.train_sampler = None
        if self.arg.phase == 'train':
            train_dataset = Feeder(**self.arg.train_feeder_args)
            if self.is_distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset,
                    shuffle=True)
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=self.arg.batch_size,
                shuffle=(self.train_sampler is None),
                sampler=self.train_sampler,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.local_rank if self.is_distributed else (self.arg.device[0] if type(self.arg.device) is list else self.arg.device)
        self.output_device = output_device
        Model = import_class(self.arg.model)
        if self.is_main_process:
            shutil.copy2(inspect.getfile(Model), self.savepath)
        print(Model)
        self.model = Model(**self.arg.model_args)
        self.loss = LabelSmoothingCrossEntropy().cuda(output_device)

        if self.arg.weights:
            self.global_step = int(arg.weights[:-3].split('_')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                if self.is_distributed:
                    self.model.module.load_state_dict(weights)
                else:
                    self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer == 'RMSProp':
            self.optimizer = optim.RMSprop(
                self.model.parameters(), 
                lr=self.arg.base_lr, 
                alpha=0.9, 
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if self.is_main_process:
            if not os.path.exists(self.arg.work_dir):
                os.makedirs(self.arg.work_dir)
            with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
                f.write('# command line: {}\n\n'.format(' '.join(sys.argv)))
                yaml.dump(arg_dict, f)

    #     def adjust_learning_rate(self, epoch): #step
#         if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
#             if epoch < self.arg.warm_up_epoch:
#                 lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
#             else:
#                 lr = self.arg.base_lr * (0.1 ** np.sum(epoch >= np.array(self.arg.step)))
#             for param_group in self.optimizer.param_groups:
#                 param_group['lr'] = lr
#             return lr
#         else:
#             raise ValueError()

#     def adjust_learning_rate(self, epoch): #cosine
#         if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
#             if epoch < self.arg.warm_up_epoch:
#                 lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
#             else:
#                 lr = self.arg.base_lr * (0.5 * (np.cos((epoch-self.arg.warm_up_epoch) / (self.arg.num_epoch-self.arg.warm_up_epoch) * np.pi) + 1))
#             for param_group in self.optimizer.param_groups:
#                 param_group['lr'] = lr
#             return lr
#         else:
#             raise ValueError()

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            num_epoch_ = self.arg.cosine_epoch + self.arg.warm_up_epoch
            lr_cos = self.arg.base_lr * (0.5 * (np.cos((epoch-self.arg.warm_up_epoch) / (num_epoch_-self.arg.warm_up_epoch) * np.pi) + 1))
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch  
            elif epoch < num_epoch_ and lr_cos > 0.01: 
                lr = lr_cos
            else:
                lr = self.arg.base_lr * (0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if not self.is_main_process:
            return
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.savepath), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=True):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        loss_value = []
        acc_value = []
        if self.train_writer is not None:
            self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, dynamic_ncols=True, disable=not self.is_main_process)
        

        for batch_idx, (data, data_rgb, label, index) in enumerate(process):
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                data_rgb = data_rgb.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
            timer['dataloader'] += self.split_time()

            # forward
            with autocast(enabled=self.arg.half):
                output = self.model(data, data_rgb)
                loss = sum([self.loss(out, label) for out in output])
                output = sum(output)

            # backward
            self.optimizer.zero_grad()
            if self.arg.half:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())
            if self.train_writer is not None:
                self.train_writer.add_scalar('acc', acc, self.global_step)
                self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            if self.train_writer is not None:
                self.train_writer.add_scalar('lr', self.lr, self.global_step)
            
            timer['statistics'] += self.split_time()
            process.set_description('Loss: {:.4f}, LR: {:.4f}'.format(loss.data.item(), self.lr))

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(np.mean(loss_value), np.mean(acc_value)*100))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model:
            if self.is_main_process:
                state_dict = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
                weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])

                torch.save(weights, self.savepath + '/epoch_' + str(epoch+1) + '_' + str(int(self.global_step)) + '.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        # Only main process performs evaluation
        if not self.is_main_process:
            return
            
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        
        # Use the underlying model for evaluation (unwrap DDP if needed)
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            step = 0
            process = tqdm(self.data_loader[ln], ncols=40, disable=not self.is_main_process)
            for batch_idx, (data, data_rgb, label, index) in enumerate(process):
                label_list.append(label)
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    data_rgb = data_rgb.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)

                    # inside Processor.eval(), before forward
                    # if (not hasattr(self, "_vis_done")):
                    #     # self._vis_done = True
                    #     # 开启某个具体 DeSGC 的缓存：建议只开一个层，否则太多
                    #     # 例如：streams.0 的 block-1
                    #     gcn = model.streams[0]._modules['block-1_tcngcn'].gcn
                    #     if hasattr(gcn, "enable_vis"):
                    #         gcn.enable_vis = True
                    #         gcn.vis_max_batch = 8

                    with autocast(enabled=self.arg.half):
                        output = model(data, data_rgb)
                        loss = sum([self.loss(out, label) for out in output])
                        output = sum(output)
                    
                    # eval 完以后（同一进程内）
                    # mhr_names 就是你贴的 70 个名字列表
                    from feeders.bone_pairs import mhr70_pairs, mhr70_pairs_upper
                    # gcn.plot_fig8_style(
                    #     joint_i=41,
                    #     skeleton_edges= mhr70_pairs_upper if self.arg.train_feeder_args['upper'] else mhr70_pairs,
                    #     raw_data=data,
                    #     action_name="none",
                    #     sample_ids=(0,1,2,3),
                    #     scale_idx=0,
                    #     head_idx=0,
                    #     save_path="fig8_handwaving.png"
                    # )

                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())

                    _, predict_label = torch.max(output.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())
                    step += 1

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            if 'ucla' in self.arg.feeder:
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1
                

            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.savepath, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

            # acc for each class:
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            confusion = confusion_matrix(label_list, pred_list)
            list_diag = np.diag(confusion)
            list_raw_sum = np.sum(confusion, axis=1)
            each_acc = list_diag / list_raw_sum
            with open('{}/epoch{}_{}_each_class_acc.csv'.format(self.savepath, epoch + 1, ln), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(each_acc)
                writer.writerows(confusion)

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Modelargs:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            self.data_shape = [3, self.arg.train_feeder_args['window_size'], self.arg.model_args['num_point'], self.arg.model_args['num_person']]
            dummy_skel = torch.rand([1] + self.data_shape)
            dummy_rgb  = torch.rand(1, 3, 160, 640)
            flops, params = thop.profile(import_class(self.arg.model)(**self.arg.model_args), inputs=(dummy_skel, dummy_rgb), verbose=False)
            self.print_log('Model profile: {:.2f}G FLOPs and {:.2f}M Parameters'.format(flops / 1e9, params / 1e6))
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.print_log('*'*100)
                save_model = self.is_main_process and (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and (epoch+1) > self.arg.save_epoch

                self.train(epoch, save_model=save_model)
                
                if self.is_main_process:
                    if epoch > self.arg.num_epoch-20:
                        self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])
                    elif epoch%5==0:
                        self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])
                    self.print_log('Best_Accuracy: {:.2f}%, epoch: {}'.format(self.best_acc*100, self.best_acc_epoch))

            # test the best model
            if self.is_main_process:
                weights_path = glob.glob(self.savepath + '/epoch_' + str(self.best_acc_epoch) + '*')[0]
                
                weights = torch.load(weights_path)
                if (not self.is_distributed) and type(self.arg.device) is list:
                    if len(self.arg.device) > 1:
                        weights = OrderedDict([['module.'+k, v.cuda(self.output_device)] for k, v in weights.items()])
                if self.is_distributed:
                    self.model.module.load_state_dict(weights)
                else:
                    self.model.load_state_dict(weights)

                wf = weights_path.replace('.pt', '_wrong.txt')
                rf = weights_path.replace('.pt', '_right.txt')
                self.arg.print_log = False
                self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
                self.arg.print_log = True


                num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                self.print_log('Best accuracy: {}'.format(self.best_acc))
                self.print_log('Epoch number: {}'.format(self.best_acc_epoch))
                self.print_log('Model name: {}'.format(self.arg.work_dir))
                self.print_log('Model total number of params: {}'.format(num_params))
                self.print_log('Weight decay: {}'.format(self.arg.weight_decay))
                self.print_log('Base LR: {}'.format(self.arg.base_lr))
                self.print_log('Batch Size: {}'.format(self.arg.batch_size))
                self.print_log('Test Batch Size: {}'.format(self.arg.test_batch_size))
                self.print_log('seed: {}'.format(self.arg.seed))

        elif self.arg.phase == 'test':
            if not self.is_main_process:
                return
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = get_parser()
    os.chdir(os.getcwd())

    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
#             default_arg = yaml.load(f)
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        arg.distributed = True
        arg.world_size = int(os.environ.get('WORLD_SIZE', 1))
        arg.rank = int(os.environ.get('RANK', 0))
        arg.local_rank = int(os.environ.get('LOCAL_RANK', arg.local_rank))
    else:
        arg.world_size = 1
        arg.rank = 0

    if arg.distributed:
        torch.cuda.set_device(arg.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')

    init_seed(arg.seed + arg.rank)
    processor = Processor(arg) 
    processor.start()
    
    # Cleanup distributed process group
    if arg.distributed:
        dist.destroy_process_group()