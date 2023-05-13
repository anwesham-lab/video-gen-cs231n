import time
import datetime
import torch
import torch.nn as NN
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, StepLR, MultiStepLR
from torchvision.utils import save_image, make_grid
from utils import *

class Trainer(object):
    def __init__(self, data_loader, config):

        # Data loader
        self.data_loader = data_loader

        # exact model and loss
        self.model = config.model
        self.adv_loss = config.adv_loss

        # Model hyper-parameters
        self.imsize = config.imsize
        self.g_num = config.g_num
        self.z_dim = config.z_dim
        self.g_chn = config.g_chn
        self.ds_chn = config.ds_chn
        self.dt_chn = config.dt_chn
        self.n_frames = config.n_frames
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.lr_schr = config.lr_schr

        self.lambda_gp = config.lambda_gp
        self.total_epoch = config.total_epoch
        self.d_iters = config.d_iters
        self.g_iters = config.g_iters
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.n_class = config.n_class
        self.k_sample = config.k_sample
        self.dataset = config.dataset
        self.use_tensorboard = config.use_tensorboard
        self.test_batch_size = config.test_batch_size

        # path
        self.image_path = config.image_path
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path

        # epoch size
        self.log_epoch = config.log_epoch
        self.sample_epoch = config.sample_epoch
        self.model_save_epoch = config.model_save_epoch
        self.version = config.version

        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)

        self.device, self.parallel, self.gpus = set_device(config)

        self.build_model()

        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            print('load_pretrained_model...')
            self.load_pretrained_model()

    def label_sample(self):
        label = torch.randint(low=0, high=self.n_class, size=(self.batch_size, ))
        # label = torch.LongTensor(self.batch_size, 1).random_()%self.n_class
        # one_hot= torch.zeros(self.batch_size, self.n_class).scatter_(1, label, 1)
        return label.to(self.device)  # , one_hot.to(self.device)

    def wgan_loss(self, real_img, fake_img, tag):
        pre_alpha = torch.rand(real_img.size(0), 1, 1, 1)
        alphad = pre_alpha.cuda()
        alpha = alphad.expand_as(real_img)
        interp = alpha * real_img.data + (1 - alpha) * fake_img.data
        interpolated = torch.tensor(interp, requires_grad=True)
        out = self.D_s(interpolated) if tag == 'S' else self.D_t(interpolated)
        grad_full = torch.autograd.grad(
            outputs=out,
            inputs=interpolated,
            grad_outputs=torch.ones(out.size()).cuda(),
            retain_graph=True,
            create_graph=True,
            only_inputs=True
        )
        grad = grad_full[0]

