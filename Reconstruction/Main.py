import os
import tqdm
import torch
import argparse
import numpy as np
from TV import TVloss
from CCN import resnet
from Model import sar_model
from torch.autograd import Variable
from scipy.io import loadmat, savemat
from torchvision.utils import save_image
from torchvision import transforms


parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', default='./ExemplarData/')
parser.add_argument('--data_name', default='data_0.25')
parser.add_argument('--save_intermediate', default=False, type=bool)
parser.add_argument('--use_init', default=True, type=bool)
parser.add_argument('--f_start', default=32e9, type=float, help='start freq')
parser.add_argument('--f_stop', default=36.98e9, type=float, help='stop freq')
parser.add_argument('--f_ds_rate', default=5, type=int, help='downsampling rate for Nf')
parser.add_argument('--Nf', default=250, type=int, help='freq points of array')
parser.add_argument('--Nx', default=186, type=int, help='horizontal points of array')
parser.add_argument('--Nz', default=430, type=int, help='vertical points of array')
parser.add_argument('--dx', default=0.005, type=float, help='horizontal scan interval')
parser.add_argument('--dz', default=0.004, type=float, help='vertical scan interval')
parser.add_argument('--R0', default=0.3, type=float, help='distance between object and array')
parser.add_argument('--Nx_im', default=186, type=int, help='horizontal resolution of reconstructed images')
parser.add_argument('--Nz_im', default=430, type=int, help='vertical resolution of reconstructed images')
parser.add_argument('--Ny_im', default=50, type=int, help='in-depth resolution of reconstructed images')
parser.add_argument('--Ny_im_true', default=32, type=int, help='in-depth resolution of reconstructed anchor regions')
parser.add_argument('--n_iter', default=1000, type=int, help='total iterations')
parser.add_argument('--device', default='cuda:0', help='default cuda:0')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--net_depth', default=5, type=int, help='depth of resbolck')
parser.add_argument('--sparse_weight', default=0, type=float,)
parser.add_argument('--tv_weight', default=5e-3, type=float,)
parser.add_argument('--enable_autocast', default=False, type=bool, help='automatic mixed precision')
opt = parser.parse_args()

# init
device = torch.device(opt.device)
data = loadmat(opt.data_folder+opt.data_name+'.mat')
S_echo_real = data['S_echo_real_sampling']
S_echo_imag = data['S_echo_imag_sampling']
S_echo_real = torch.from_numpy(S_echo_real.astype('float32')).unsqueeze(0)
S_echo_imag = torch.from_numpy(S_echo_imag.astype('float32')).unsqueeze(0)
S_echo = torch.complex(real=S_echo_real, imag=S_echo_imag)
measurement = Variable(S_echo, requires_grad=True).to(device)

if opt.use_init:
    input_init = data['scene_init']
    if opt.Ny_im_true < input_init.shape[-1]:
        input_init = input_init[:, :, int(opt.Ny_im/2-(opt.Ny_im-opt.Ny_im_true)/2-1):int(opt.Ny_im/2 + (opt.Ny_im-opt.Ny_im_true)/2 -1)]
    input_real = torch.from_numpy(np.real(input_init).astype('float32')).permute(2, 0, 1).unsqueeze(0)
    input_imag = torch.from_numpy(np.imag(input_init).astype('float32')).permute(2, 0, 1).unsqueeze(0)
    input_init = torch.from_numpy(np.abs(input_init).astype('float32')).permute(2, 0, 1).unsqueeze(0)
else:
    input_real = torch.rand(1, opt.Ny_im_true, opt.Nz_im, opt.Nx_im)
    input_imag = torch.rand(1, opt.Ny_im_true, opt.Nz_im, opt.Nx_im)
    input_init = torch.rand(1, opt.Ny_im_true, opt.Nz_im, opt.Nx_im)
input_real = Variable(input_real, requires_grad=True).to(device)
input_imag = Variable(input_imag, requires_grad=True).to(device)
input_init = Variable(input_init, requires_grad=True).to(device)

mask = data['sampling_mask']
mask = torch.from_numpy(mask.astype('float32')).unsqueeze(2).to(device)
sar_model = sar_model(f_start=opt.f_start,
                      f_stop=opt.f_stop,
                      f_ds_rate=opt.f_ds_rate,
                      r0=opt.R0,
                      Nx=opt.Nx,
                      Nz=opt.Nz,
                      Nf=opt.Nf,
                      dx=opt.dx,
                      dz=opt.dz,
                      nx_im=opt.Nx_im,
                      nz_im=opt.Nz_im,
                      ny_im=opt.Ny_im,
                      ny_im_true=opt.Ny_im_true,
                      dev=opt.device,
                      mask=mask,
                      )
input_depth = input_init.shape[1]
net = resnet(in_channel=input_depth, out_channel=opt.Ny_im_true, depth=opt.net_depth).to(device).bfloat16()

criterion = torch.nn.L1Loss().to(device)
tvl1_loss = TVloss(type=1).to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=100, verbose=True, min_lr=5e-6)
scaler = torch.cuda.amp.GradScaler()

if not os.path.exists('./Result'):
    os.makedirs('./Result')

# inference
pbar = tqdm.tqdm(range(opt.n_iter))
for epoch in range(opt.n_iter+1):
    optimizer.zero_grad()

    output_real, output_imag = net(input_real.bfloat16(), input_imag.bfloat16())
    output_real = output_real.permute(0, 2, 3, 1).unsqueeze(1).float()
    output_imag = output_imag.permute(0, 2, 3, 1).unsqueeze(1).float()

    pred_measurement = sar_model(output_real, output_imag, complex=False)
    if opt.enable_autocast:
        with torch.cuda.amp.autocast():
            loss_meas = criterion(pred_measurement.real, measurement.real) + criterion(pred_measurement.imag, measurement.imag)
            loss_tv = tvl1_loss(output_real) + tvl1_loss(output_imag)
            loss = loss_meas + opt.tv_weight*loss_tv
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss_meas = criterion(pred_measurement.real, measurement.real) + criterion(pred_measurement.imag, measurement.imag)
        loss_tv = tvl1_loss(output_real) + tvl1_loss(output_imag)
        loss = loss_meas + opt.tv_weight*loss_tv
        loss.backward()
        optimizer.step()
        schedular.step(loss_meas)

    if (loss_meas <= 1e-3) | (optimizer.param_groups[0]['lr'] <= 5e-6):
        break

    pbar.set_description("iteration: {}, lr: {}, fidelity loss: {:2f},  tv loss: {:2f}"
                         .format(epoch, optimizer.param_groups[0]['lr'], loss_meas.item(),  loss_tv.item()))
    pbar.update(1)

    with torch.no_grad():
        if (opt.save_intermediate) and (epoch>0) and (epoch % 100 == 0):
            abs_output = torch.sqrt(output_real ** 2 + output_imag ** 2)
            front_view = torch.max(abs_output, dim=4).values
            front_view = front_view/torch.max(front_view)
            save_image(front_view, "./Result/{}_iteration{}.jpg".format(opt.data_name, epoch))
        if epoch == opt.n_iter:
            proj_db = 20*torch.log10(front_view + 1e-10)
            proj_db = torch.roll(torch.flip(proj_db, [2]), -16, 2)
            proj_db = transforms.functional.resize(proj_db, (826, 512))
            proj_db[proj_db < -20] = -20
            proj_db += 20
            proj_db /= 20
            save_image(proj_db[0], "./Result/{}_to_detection.jpg".format(opt.data_name))

mat = {'real':output_real.detach().clone().cpu().numpy()[0, 0],
        'imag':output_imag.detach().clone().cpu().numpy()[0, 0]}
savemat('./Result/' + opt.data_name + '_reconstruction.mat', mat)
print('Reconstruction done!\n')
