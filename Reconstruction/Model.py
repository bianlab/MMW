import torch
import numpy as np
import torch.nn.functional as F
from torch.fft import fft2, fftshift, ifft2, ifftshift


class sar_model(torch.nn.Module):
    def __init__(self,
                 f_start=30e9,
                 f_stop=35e9,
                 f_ds_rate=1,
                 r0=0.58,
                 Nx=121,
                 Nz=121,
                 Nf=51,
                 dx=0.005,
                 dz=0.005,
                 nx_im=128,
                 nz_im=128,
                 ny_im=64,
                 ny_im_true=None,
                 ds_rate=1,
                 rnd_seed=0,
                 mask=None,
                 dev=torch.device('cuda:0'),
                 ):
        super(sar_model, self).__init__()
        c = 299792458
        self.dev = dev
        self.ds_rate = ds_rate
        self.rnd_seed = rnd_seed
        self.R0 = r0
        self.Nx = Nx
        self.Nz = Nz
        self.nx_im = nx_im
        self.nz_im = nz_im
        self.ny_im = ny_im
        if ny_im_true is not None:
            self.ny_im_true = ny_im_true
        else:
            self.ny_im_true = ny_im

        if f_ds_rate == 1:
            f_step = (f_stop - f_start) / (Nf - 1)
            freq = torch.range(0, Nf - 1, device=self.dev)
            freq = freq * f_step
            freq = freq + f_start
            k = 2 * np.pi * freq / c
            k_mat = k.unsqueeze(0).unsqueeze(0)
            self.k_mat = k_mat.expand([self.Nz, self.Nx, -1])
            self.Nf = Nf
        else:
            f_step = (f_stop - f_start) / (Nf - 1)
            freq = torch.range(0, Nf-1, device=self.dev)
            freq = freq * f_step
            freq = freq + f_start
            freq = freq[0:-1:f_ds_rate]
            k = 2 * np.pi * freq / c
            k_mat = k.unsqueeze(0).unsqueeze(0)
            self.k_mat = k_mat.expand([self.Nz, self.Nx, -1])
            self.Nf = len(freq)

        B = f_stop - f_start
        f_now = f_start
        self.k_now = 2 * np.pi * f_now / c

        if mask is not None:
            self.sampling_mask_3d = mask
            self.sampling_mask_2d = mask.squeeze(2)
        else:
            self.sampling_mask_2d, self.sampling_mask_3d = self.random_sampling_mask()

        len_y = c / 2 / (f_step*f_ds_rate)
        dy_im = len_y / (ny_im - 1)
        y_im_arr = torch.range(0, self.ny_im_true - 1, device=self.dev)
        y_im_arr = y_im_arr - (self.ny_im_true - 1) / 2
        self.y_im_arr = y_im_arr * dy_im

        len_x = (Nx - 1) * dx
        len_z = (Nz - 1) * dz
        kx_step = (2 * np.pi / dx - 2 * np.pi / len_x) / (Nx - 1)
        kz_step = (2 * np.pi / dz - 2 * np.pi / len_z) / (Nz - 1)
        dx_im = len_x / (nx_im - 1)
        dz_im = len_z / (nz_im - 1)
        kx_im_step = (2 * np.pi / dx_im - 2 * np.pi / len_x) / (nx_im - 1)
        kz_im_step = (2 * np.pi / dz_im - 2 * np.pi / len_z) / (nz_im - 1)

        kx = torch.range(0, Nx - 1, 1, device=self.dev)
        kx = kx - (Nx - 1) / 2
        kx1 = 2 * kx / (Nx - 1)
        kx2 = kx*kx_step

        kx_im = torch.range(0, nx_im - 1, 1, device=self.dev)
        kx_im = kx_im - (nx_im - 1) / 2
        kx_im = kx_im * kx_im_step

        kz = torch.range(0, Nz - 1, 1, device=self.dev)
        kz = kz - (Nz - 1) / 2
        kz1 = 2 * kz / (Nz - 1)
        kz2 = kz*kz_step

        kz_im = torch.range(0, nz_im - 1, 1, device=self.dev)
        kz_im = kz_im - (nz_im - 1) / 2
        kz_im = kz_im * kz_im_step

        self.kz2_mat, self.kx2_mat = torch.meshgrid(kz2, kx2)
        self.kz2_mat, self.kx2_mat = self.kz2_mat, self.kx2_mat

        ind_kz_interp = (kz_im < torch.max(kz2+kz_im_step)) & (kz_im > torch.min(kz2-kz_im_step))
        ind_kz_interp = ind_kz_interp.nonzero()
        self.ind_kz_interp_start = ind_kz_interp[0]
        self.ind_kz_interp_end = ind_kz_interp[-1]
        ind_kx_interp = (kx_im < torch.max(kx2+kx_im_step)) & (kx_im > torch.min(kx2-kx_im_step))
        ind_kx_interp = ind_kx_interp.nonzero()
        self.ind_kx_interp_start = ind_kx_interp[0]
        self.ind_kx_interp_end = ind_kx_interp[-1]

        kz_im_mat, kx_im_mat = torch.meshgrid(kz_im, kx_im)
        self.kx_im_mat = torch.complex(real=kx_im_mat, imag=torch.zeros_like(kx_im_mat, device=self.dev))
        self.kz_im_mat = torch.complex(real=kz_im_mat, imag=torch.zeros_like(kz_im_mat, device=self.dev))

        kz_mat, kx_mat = torch.meshgrid(kz2, kx2)
        self.kx_mat = torch.complex(real=kx_mat, imag=torch.zeros_like(kx_mat, device=self.dev))
        self.kz_mat = torch.complex(real=kz_mat, imag=torch.zeros_like(kz_mat, device=self.dev))
        self.kx_mat_3d = self.kx_mat.unsqueeze(2).expand([-1, -1, self.Nf])
        self.kz_mat_3d = self.kz_mat.unsqueeze(2).expand([-1, -1, self.Nf])

        self.j1 = torch.complex(real=torch.Tensor([0]), imag=torch.Tensor([1])).to(self.dev)

        kkzx_mat = torch.sqrt(4*self.k_mat**2 - self.kz_mat_3d.real**2 - self.kx_mat_3d.real**2)
        kkzx_mat = kkzx_mat.unsqueeze(-1).repeat(1, 1, 1, self.ny_im_true)
        y_im_arr_mat = self.y_im_arr.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        y_im_arr_mat = y_im_arr_mat.repeat(self.Nz, self.Nx, self.Nf, 1)
        self.phase_mat = torch.exp(-self.j1*(self.R0 + y_im_arr_mat)*kkzx_mat)
        

    def default_2d_point_target(self):
        zx_img = torch.zeros(1, 1, self.nz_im, self.nx_im, device=self.dev)
        ind_mid = int(np.ceil(self.nz_im / 2))
        zx_img[:, :, ind_mid-1, ind_mid-1] = 1
        return zx_img

    def default_3d_point_target(self):
        zxy_img = torch.zeros(1, 1, self.nz_im, self.nx_im, self.ny_im_true, device=self.dev)
        zxy_img[:, :, 49, 99, 31] = 1
        # zxy_img[:, :, 383, 127, 11] = 1
        # zxy_img[:, :, 255, 163, 19] = 1
        return zxy_img

    def omega_k_for_2d_scene(self, S_echo):
        S_kxk = fft2(S_echo)
        S_kxk = fftshift(S_kxk)

        ky_sq = (2 * self.k_now)**2 - self.kx2_mat**2 - self.kz2_mat**2
        Neg = (ky_sq < 0)
        ky_sq[Neg] = 0
        ky_sq = torch.sqrt(ky_sq)

        Stolt = torch.complex(real=torch.zeros(1, 1, self.nz_im, self.nx_im, device=self.dev), imag=torch.zeros(1, 1, self.nz_im, self.nx_im, device=self.dev))
        Stolt[:, :, 0:self.Nz, 0:self.Nx] = S_kxk * torch.exp(self.j1 * ky_sq * self.R0)
        return ifft2(Stolt)

    def get_echo_for_2d_scene(self, zx_img):
        zx_img = fftshift(zx_img)
        zx_img = fft2(zx_img)
        zx_img = fftshift(zx_img)

        zx_img = zx_img[:, :, self.ind_kz_interp_start:self.ind_kz_interp_end, self.ind_kx_interp_start:self.ind_kx_interp_end]
        zx_img_real = F.interpolate(zx_img.real, size=(self.Nz, self.Nx), mode='bicubic')
        zx_img_imag = F.interpolate(zx_img.imag, size=(self.Nz, self.Nx), mode='bicubic')
        zx_img = torch.complex(real=zx_img_real, imag=zx_img_imag)

        phase_mat = torch.exp(-self.j1 * self.R0 * torch.sqrt(4 * self.k_now ** 2 - self.kz_mat ** 2 - self.kx_mat ** 2))
        zx_img = zx_img * phase_mat

        zx_img = ifftshift(zx_img)
        zx_img = ifft2(zx_img)
        S_echo = ifftshift(zx_img)
        S_echo = self.sampling_mask_2d * S_echo
        return S_echo

    def get_echo_for_3d_scene(self, zxy_img):
        zxy_img = fftshift(zxy_img, dim=(-3, -2))
        kz_kx_y_img = fft2(zxy_img, dim=(-3, -2))
        kz_kx_y_img = fftshift(kz_kx_y_img, dim=(-3, -2))
        if self.Nx % 2 == 0:
            ind_kz_interp_end = self.ind_kz_interp_end + 1
        else:
            ind_kz_interp_end = self.ind_kz_interp_end
        if self.Nz % 2 == 0:
            ind_kx_interp_end = self.ind_kx_interp_end + 1
        else:
            ind_kx_interp_end = self.ind_kx_interp_end

        kz_kx_y_img_img_interp_mat = kz_kx_y_img[:, :, self.ind_kz_interp_start:ind_kz_interp_end, self.ind_kx_interp_start:ind_kx_interp_end, :]
        # kz_kx_y_img_img_interp_mat = kz_kx_y_img[:, :, self.ind_kz_interp_start:self.ind_kz_interp_end+1, self.ind_kx_interp_start:self.ind_kx_interp_end+1, :]
        kz_kx_y_img_img_interp_mat = kz_kx_y_img_img_interp_mat.squeeze(1).unsqueeze(-2).repeat(1, 1, 1, self.Nf, 1)      
        kz_kx_y_img_img_interp_mat = torch.sum(kz_kx_y_img_img_interp_mat * self.phase_mat, dim=-1, keepdim=False)

        torch.autograd.set_detect_anomaly(True)

        zxy_img = ifftshift(kz_kx_y_img_img_interp_mat, dim=(-3, -2))
        zxy_img = ifft2(zxy_img, dim=(-3, -2))
        S_echo = ifftshift(zxy_img, dim=(-3, -2))
        S_echo = S_echo * self.sampling_mask_3d
        return S_echo

    def random_sampling_mask(self):
        if self.ds_rate == 1:
            return torch.ones(self.Nz, self.Nx, device=self.dev), torch.ones(self.Nz, self.Nx, 1, device=self.dev),
        else:
            num_ds = int(self.ds_rate*self.Nz*self.Nx)
            ind = np.concatenate((np.ones(num_ds), np.zeros(int(self.Nz*self.Nx-num_ds))), axis=0)
            np.random.seed(self.rnd_seed)
            ind_rand = np.random.permutation(ind)
            ind_rand_2d = np.reshape(ind_rand, (self.Nz, self.Nx))
            mask_rand_2d = torch.from_numpy(ind_rand_2d.astype('float32')).to(self.dev)
            mask_rand_3d = mask_rand_2d.unsqueeze(2)
            return mask_rand_2d, mask_rand_3d

    def forward(self, scene_real, scene_imag=0, complex=False):
        if complex:
            scene = scene_real
        else:
            scene = torch.complex(real=scene_real, imag=scene_imag)

        if scene.shape.__len__() == 4:
            S_echo = self.get_echo_for_2d_scene(scene)
        elif scene.shape.__len__() == 5:
            S_echo = self.get_echo_for_3d_scene(scene)
        else:
            print('Input must be 2d (B-C-H-W) or 3d (B-C-H-W-D) scene!')
            S_echo = None
        return S_echo
