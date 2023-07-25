import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
import os
import torch
from numpy.fft import fft, ifft, fft2, ifft2, fftn, ifftn, fftshift, ifftshift
from concurrent.futures import ProcessPoolExecutor
from .util import *
from .optics import *


def Jones_PC_forward_model(
    t_eigen,
    sa_orientation,
    fxx,
    fyy,
    xx,
    yy,
    N_defocus,
    N_channel,
    analyzer_para,
    Pupil_obj,
    Hz_det,
    time_re,
):
    plane_wave = np.exp(1j * 2 * np.pi * (fyy * yy + fxx * xx))

    N, M = xx.shape
    E_field = []
    E_field.append(plane_wave)
    E_field.append(1j * plane_wave)  # RHC illumination
    E_field = np.array(E_field)

    E_sample = Jones_sample(E_field, t_eigen, sa_orientation)

    Stokes_ang = np.zeros((4, N, M, N_defocus))
    I_meas_ang = np.zeros((N_channel, N, M, N_defocus))

    for m in range(N_defocus):
        Pupil_eff = Pupil_obj * Hz_det[:, :, m]
        E_field_out = ifft2(fft2(E_sample) * Pupil_eff)
        Stokes_ang[:, :, :, m] = Jones_to_Stokes(E_field_out)

        for n in range(N_channel):
            I_meas_ang[n, :, :, m] = (
                np.abs(
                    analyzer_output(
                        E_field_out, analyzer_para[n, 0], analyzer_para[n, 1]
                    )
                )
                ** 2
            )

    #     print('Processed %d, elapsed time: %.2f'%(os.getpid(), time.time() - time_re))

    return (Stokes_ang, I_meas_ang)


class waveorder_microscopy_simulator:
    def __init__(
        self,
        img_dim,
        lambda_illu,
        ps,
        NA_obj,
        NA_illu,
        z_defocus,
        chi,
        n_media=1,
        illu_mode="BF",
        NA_illu_in=None,
        Source=None,
        Source_PolState=np.array([1, 1j]),
        use_gpu=False,
        gpu_id=0,
    ):
        """

        initialize the system parameters for phase and orders microscopy

        """

        t0 = time.time()

        # GPU/CPU

        self.use_gpu = use_gpu
        self.gpu_id = gpu_id

        if self.use_gpu:
            globals()["cp"] = __import__("cupy")
            cp.cuda.Device(self.gpu_id).use()

        # Basic parameter
        self.N, self.M = img_dim
        self.n_media = n_media
        self.lambda_illu = lambda_illu / n_media
        self.ps = ps
        self.z_defocus = z_defocus.copy()
        if len(z_defocus) >= 2:
            self.psz = np.abs(z_defocus[0] - z_defocus[1])
        self.NA_obj = NA_obj / n_media
        self.NA_illu = NA_illu / n_media
        self.N_defocus = len(z_defocus)
        self.chi = chi

        # setup microscocpe variables
        self.xx, self.yy, self.fxx, self.fyy = gen_coordinate(
            (self.N, self.M), ps
        )
        self.frr = np.sqrt(self.fxx**2 + self.fyy**2)

        self.Pupil_obj = generate_pupil(
            self.frr, self.NA_obj, self.lambda_illu
        ).numpy()
        self.Pupil_support = self.Pupil_obj.copy()

        self.Hz_det = (
            generate_propagation_kernel(
                torch.tensor(self.frr),
                torch.tensor(self.Pupil_support),
                self.lambda_illu,
                torch.tensor(self.z_defocus),
            )
            .numpy()
            .transpose((1, 2, 0))
        )

        # illumination setup

        self.illumination_setup(illu_mode, NA_illu_in, Source, Source_PolState)

        self.analyzer_para = np.array(
            [
                [np.pi / 2, np.pi],
                [np.pi / 2 - self.chi, np.pi],
                [np.pi / 2, np.pi - self.chi],
                [np.pi / 2 + self.chi, np.pi],
                [np.pi / 2, np.pi + self.chi],
            ]
        )  # [alpha, beta]

        self.N_channel = len(self.analyzer_para)

    def illumination_setup(
        self, illu_mode, NA_illu_in, Source, Source_PolState
    ):
        if illu_mode == "BF":
            self.Source = generate_pupil(
                self.frr, self.NA_illu, self.lambda_illu
            ).numpy()
            self.N_pattern = 1

        elif illu_mode == "PH":
            if NA_illu_in == None:
                raise ("No inner rim NA specified in the PH illumination mode")
            else:
                self.NA_illu_in = NA_illu_in / self.n_media
                inner_pupil = generate_pupil(
                    self.frr,
                    self.NA_illu_in / self.n_media,
                    self.lambda_illu,
                ).numpy()
                self.Source = generate_pupil(
                    self.frr, self.NA_illu, self.lambda_illu
                ).numpy()
                self.Source -= inner_pupil

                Pupil_ring_out = generate_pupil(
                    self.frr,
                    self.NA_illu / self.n_media,
                    self.lambda_illu,
                ).numpy()
                Pupil_ring_in = generate_pupil(
                    self.frr,
                    self.NA_illu_in / self.n_media,
                    self.lambda_illu,
                ).numpy()

                self.Pupil_obj = self.Pupil_obj * np.exp(
                    (Pupil_ring_out - Pupil_ring_in)
                    * (np.log(0.7) - 1j * (np.pi / 2 - 0.0 * np.pi))
                )
                self.N_pattern = 1

        elif illu_mode == "Arbitrary":
            self.Source = Source.copy()
            if Source.ndim == 2:
                self.N_pattern = 1
            else:
                self.N_pattern = len(Source)

        self.Source_PolState = np.zeros((self.N_pattern, 2), complex)

        if Source_PolState.ndim == 1:
            for i in range(self.N_pattern):
                self.Source_PolState[i] = Source_PolState / (
                    np.sum(np.abs(Source_PolState) ** 2)
                ) ** (1 / 2)
        else:
            if len(Source_PolState) != self.N_pattern:
                raise (
                    "The length of Source_PolState needs to be either 1 or the same as N_pattern"
                )
            for i in range(self.N_pattern):
                self.Source_PolState[i] = Source_PolState[i] / (
                    np.sum(np.abs(Source_PolState[i]) ** 2)
                ) ** (1 / 2)

    def simulate_waveorder_measurements(
        self, t_eigen, sa_orientation, multiprocess=False
    ):
        Stokes_out = np.zeros(
            (4, self.N, self.M, self.N_defocus * self.N_pattern)
        )
        I_meas = np.zeros(
            (self.N_channel, self.N, self.M, self.N_defocus * self.N_pattern)
        )

        if multiprocess:
            t0 = time.time()
            for j in range(self.N_pattern):
                if self.N_pattern == 1:
                    [idx_y, idx_x] = np.where(self.Source == 1)
                else:
                    [idx_y, idx_x] = np.where(self.Source[j] == 1)

                N_source = len(idx_y)

                t_eigen_re = itertools.repeat(t_eigen, N_source)
                sa_orientation_re = itertools.repeat(sa_orientation, N_source)
                fxx = self.fxx[idx_y, idx_x].tolist()
                fyy = self.fyy[idx_y, idx_x].tolist()
                xx = itertools.repeat(self.xx, N_source)
                yy = itertools.repeat(self.yy, N_source)
                N_defocus = itertools.repeat(self.N_defocus, N_source)
                N_channel = itertools.repeat(self.N_channel, N_source)
                analyzer_para = itertools.repeat(self.analyzer_para, N_source)
                Pupil_obj = itertools.repeat(self.Pupil_obj, N_source)
                Hz_det = itertools.repeat(self.Hz_det, N_source)
                time_re = itertools.repeat(t0, N_source)

                with ProcessPoolExecutor(max_workers=64) as executor:
                    for result in executor.map(
                        Jones_PC_forward_model,
                        t_eigen_re,
                        sa_orientation_re,
                        fxx,
                        fyy,
                        xx,
                        yy,
                        N_defocus,
                        N_channel,
                        analyzer_para,
                        Pupil_obj,
                        Hz_det,
                        time_re,
                    ):
                        Stokes_out += result[0]
                        I_meas += result[1]

                print(
                    "Number of sources considered (%d / %d) in pattern (%d / %d), elapsed time: %.2f"
                    % (
                        N_source,
                        N_source,
                        j + 1,
                        self.N_pattern,
                        time.time() - t0,
                    )
                )

        else:
            t0 = time.time()
            for j in range(self.N_pattern):
                if self.N_pattern == 1:
                    [idx_y, idx_x] = np.where(self.Source == 1)
                    Source_current = self.Source.copy()
                else:
                    [idx_y, idx_x] = np.where(self.Source[j] == 1)
                    Source_current = self.Source[j].copy()
                N_source = len(idx_y)

                for i in range(N_source):
                    plane_wave = Source_current[idx_y[i], idx_x[i]] * np.exp(
                        1j
                        * 2
                        * np.pi
                        * (
                            self.fyy[idx_y[i], idx_x[i]] * self.yy
                            + self.fxx[idx_y[i], idx_x[i]] * self.xx
                        )
                    )
                    E_field = []
                    E_field.append(plane_wave)
                    E_field.append(1j * plane_wave)  # RHC illumination
                    E_field = np.array(E_field)

                    E_sample = Jones_sample(E_field, t_eigen, sa_orientation)

                    for m in range(self.N_defocus):
                        Pupil_eff = self.Pupil_obj * self.Hz_det[:, :, m]
                        E_field_out = ifft2(fft2(E_sample) * Pupil_eff)
                        Stokes_out[:, :, :, m] += Jones_to_Stokes(E_field_out)

                        for n in range(self.N_channel):
                            I_meas[n, :, :, m] += (
                                np.abs(
                                    analyzer_output(
                                        E_field_out,
                                        self.analyzer_para[n, 0],
                                        self.analyzer_para[n, 1],
                                    )
                                )
                                ** 2
                            )

                    if np.mod(i + 1, 100) == 0 or i + 1 == N_source:
                        print(
                            "Number of sources considered (%d / %d) in pattern (%d / %d), elapsed time: %.2f"
                            % (
                                i + 1,
                                N_source,
                                j + 1,
                                self.N_pattern,
                                time.time() - t0,
                            )
                        )

        return I_meas, Stokes_out

    def simulate_waveorder_inc_measurements(
        self, n_e, n_o, dz, mu, orientation, inclination
    ):
        Stokes_out = np.zeros(
            (4, self.N, self.M, self.N_defocus * self.N_pattern)
        )
        I_meas = np.zeros(
            (self.N_channel, self.N, self.M, self.N_defocus * self.N_pattern)
        )

        sample_norm_x = np.sin(inclination) * np.cos(orientation)
        sample_norm_y = np.sin(inclination) * np.sin(orientation)
        sample_norm_z = np.cos(inclination)

        wave_x = self.lambda_illu * self.fxx
        wave_y = self.lambda_illu * self.fyy
        wave_z = (np.maximum(0, 1 - wave_x**2 - wave_y**2)) ** (0.5)

        t0 = time.time()

        for j in range(self.N_pattern):
            if self.N_pattern == 1:
                [idx_y, idx_x] = np.where(self.Source >= 1)
                Source_current = self.Source.copy()
            else:
                [idx_y, idx_x] = np.where(self.Source[j] >= 1)
                Source_current = self.Source[j].copy()
            N_source = len(idx_y)

            for i in range(N_source):
                cos_alpha = (
                    sample_norm_x * wave_x[idx_y[i], idx_x[i]]
                    + sample_norm_y * wave_y[idx_y[i], idx_x[i]]
                    + sample_norm_z * wave_z[idx_y[i], idx_x[i]]
                )

                n_e_alpha = 1 / (
                    (1 - cos_alpha**2) / n_e**2 + cos_alpha**2 / n_o**2
                ) ** (0.5)

                t_eigen = np.zeros((2, self.N, self.M), complex)

                t_eigen[0] = np.exp(
                    -mu
                    + 1j
                    * 2
                    * np.pi
                    * dz
                    * (n_e_alpha / self.n_media - 1)
                    / self.lambda_illu
                )
                t_eigen[1] = np.exp(
                    -mu
                    + 1j
                    * 2
                    * np.pi
                    * dz
                    * (n_o / self.n_media - 1)
                    / self.lambda_illu
                )

                plane_wave = Source_current[idx_y[i], idx_x[i]] * np.exp(
                    1j
                    * 2
                    * np.pi
                    * (
                        self.fyy[idx_y[i], idx_x[i]] * self.yy
                        + self.fxx[idx_y[i], idx_x[i]] * self.xx
                    )
                )
                E_field = []
                E_field.append(plane_wave)
                E_field.append(1j * plane_wave)  # RHC illumination
                E_field = np.array(E_field)

                E_sample = Jones_sample(E_field, t_eigen, orientation)

                for m in range(self.N_defocus):
                    Pupil_eff = self.Pupil_obj * self.Hz_det[:, :, m]
                    E_field_out = ifft2(fft2(E_sample) * Pupil_eff)
                    Stokes_out[
                        :, :, :, m * self.N_pattern + j
                    ] += Jones_to_Stokes(E_field_out)

                    for n in range(self.N_channel):
                        I_meas[n, :, :, m * self.N_pattern + j] += (
                            np.abs(
                                analyzer_output(
                                    E_field_out,
                                    self.analyzer_para[n, 0],
                                    self.analyzer_para[n, 1],
                                )
                            )
                            ** 2
                        )

                if np.mod(i + 1, 100) == 0 or i + 1 == N_source:
                    print(
                        "Number of sources considered (%d / %d) in pattern (%d / %d), elapsed time: %.2f"
                        % (
                            i + 1,
                            N_source,
                            j + 1,
                            self.N_pattern,
                            time.time() - t0,
                        )
                    )

        return I_meas, Stokes_out

    def simulate_3D_scalar_measurements(self, t_obj):
        fr = (self.fxx**2 + self.fyy**2) ** (0.5)
        Pupil_prop = generate_pupil(self.frr, 1, self.lambda_illu).numpy()
        oblique_factor_prop = (
            (1 - self.lambda_illu**2 * fr**2) * Pupil_prop
        ) ** (1 / 2) / self.lambda_illu
        z_defocus = self.z_defocus - (self.N_defocus / 2 - 1) * self.psz
        Hz_defocus = Pupil_prop[:, :, np.newaxis] * np.exp(
            1j
            * 2
            * np.pi
            * z_defocus[np.newaxis, np.newaxis, :]
            * oblique_factor_prop[:, :, np.newaxis]
        )
        Hz_step = Pupil_prop * np.exp(
            1j * 2 * np.pi * self.psz * oblique_factor_prop
        )
        I_meas = np.zeros((self.N_pattern, self.N, self.M, self.N_defocus))

        if self.use_gpu:
            Hz_step = cp.array(Hz_step)
            Hz_defocus = cp.array(Hz_defocus)
            t_obj = cp.array(t_obj)
            Pupil_obj = cp.array(self.Pupil_obj)

        t0 = time.time()
        for i in range(self.N_pattern):
            if self.N_pattern == 1:
                [idx_y, idx_x] = np.where(self.Source >= 1)
                Source_current = self.Source.copy()
            else:
                [idx_y, idx_x] = np.where(self.Source[i] >= 1)
                Source_current = self.Source[i].copy()

            N_pt_source = len(idx_y)

            if self.use_gpu:
                I_temp = cp.zeros((self.N, self.M, self.N_defocus))

                for j in range(N_pt_source):
                    plane_wave = cp.array(
                        Source_current[idx_y[j], idx_x[j]]
                        * np.exp(
                            1j
                            * 2
                            * np.pi
                            * (
                                self.fyy[idx_y[j], idx_x[j]] * self.yy
                                + self.fxx[idx_y[j], idx_x[j]] * self.xx
                            )
                        )
                    )

                    for m in range(self.N_defocus):
                        if m == 0:
                            f_field = plane_wave.copy()

                        g_field = f_field * t_obj[:, :, m]

                        if m == self.N_defocus - 1:
                            f_field_stack_f = (
                                cp.fft.fft2(
                                    g_field[:, :, cp.newaxis], axes=(0, 1)
                                )
                                * Hz_defocus
                            )
                            I_temp += (
                                cp.abs(
                                    cp.fft.ifft2(
                                        f_field_stack_f
                                        * Pupil_obj[:, :, cp.newaxis],
                                        axes=(0, 1),
                                    )
                                )
                                ** 2
                            )

                        else:
                            f_field = cp.fft.ifft2(
                                cp.fft.fft2(g_field) * Hz_step
                            )

                    if np.mod(j + 1, 100) == 0 or j + 1 == N_pt_source:
                        print(
                            "Number of point sources considered (%d / %d) in pattern (%d / %d), elapsed time: %.2f"
                            % (
                                j + 1,
                                N_pt_source,
                                i + 1,
                                self.N_pattern,
                                time.time() - t0,
                            )
                        )
                I_meas[i] = cp.asnumpy(I_temp.copy())

            else:
                for j in range(N_pt_source):
                    plane_wave = Source_current[idx_y[j], idx_x[j]] * np.exp(
                        1j
                        * 2
                        * np.pi
                        * (
                            self.fyy[idx_y[j], idx_x[j]] * self.yy
                            + self.fxx[idx_y[j], idx_x[j]] * self.xx
                        )
                    )

                    for m in range(self.N_defocus):
                        if m == 0:
                            f_field = plane_wave

                        g_field = f_field * t_obj[:, :, m]

                        if m == self.N_defocus - 1:
                            f_field_stack_f = (
                                fft2(g_field[:, :, np.newaxis], axes=(0, 1))
                                * Hz_defocus
                            )
                            I_meas[i] += (
                                np.abs(
                                    ifft2(
                                        f_field_stack_f
                                        * self.Pupil_obj[:, :, np.newaxis],
                                        axes=(0, 1),
                                    )
                                )
                                ** 2
                            )

                        else:
                            f_field = ifft2(fft2(g_field) * Hz_step)

                    if np.mod(j + 1, 100) == 0 or j + 1 == N_pt_source:
                        print(
                            "Number of point sources considered (%d / %d) in pattern (%d / %d), elapsed time: %.2f"
                            % (
                                j + 1,
                                N_pt_source,
                                i + 1,
                                self.N_pattern,
                                time.time() - t0,
                            )
                        )

        return np.squeeze(I_meas)

    def simulate_3D_scalar_measurements_SEAGLE(
        self, RI_map, itr_max=100, tolerance=1e-4, verbose=False
    ):
        G_real = -gen_Greens_function_real(
            (2 * self.N, 2 * self.M, 2 * self.N_defocus),
            self.ps,
            self.psz,
            self.lambda_illu,
        )
        G_real_f = fftn(ifftshift(G_real)) * (self.ps**2) * (self.psz)

        f_scat = (2 * np.pi / self.lambda_illu) ** 2 * (
            1 - (RI_map / self.n_media) ** 2
        )

        fr = (self.fxx**2 + self.fyy**2) ** (0.5)
        Pupil_prop = gen_Pupil(self.fxx, self.fyy, 1, self.lambda_illu)
        oblique_factor_prop = (
            (1 - self.lambda_illu**2 * fr**2) * Pupil_prop
        ) ** (1 / 2) / self.lambda_illu
        z_defocus_m = self.z_defocus - (self.N_defocus / 2 - 1) * self.psz
        Hz_defocus = Pupil_prop[:, :, np.newaxis] * np.exp(
            1j
            * 2
            * np.pi
            * z_defocus_m[np.newaxis, np.newaxis, :]
            * oblique_factor_prop[:, :, np.newaxis]
        )

        I_meas = np.zeros((self.N_pattern, self.N, self.M, self.N_defocus))

        if self.use_gpu:
            Hz_defocus = cp.array(Hz_defocus)
            f_scat = cp.array(f_scat)
            Pupil_obj = cp.array(self.Pupil_obj)
            G_real_f = cp.array(G_real_f)

            pad_convolve_G = lambda x, y, z: cp.fft.ifftn(
                cp.fft.fftn(
                    cp.pad(
                        x,
                        (
                            (self.N // 2, self.N // 2),
                            (self.M // 2, self.M // 2),
                            (self.N_defocus // 2, self.N_defocus // 2),
                        ),
                        mode="constant",
                        constant_values=y,
                    )
                )
                * z
            )[
                self.N // 2 : -self.N // 2,
                self.M // 2 : -self.M // 2,
                self.N_defocus // 2 : -self.N_defocus // 2,
            ]

        else:
            pad_convolve_G = lambda x, y, z: ifftn(
                fftn(
                    np.pad(
                        x,
                        (
                            (self.N // 2,),
                            (self.M // 2,),
                            (self.N_defocus // 2,),
                        ),
                        mode="constant",
                        constant_values=y,
                    )
                )
                * z
            )[
                self.N // 2 : -self.N // 2,
                self.M // 2 : -self.M // 2,
                self.N_defocus // 2 : -self.N_defocus // 2,
            ]

        t0 = time.time()
        for i in range(self.N_pattern):
            if self.N_pattern == 1:
                [idx_y, idx_x] = np.where(self.Source > 0)
                Source_current = self.Source.copy()
            else:
                [idx_y, idx_x] = np.where(self.Source[i] > 0)
                Source_current = self.Source[i].copy()

            N_pt_source = len(idx_y)

            if self.use_gpu:
                I_temp = cp.zeros((self.N, self.M, self.N_defocus))

                for j in range(N_pt_source):
                    plane_wave = cp.array(
                        Source_current[idx_y[j], idx_x[j]]
                        * np.exp(
                            1j
                            * 2
                            * np.pi
                            * (
                                self.fyy[idx_y[j], idx_x[j]] * self.yy
                                + self.fxx[idx_y[j], idx_x[j]] * self.xx
                            )
                        )[:, :, np.newaxis]
                        * np.exp(
                            1j
                            * 2
                            * np.pi
                            * oblique_factor_prop[idx_y[j], idx_x[j]]
                            * self.z_defocus[np.newaxis, np.newaxis, :]
                        )
                    )
                    u = plane_wave + pad_convolve_G(
                        plane_wave * f_scat,
                        cp.asnumpy(cp.abs(cp.mean(plane_wave * f_scat))),
                        G_real_f,
                    )
                    err = np.zeros((itr_max + 1,))

                    tic_time = time.time()

                    for m in range(itr_max):
                        u_in_est = u - pad_convolve_G(
                            u * f_scat,
                            cp.asnumpy(cp.abs(cp.mean(u * f_scat))),
                            G_real_f,
                        )
                        diff_u = u_in_est - plane_wave
                        err[m + 1] = cp.asnumpy(cp.sum(cp.abs(diff_u) ** 2))

                        if err[m + 1] / err[1] < tolerance:
                            break

                        grad_u = (
                            diff_u
                            - pad_convolve_G(
                                diff_u,
                                cp.asnumpy(cp.abs(cp.mean(diff_u))),
                                G_real_f.conj(),
                            )
                            * f_scat.conj()
                        )

                        A_grad_u = grad_u - pad_convolve_G(
                            grad_u * f_scat,
                            cp.asnumpy(cp.abs(cp.mean(grad_u * f_scat))),
                            G_real_f,
                        )
                        step_size = cp.sum(cp.abs(grad_u) ** 2) / cp.sum(
                            cp.abs(A_grad_u) ** 2
                        )

                        temp = u - step_size * grad_u

                        if m == 0:
                            t = 1
                            u = temp.copy()
                            tempp = temp.copy()
                        else:
                            if err[m] < err[m + 1]:
                                t = 1
                                u = temp.copy()
                                tempp = temp.copy()
                            else:
                                tp = t
                                t = (1 + (1 + 4 * tp**2) ** (1 / 2)) / 2

                                u = temp + (tp - 1) * (temp - tempp) / t
                                tempp = temp.copy()
                        if verbose:
                            print(
                                "|  %d  |  %.2e  |   %.2f   |"
                                % (m + 1, err[m + 1], time.time() - tic_time)
                            )

                    I_temp += (
                        cp.abs(
                            cp.fft.ifft2(
                                cp.fft.fft2(u[:, :, -1])[:, :, cp.newaxis]
                                * Pupil_obj[:, :, cp.newaxis]
                                * Hz_defocus,
                                axes=(0, 1),
                            )
                        )
                        ** 2
                    )
                    if np.mod(j + 1, 1) == 0 or j + 1 == N_pt_source:
                        print(
                            "Number of point sources considered (%d / %d) in pattern (%d / %d), elapsed time: %.2f"
                            % (
                                j + 1,
                                N_pt_source,
                                i + 1,
                                self.N_pattern,
                                time.time() - t0,
                            )
                        )
                I_meas[i] = cp.asnumpy(I_temp.copy())

            else:
                for j in range(N_pt_source):
                    plane_wave = (
                        Source_current[idx_y[j], idx_x[j]]
                        * np.exp(
                            1j
                            * 2
                            * np.pi
                            * (
                                self.fyy[idx_y[j], idx_x[j]] * self.yy
                                + self.fxx[idx_y[j], idx_x[j]] * self.xx
                            )
                        )[:, :, np.newaxis]
                        * np.exp(
                            1j
                            * 2
                            * np.pi
                            * oblique_factor_prop[idx_y[j], idx_x[j]]
                            * self.z_defocus[np.newaxis, np.newaxis, :]
                        )
                    )
                    u = plane_wave + pad_convolve_G(
                        plane_wave * f_scat,
                        np.abs(np.mean(plane_wave * f_scat)),
                        G_real_f,
                    )
                    err = np.zeros((itr_max + 1,))

                    tic_time = time.time()

                    for m in range(itr_max):
                        u_in_est = u - pad_convolve_G(
                            u * f_scat, np.abs(np.mean(u * f_scat)), G_real_f
                        )
                        diff_u = u_in_est - plane_wave
                        err[m + 1] = np.sum(np.abs(diff_u) ** 2)

                        if err[m + 1] / err[1] < tolerance:
                            break

                        grad_u = (
                            diff_u
                            - pad_convolve_G(
                                diff_u,
                                np.abs(np.mean(diff_u)),
                                G_real_f.conj(),
                            )
                            * f_scat.conj()
                        )

                        A_grad_u = grad_u - pad_convolve_G(
                            grad_u * f_scat,
                            np.abs(np.mean(grad_u * f_scat)),
                            G_real_f,
                        )
                        step_size = np.sum(np.abs(grad_u) ** 2) / np.sum(
                            np.abs(A_grad_u) ** 2
                        )

                        temp = u - step_size * grad_u

                        if m == 0:
                            t = 1
                            u = temp.copy()
                            tempp = temp.copy()
                        else:
                            if err[m] < err[m + 1]:
                                t = 1
                                u = temp.copy()
                                tempp = temp.copy()
                            else:
                                tp = t
                                t = (1 + (1 + 4 * tp**2) ** (1 / 2)) / 2

                                u = temp + (tp - 1) * (temp - tempp) / t
                                tempp = temp.copy()
                        if verbose:
                            print(
                                "|  %d  |  %.2e  |   %.2f   |"
                                % (m + 1, err[m + 1], time.time() - tic_time)
                            )

                    I_meas[i] += (
                        np.abs(
                            ifft2(
                                fft2(u[:, :, -1])[:, :, np.newaxis]
                                * self.Pupil_obj[:, :, np.newaxis]
                                * Hz_defocus,
                                axes=(0, 1),
                            )
                        )
                        ** 2
                    )
                    if np.mod(j + 1, 1) == 0 or j + 1 == N_pt_source:
                        print(
                            "Number of point sources considered (%d / %d) in pattern (%d / %d), elapsed time: %.2f"
                            % (
                                j + 1,
                                N_pt_source,
                                i + 1,
                                self.N_pattern,
                                time.time() - t0,
                            )
                        )

        return np.squeeze(I_meas)

    def simulate_3D_vectorial_measurements_SEAGLE(
        self, epsilon_tensor, itr_max=100, tolerance=1e-4, verbose=False
    ):
        G_real = gen_Greens_function_real(
            (2 * self.N, 2 * self.M, 2 * self.N_defocus),
            self.ps,
            self.psz,
            self.lambda_illu,
        )
        G_tensor = gen_dyadic_Greens_tensor(
            G_real, self.ps, self.psz, self.lambda_illu, space="Fourier"
        )

        xx = fftshift(self.xx)
        yy = fftshift(self.yy)

        f_scat_tensor = np.zeros(
            (3, 3, self.N, self.M, self.N_defocus), complex
        )
        for p, q in itertools.product(range(3), range(3)):
            if p == q:
                f_scat_tensor[p, q] = (2 * np.pi / self.lambda_illu) ** 2 * (
                    1 - epsilon_tensor[p, q] / self.n_media**2
                )
            else:
                f_scat_tensor[p, q] = (2 * np.pi / self.lambda_illu) ** 2 * (
                    -epsilon_tensor[p, q] / self.n_media**2
                )

        fr = (self.fxx**2 + self.fyy**2) ** (0.5)
        Pupil_prop = generate_pupil(fr, 1, self.lambda_illu).numpy()
        oblique_factor_prop = (
            (1 - self.lambda_illu**2 * fr**2) * Pupil_prop
        ) ** (1 / 2) / self.lambda_illu
        z_defocus_m = self.z_defocus - (self.N_defocus / 2 - 1) * self.psz
        Hz_defocus = Pupil_prop[:, :, np.newaxis] * np.exp(
            1j
            * 2
            * np.pi
            * z_defocus_m[np.newaxis, np.newaxis, :]
            * oblique_factor_prop[:, :, np.newaxis]
        )

        if self.use_gpu:
            Hz_defocus = cp.array(Hz_defocus)
            f_scat_tensor = cp.array(f_scat_tensor)
            Pupil_obj = cp.array(self.Pupil_obj)
            G_tensor = cp.array(G_tensor)

        Stokes_SEAGLE = np.zeros(
            (4, self.N_pattern, self.N, self.M, self.N_defocus)
        )
        I_meas_SEAGLE = np.zeros(
            (self.N_channel, self.N_pattern, self.N, self.M, self.N_defocus)
        )

        t0 = time.time()
        for i in range(self.N_pattern):
            if self.N_pattern == 1:
                [idx_y, idx_x] = np.where(self.Source > 0)
                Source_current = self.Source.copy()
            else:
                [idx_y, idx_x] = np.where(self.Source[i] > 0)
                Source_current = self.Source[i].copy()

            N_pt_source = len(idx_y)

            for j in range(N_pt_source):
                E_in = np.zeros((3, self.N, self.M, self.N_defocus), complex)
                E_tot = np.zeros((3, self.N, self.M, self.N_defocus), complex)
                E_in_amp = np.zeros((3,), complex)

                if fr[idx_y[j], idx_x[j]] == 0:
                    E_in_amp[0] = self.Source_PolState[i, 0]
                    E_in_amp[1] = self.Source_PolState[i, 1]
                else:
                    E_in_amp[0] = (
                        self.Source_PolState[i, 0]
                        * (
                            (self.fxx[idx_y[j], idx_x[j]] ** 2)
                            * (
                                1
                                - self.lambda_illu**2
                                * fr[idx_y[j], idx_x[j]] ** 2
                            )
                            ** (1 / 2)
                            + self.fyy[idx_y[j], idx_x[j]] ** 2
                        )
                        + self.Source_PolState[i, 1]
                        * (
                            self.fxx[idx_y[j], idx_x[j]]
                            * self.fyy[idx_y[j], idx_x[j]]
                            * (
                                (
                                    1
                                    - self.lambda_illu**2
                                    * fr[idx_y[j], idx_x[j]] ** 2
                                )
                                ** (1 / 2)
                                - 1
                            )
                        )
                    ) / fr[idx_y[j], idx_x[j]] ** 2

                    E_in_amp[1] = (
                        self.Source_PolState[i, 0]
                        * (
                            self.fxx[idx_y[j], idx_x[j]]
                            * self.fyy[idx_y[j], idx_x[j]]
                            * (
                                (
                                    1
                                    - self.lambda_illu**2
                                    * fr[idx_y[j], idx_x[j]] ** 2
                                )
                                ** (1 / 2)
                                - 1
                            )
                        )
                        + self.Source_PolState[i, 1]
                        * (
                            (self.fyy[idx_y[j], idx_x[j]] ** 2)
                            * (
                                1
                                - self.lambda_illu**2
                                * fr[idx_y[j], idx_x[j]] ** 2
                            )
                            ** (1 / 2)
                            + self.fxx[idx_y[j], idx_x[j]] ** 2
                        )
                    ) / fr[idx_y[j], idx_x[j]] ** 2
                    E_in_amp[2] = -self.lambda_illu * (
                        self.Source_PolState[i, 0]
                        * self.fxx[idx_y[j], idx_x[j]]
                        + self.Source_PolState[i, 1]
                        * self.fyy[idx_y[j], idx_x[j]]
                    )

                #                     E_in_amp[0] = (self.fyy[idx_y[j], idx_x[j]] + \
                #                                    1j*self.fxx[idx_y[j], idx_x[j]]*(1 - self.lambda_illu**2 * fr[idx_y[j], idx_x[j]]**2)**(1/2))/fr[idx_y[j], idx_x[j]]
                #                     E_in_amp[1] = (-self.fxx[idx_y[j], idx_x[j]] + \
                #                                    1j*self.fyy[idx_y[j], idx_x[j]]*(1 - self.lambda_illu**2 * fr[idx_y[j], idx_x[j]]**2)**(1/2))/fr[idx_y[j], idx_x[j]]
                #                     E_in_amp[2] = -1j*self.lambda_illu*fr[idx_y[j], idx_x[j]]

                E_in_amp *= (Source_current[idx_y[j], idx_x[j]] / 2) ** (1 / 2)

                for p in range(3):
                    E_in[p] = (
                        E_in_amp[p]
                        * np.exp(
                            1j
                            * 2
                            * np.pi
                            * (
                                self.fyy[idx_y[j], idx_x[j]] * yy
                                + self.fxx[idx_y[j], idx_x[j]] * xx
                            )
                        )[:, :, np.newaxis]
                        * np.exp(
                            1j
                            * 2
                            * np.pi
                            * oblique_factor_prop[idx_y[j], idx_x[j]]
                            * self.z_defocus[np.newaxis, np.newaxis, :]
                        )
                    )

                if self.use_gpu:
                    #                     E_tot = cp.array(E_in.copy())
                    E_in = cp.array(E_in.copy())
                    E_tot = 2 * E_in - SEAGLE_vec_forward(
                        E_in,
                        f_scat_tensor,
                        G_tensor,
                        use_gpu=self.use_gpu,
                        gpu_id=self.gpu_id,
                    )

                else:
                    #                     E_tot = E_in.copy()
                    E_tot = 2 * E_in - SEAGLE_vec_forward(
                        E_in,
                        f_scat_tensor,
                        G_tensor,
                        use_gpu=self.use_gpu,
                        gpu_id=self.gpu_id,
                    )

                err = np.zeros((itr_max + 1,))

                tic_time = time.time()

                for m in range(itr_max):
                    E_in_est = SEAGLE_vec_forward(
                        E_tot,
                        f_scat_tensor,
                        G_tensor,
                        use_gpu=self.use_gpu,
                        gpu_id=self.gpu_id,
                    )
                    E_diff = E_in_est - E_in
                    if self.use_gpu:
                        err[m + 1] = cp.asnumpy(cp.sum(cp.abs(E_diff) ** 2))
                    else:
                        err[m + 1] = np.sum(np.abs(E_diff) ** 2)

                    if err[m + 1] / err[1] < tolerance:
                        break
                    grad_E = SEAGLE_vec_backward(
                        E_diff,
                        f_scat_tensor,
                        G_tensor,
                        use_gpu=self.use_gpu,
                        gpu_id=self.gpu_id,
                    )

                    A_grad_E = SEAGLE_vec_forward(
                        grad_E,
                        f_scat_tensor,
                        G_tensor,
                        use_gpu=self.use_gpu,
                        gpu_id=self.gpu_id,
                    )

                    if self.use_gpu:
                        step_size = cp.sum(cp.abs(grad_E) ** 2) / cp.sum(
                            cp.abs(A_grad_E) ** 2
                        )
                    else:
                        step_size = np.sum(np.abs(grad_E) ** 2) / np.sum(
                            np.abs(A_grad_E) ** 2
                        )

                    temp = E_tot - step_size * grad_E

                    if m == 0:
                        t = 1
                        E_tot = temp.copy()
                        tempp = temp.copy()
                    else:
                        if err[m] < err[m + 1]:
                            t = 1
                            E_tot = temp.copy()
                            tempp = temp.copy()
                        else:
                            tp = t
                            t = (1 + (1 + 4 * tp**2) ** (1 / 2)) / 2

                            E_tot = temp + (tp - 1) * (temp - tempp) / t
                            tempp = temp.copy()
                    if verbose:
                        print(
                            "|  %d  |  %.2e  |   %.2f   |"
                            % (m + 1, err[m + 1], time.time() - tic_time)
                        )

                if self.use_gpu:
                    E_field_out = cp.fft.ifft2(
                        cp.fft.fft2(E_tot[:2, :, :, -1], axes=(1, 2))[
                            :, :, :, cp.newaxis
                        ]
                        * (Pupil_obj[:, :, cp.newaxis] * Hz_defocus)[
                            cp.newaxis, :, :, :
                        ],
                        axes=(1, 2),
                    )
                    Stokes_SEAGLE[:, i, :, :, :] += cp.asnumpy(
                        Jones_to_Stokes(
                            E_field_out,
                            use_gpu=self.use_gpu,
                            gpu_id=self.gpu_id,
                        )
                    )
                    for n in range(self.N_channel):
                        I_meas_SEAGLE[n, i, :, :, :] += cp.asnumpy(
                            cp.abs(
                                analyzer_output(
                                    E_field_out,
                                    self.analyzer_para[n, 0],
                                    self.analyzer_para[n, 1],
                                )
                            )
                            ** 2
                        )

                else:
                    E_field_out = ifft2(
                        fft2(E_tot[:2, :, :, -1], axes=(1, 2))[
                            :, :, :, np.newaxis
                        ]
                        * (self.Pupil_obj[:, :, np.newaxis] * Hz_defocus)[
                            np.newaxis, :, :, :
                        ],
                        axes=(1, 2),
                    )
                    Stokes_SEAGLE[:, i, :, :, :] += Jones_to_Stokes(
                        E_field_out
                    )
                    for n in range(self.N_channel):
                        I_meas_SEAGLE[n, i, :, :, :] += (
                            np.abs(
                                analyzer_output(
                                    E_field_out,
                                    self.analyzer_para[n, 0],
                                    self.analyzer_para[n, 1],
                                )
                            )
                            ** 2
                        )

                if np.mod(j + 1, 1) == 0 or j + 1 == N_pt_source:
                    print(
                        "Number of point sources considered (%d / %d) in pattern (%d / %d), elapsed time: %.2f"
                        % (
                            j + 1,
                            N_pt_source,
                            i + 1,
                            self.N_pattern,
                            time.time() - t0,
                        )
                    )

        return I_meas_SEAGLE, Stokes_SEAGLE
