import numpy as np
import waveorder as wo


def pol_3D_from_phantom():
    # Adapted from waveorder/examples/2D_QLIPP_simulation/2D_QLIPP_forward.py

    N = 512  # number of pixel in y dimension
    M = 512  # number of pixel in x dimension
    mag = 20  # magnification
    ps = 6.5 / mag  # effective pixel size
    lambda_illu = 0.532  # wavelength
    n_media = 1  # refractive index in the media
    NA_obj = 0.4  # objective NA
    NA_illu = 0.2  # illumination NA (condenser)
    N_defocus = 11
    z_defocus = (
        np.r_[:N_defocus] - N_defocus // 2
    ) * 2  # a set of defocus planes
    chi = 0.1 * 2 * np.pi  # swing of Polscope analyzer

    # Generate phantom
    star, theta, _ = wo.genStarTarget(N, M)
    phase_value = 1  # average phase in radians (optical path length)
    phi_s = star * (phase_value + 0.15)  # slower OPL across target
    phi_f = star * (phase_value - 0.15)  # faster OPL across target
    mu_s = np.zeros((N, M))  # absorption
    mu_f = mu_s.copy()
    t_eigen = np.zeros((2, N, M), complex)  # complex specimen transmission
    t_eigen[0] = np.exp(-mu_s + 1j * phi_s)
    t_eigen[1] = np.exp(-mu_f + 1j * phi_f)
    _, _, fxx, fyy = wo.gen_coordinate((N, M), ps)
    sa = theta % np.pi  # slow axes.

    # Generate source
    Source_cont = wo.gen_Pupil(fxx, fyy, NA_illu, lambda_illu)
    Source_discrete = wo.Source_subsample(
        Source_cont, lambda_illu * fxx, lambda_illu * fyy, subsampled_NA=0.1
    )

    # Generate simulator
    simulator = wo.waveorder_microscopy_simulator(
        (N, M),
        lambda_illu,
        ps,
        NA_obj,
        NA_illu,
        z_defocus,
        chi,
        n_media=n_media,
        illu_mode="Arbitrary",
        Source=Source_discrete,
    )

    # Simulate measurements
    I_meas, _ = simulator.simulate_waveorder_measurements(
        t_eigen, sa, multiprocess=False
    )

    # Add background
    photon_count = 300000
    ext_ratio = 10000
    const_bg = photon_count / (0.5 * (1 - np.cos(chi))) / ext_ratio
    I_meas_noise = (
        np.random.poisson(I_meas / np.max(I_meas) * photon_count + const_bg)
    ).astype("float64")

    # Return data and background
    data = I_meas_noise.transpose(0, 3, 1, 2)
    bkg = np.ones((5, N, M)) * const_bg
    return data, bkg


def bf_3D_from_phantom():
    # Adapted from waveorder/3D_PODT_phase_simulation/3D_PODT_Phase_forward.py

    N = 512  # number of pixel in y dimension
    M = 512  # number of pixel in x dimension
    mag = 20  # magnification
    ps = 6.5 / mag  # effective pixel size
    lambda_illu = 0.532  # wavelength
    n_media = 1  # refractive index in the media
    NA_obj = 0.4  # objective NA
    NA_illu = 0.2  # illumination NA (condenser)
    N_defocus = 31
    psz = 2
    z_defocus = (
        np.r_[:N_defocus] - N_defocus // 2
    ) * psz  # a set of defocus planes
    chi = 0.1  # not used

    # Generate sample
    phantom = np.zeros((N, M, N_defocus), dtype=np.complex64)
    star, _, _ = wo.genStarTarget(N, M)
    n_sample = 1.50
    t_obj = np.exp(1j * 2 * np.pi * psz * (star * n_sample - n_media))
    phantom[:, :, N_defocus // 2] = t_obj
    phantom += n_media

    # Generate source
    _, _, fxx, fyy = wo.gen_coordinate((N, M), ps)
    Source_cont = wo.gen_Pupil(fxx, fyy, NA_illu, lambda_illu)
    Source_discrete = wo.Source_subsample(
        Source_cont, lambda_illu * fxx, lambda_illu * fyy, subsampled_NA=0.1
    )

    # Simulate
    simulator = wo.waveorder_microscopy_simulator(
        (N, M),
        lambda_illu,
        ps,
        NA_obj,
        NA_illu,
        z_defocus,
        chi,
        n_media=n_media,
        illu_mode="Arbitrary",
        Source=Source_discrete,
    )
    data = simulator.simulate_3D_scalar_measurements(phantom)
    return data.transpose((2, 0, 1))


def fluorescence_from_phantom():
    # Adapted from waveorder/examples/fluorescence_deconvolution/fluorescence_deconv.ipynb

    N = 512  # number of pixel in y dimension
    M = 512  # number of pixel in x dimension
    mag = 20  # magnification
    ps = 6.5 / mag  # effective pixel size
    psz = 2
    lambda_emiss = [0.532]  # wavelength
    n_media = 1  # refractive index in the media
    NA_obj = 0.4  # objective NA
    N_defocus = 11

    fluor_setup = wo.fluorescence_microscopy(
        (N, M, N_defocus),
        lambda_emiss,
        ps,
        psz,
        NA_obj,
        n_media=n_media,
        deconv_mode="3D-WF",
        pad_z=10,
        use_gpu=False,
        gpu_id=0,
    )

    star, _, _ = wo.genStarTarget(N, M)

    f = np.zeros((N, M, fluor_setup.N_defocus_3D))
    f[:, :, fluor_setup.N_defocus_3D // 2] = star
    F = np.fft.fftn(f)
    H = fluor_setup.OTF_WF_3D[0, ...]
    g = np.abs(np.fft.ifftn(F * H))
    return g.transpose((2, 0, 1))
