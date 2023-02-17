import numpy as np

def test_stokes_recon(setup_single_voxel_recon):
    recon = setup_single_voxel_recon

    # NOTE: skip retardance = 0 because orientation is not defined
    for retardance in np.linspace(1e-3, 1.0, 25): # fractions of a wave
        for orientation in np.linspace(0, np.pi-0.01, 25): # radians
            
            stokes = np.array([1,
                               -np.sin(retardance)*np.sin(2*orientation),
                               -np.sin(retardance)*np.cos(2*orientation),
                               np.cos(retardance)
                               ])[:, np.newaxis, np.newaxis]

            norm_stokes = recon.Stokes_transform(stokes)
            recon_params = recon.Polarization_recon(norm_stokes)

            assert retardance - recon_params[0] < 1e-8
            assert orientation - recon_params[1] < 1e-8
