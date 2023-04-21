import torch
from waveorder import stokes
from waveorder import background_estimator


def calculate_transfer_function(
    swing,
    polarized_illumination_scheme,
    no_sample_intensities=None,  # if not None, apply this as background correction
    with_sample_intensities=None,  # if not None, estimate background, and remove it
):
    # Average `with_sample_intensities` over Z
    if with_sample_intensities.ndim == 4:
        with_sample_intensities = torch.mean(with_sample_intensities, dim=1)

    # Check input shapes
    if (
        no_sample_intensities is not None
        and with_sample_intensities is not None
    ):
        if no_sample_intensities.shape != with_sample_intensities.shape:
            raise ValueError(
                "no_sample_intensities.shape is not compatible with_sample_intensities.shape"
            )

    # Calculate A matrix
    intensity_to_stokes_matrix = stokes.calculate_intensity_to_stokes_matrix(
        swing, scheme=polarized_illumination_scheme
    )

    # Set default background correction matrix as identity
    inverse_background_mueller = torch.eye(4)

    # Calculate measured background correction
    if no_sample_intensities is not None:
        measured_no_sample_stokes = stokes.mmul(
            intensity_to_stokes_matrix, no_sample_intensities
        )
        inverse_background_mueller = stokes.mueller_from_stokes(
            *measured_no_sample_stokes, model="adr", direction="inverse"
        )

    # Calculate estimated background correction
    if with_sample_intensities is not None:
        measured_with_sample_stokes = stokes.mmul(
            intensity_to_stokes_matrix, with_sample_intensities
        )

        # Apply measured background correction to raw data
        # If no measured background correction is given, this applies the identity
        background_corrected_stokes = stokes.mmul(
            inverse_background_mueller, measured_with_sample_stokes
        )

        # Estimate additional background
        estimator = background_estimator.BackgroundEstimator2D()
        for stokes_index in range(background_corrected_stokes.shape[0]):
            background_corrected_stokes[
                stokes_index
            ] -= estimator.get_background(
                background_corrected_stokes[stokes_index],
                normalize=False,
            )

        # Calculate the "estimated-from-data" mueller matrix
        estimated_inverse_background_mueller = stokes.mueller_from_stokes(
            *background_corrected_stokes, model="adr", direction="inverse"
        )

        # Multiply the "estimated-from-data" and the "estimated-from-no-data"
        # Mueller matrices together. The resulting matrix is a one-step correction.
        inverse_background_mueller = stokes.mmul(
            estimated_inverse_background_mueller, inverse_background_mueller
        )

    return intensity_to_stokes_matrix, inverse_background_mueller


def visualize_transfer_function(
    viewer, intensity_to_stokes_matrix, inverse_background_mueller
):
    viewer.add_image(
        intensity_to_stokes_matrix.cpu().numpy(),
        name="Intensity to stokes matrix",
    )
    viewer.add_image(
        inverse_background_mueller.cpu().numpy(),
        name="Inverse background mueller",
    )


def apply_transfer_function(
    intensity_to_stokes_matrix,
    inverse_background_mueller,
    retardance,
    orientation,
    transmittance,
    depolarization,
):
    s = stokes.stokes_after_adr(
        retardance, orientation, transmittance, depolarization
    )

    stokes_to_intensity_matrix = torch.linalg.pinv(intensity_to_stokes_matrix)
    reverse_background = torch.linalg.pinv(intensity_to_stokes_matrix)

    return NotImplementedError


def apply_inverse_transfer_function(
    czyx_data, intensity_to_stokes_matrix, inverse_background_mueller
):
    data_stokes = stokes.mmul(intensity_to_stokes_matrix, czyx_data)
    corrected_stokes = stokes.mmul(inverse_background_mueller, data_stokes)
    return stokes.estimate_adr_from_stokes(*corrected_stokes)
