"""Bounded optical preparation and a differentiable real-FFT consumer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from waveorder import optics, sampling, util
from waveorder._pixel_size import YXPixelSize

# Execution choices, not changes to the sampling or reconstruction domain.
_Z_SLAB_SIZE = 32
_FREQUENCY_BATCH_SIZE = 4096


@dataclass(frozen=True)
class _InverseFilter:
    values: Tensor
    logical_shape: tuple[int, int, int]
    z_padding: int


def _crop_indices(source_size, target_size, device):
    centered = (
        torch.arange(target_size, device=device) + (source_size - target_size) // 2 - source_size // 2
    ) % source_size
    return torch.fft.ifftshift(centered)


def _crop_and_partner_indices(source_size, target_size, device):
    indices = _crop_indices(source_size, target_size, device)
    retained, inverse = torch.unique(torch.cat((indices, (-indices) % source_size)), sorted=True, return_inverse=True)
    return retained, inverse[:target_size], inverse[target_size:]


def _correlation_slab(z_positions, window, oblique_factor, source_weight, detection_pupil, transverse_indices):
    carrier = torch.exp(1j * 2 * math.pi * z_positions[:, None, None] * oblique_factor[None, :, :])
    source_propagation = source_weight[:, None, :, :] * carrier
    greens = optics._greens_function_from_geometry(detection_pupil, oblique_factor, carrier)
    pupil_greens = detection_pupil[None, :, :] * greens
    del carrier, greens
    source_hat = torch.fft.fft2(source_propagation, dim=(-2, -1))
    greens_hat = torch.fft.fft2(pupil_greens, dim=(-2, -1))
    del source_propagation, pupil_greens
    correlation = torch.fft.ifft2(source_hat.conj() * greens_hat, dim=(-2, -1))
    del source_hat, greens_hat
    cropped = correlation.flatten(-2).index_select(-1, transverse_indices)
    del correlation
    return (cropped * window[None, :, None]).transpose(1, 2)


def _project_inverse(
    spectrum, crop_xy, partner_xy, crop_z, partner_z, direct_intensity, dz,
    logical_shape, z_padding, regularization, absorption_ratio, differentiating,
):
    out_z, out_y, out_x = logical_shape
    half_x = out_x // 2 + 1
    pixel_count = out_y * half_x
    negative_z = (-torch.arange(out_z, device=spectrum.device)) % out_z
    negative_crop_z = crop_z.index_select(0, negative_z)
    negative_partner_z = partner_z.index_select(0, negative_z)
    if differentiating:
        chunks = []
    else:
        output = torch.empty((1, out_z, pixel_count), dtype=torch.complex64, device=spectrum.device)

    def transfer(xy, z_crop, z_partner):
        h1 = spectrum[:, crop_xy[xy, None], z_crop[None, :]]
        h2 = spectrum[:, partner_xy[xy, None], z_partner[None, :]].conj()
        phase = ((h1 + h2) / direct_intensity) * dz
        if absorption_ratio is not None:
            absorption = (1j * (h1 - h2) / direct_intensity) * dz
            phase = phase + absorption_ratio * absorption
        return phase

    for start in range(0, pixel_count, _FREQUENCY_BATCH_SIZE):
        half_indices = torch.arange(start, min(start + _FREQUENCY_BATCH_SIZE, pixel_count), device=spectrum.device)
        y, x = half_indices // half_x, half_indices % half_x
        xy = y * out_x + x
        # Negate in the compact logical domain BEFORE original-grid mapping.
        negative_xy = ((-y) % out_y) * out_x + ((-x) % out_x)
        forward = transfer(xy, crop_z, partner_z)
        negative = transfer(negative_xy, negative_crop_z, negative_partner_z)
        inverse = forward.conj() / (forward.conj() * forward + regularization)
        negative_inverse = negative.conj() / (negative.conj() * negative + regularization)
        projected = ((inverse + negative_inverse.conj()) * 0.5).transpose(1, 2)
        if differentiating:
            chunks.append(projected)
        else:
            output[:, :, start:start + half_indices.numel()] = projected
        del forward, negative, inverse, negative_inverse, projected
    if differentiating:
        output = torch.cat(chunks, dim=-1)
    return _InverseFilter(output.reshape(out_z, out_y, half_x), logical_shape, z_padding)


def prepare_filter(
    zyx_shape, *, yx_pixel_size, z_pixel_size, wavelength_illumination, z_padding,
    index_of_refraction_media, numerical_aperture_illumination, numerical_aperture_detection,
    invert_phase_contrast, tilt_angle_zenith, tilt_angle_azimuth, pupil_steepness,
    regularization_strength, absorption_ratio, device,
):
    """Build projected compact G without materializing full H or full G.

    Scalar validation belongs to the facade. Tensor conversions retain gradients.
    Checkpointing bounds recomputed optical intermediates, not total backward memory.
    """
    na_ill, na_det, zen, azi = (
        torch.as_tensor(value, dtype=torch.float32, device=device).reshape(1)
        for value in (
            numerical_aperture_illumination, numerical_aperture_detection,
            tilt_angle_zenith, tilt_angle_azimuth,
        )
    )
    regularization = torch.as_tensor(regularization_strength, dtype=torch.float32, device=device)
    ratio = None if absorption_ratio is None else torch.as_tensor(absorption_ratio, dtype=torch.float32, device=device)
    # Integer sampling is not differentiable, matching the existing optical model.
    transverse_nyquist = sampling.transverse_nyquist(
        wavelength_illumination, float(na_ill[0].detach()), float(na_det[0].detach())
    )
    y_factor = int(np.ceil(yx_pixel_size.y / transverse_nyquist))
    x_factor = int(np.ceil(yx_pixel_size.x / transverse_nyquist))
    z_factor = int(np.ceil(z_pixel_size / sampling.axial_nyquist(
        wavelength_illumination, float(na_det[0].detach()), index_of_refraction_media
    )))
    out_z, out_y, out_x = zyx_shape[0] + 2 * z_padding, zyx_shape[1], zyx_shape[2]
    # Padding is added after oversampling; it is not itself oversampled.
    full_z = zyx_shape[0] * z_factor + 2 * z_padding
    full_y, full_x = out_y * y_factor, out_x * x_factor
    dz = z_pixel_size / z_factor
    sampled_pixels = YXPixelSize(y=yx_pixel_size.y / y_factor, x=yx_pixel_size.x / x_factor)
    fyy, fxx = util.generate_frequencies((full_y, full_x), sampled_pixels, device=device)
    radial_frequencies = torch.sqrt(fyy**2 + fxx**2)
    medium_wavelength = wavelength_illumination / index_of_refraction_media
    oblique_factor = torch.sqrt(torch.clamp(1 - medium_wavelength**2 * radial_frequencies**2, min=0.0))
    oblique_factor = oblique_factor / medium_wavelength
    z_positions = torch.fft.ifftshift((torch.arange(full_z, device=device) - full_z // 2) * dz)
    if invert_phase_contrast:
        z_positions = torch.flip(z_positions, dims=(0,))
    window = optics._wotf_axial_window(full_z, device)
    detection_pupil = optics.generate_pupil(
        radial_frequencies, na_det[0], wavelength_illumination, steepness=pupil_steepness
    )
    tilt_geometry = optics._tilted_pupil_geometry(fxx, fyy, wavelength_illumination, index_of_refraction_media)
    illumination_pupil = optics._tilted_pupil_from_geometry(
        fxx, fyy, na_ill[0], index_of_refraction_media, zen[:, None, None], azi[:, None, None], tilt_geometry
    )
    illumination_detection = illumination_pupil * detection_pupil
    direct_intensity = torch.sum(
        illumination_pupil * (detection_pupil * detection_pupil.conj()), dim=(-2, -1), keepdim=True
    )
    differentiating = torch.is_grad_enabled() and (
        illumination_detection.requires_grad or detection_pupil.requires_grad
    )
    del illumination_pupil, radial_frequencies, tilt_geometry, fxx, fyy
    # Hoist source-only work; preserve the cancellation-sensitive Green arithmetic.
    source_weight = illumination_detection * detection_pupil
    del illumination_detection
    retained_y, crop_y, partner_y = _crop_and_partner_indices(full_y, out_y, device)
    retained_x, crop_x, partner_x = _crop_and_partner_indices(full_x, out_x, device)
    retained_x_size = retained_x.numel()
    transverse_indices = (retained_y[:, None] * full_x + retained_x[None, :]).flatten()
    crop_xy = (crop_y[:, None] * retained_x_size + crop_x[None, :]).flatten()
    partner_xy = (partner_y[:, None] * retained_x_size + partner_x[None, :]).flatten()
    retained_count = transverse_indices.numel()
    if differentiating:
        slabs = []
    else:
        correlation = torch.empty((1, retained_count, full_z), dtype=torch.complex64, device=device)
    for start in range(0, full_z, _Z_SLAB_SIZE):
        stop = min(start + _Z_SLAB_SIZE, full_z)
        args = (
            z_positions[start:stop], window[start:stop], oblique_factor,
            source_weight, detection_pupil, transverse_indices,
        )
        if differentiating:
            slab = checkpoint(_correlation_slab, *args, use_reentrant=False, preserve_rng_state=False)
            slabs.append(slab)
        else:
            slab = _correlation_slab(*args)
            correlation[:, :, start:stop] = slab
        del slab, args
    if differentiating:
        correlation = torch.cat(slabs, dim=-1)
        del slabs
    del oblique_factor, source_weight, detection_pupil, transverse_indices, z_positions, window
    if differentiating:
        retained_z, crop_z, partner_z = _crop_and_partner_indices(full_z, out_z, device)
        spectra = []
        for start in range(0, retained_count, _FREQUENCY_BATCH_SIZE):
            transformed = torch.fft.fft(correlation[:, start:start + _FREQUENCY_BATCH_SIZE, :], dim=-1)
            spectra.append(transformed.index_select(-1, retained_z))
            del transformed
        spectrum = torch.cat(spectra, dim=1)
        del spectra, correlation
    else:
        for start in range(0, retained_count, _FREQUENCY_BATCH_SIZE):
            frequency_slice = correlation[:, start:start + _FREQUENCY_BATCH_SIZE, :]
            torch.fft.fft(frequency_slice, dim=-1, out=frequency_slice)
        del frequency_slice
        spectrum = correlation
        del correlation
        crop_z = _crop_indices(full_z, out_z, device)
        partner_z = (-crop_z) % full_z
    filter_differentiating = torch.is_grad_enabled() and (
        differentiating or regularization.requires_grad or (ratio is not None and ratio.requires_grad)
    )
    return _project_inverse(
        spectrum, crop_xy, partner_xy, crop_z, partner_z, direct_intensity, dz,
        (out_z, out_y, out_x), z_padding, regularization, ratio, filter_differentiating,
    )


def reconstruct(data: Tensor, inverse: _InverseFilter, out: Tensor | None = None) -> Tensor:
    padded = util.pad_zyx_along_z(data, inverse.z_padding)
    spectrum = torch.fft.rfftn(util.inten_normalization_3D(padded), dim=(-3, -2, -1))
    result = torch.fft.irfftn(spectrum * inverse.values, s=inverse.logical_shape, dim=(-3, -2, -1))
    if inverse.z_padding:
        result = result[inverse.z_padding:-inverse.z_padding]
    if out is not None:
        out.copy_(result)
        return out
    return result
