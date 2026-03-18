"""Linear algebra utilities for waveorder.

Provides a closed-form SVD for 2×N complex matrices, replacing
torch.linalg.svd (cuSOLVER) with ~10-19× faster elementwise operations.
Uses real-arithmetic decomposition for torch.compile compatibility.
"""

import torch
from torch import Tensor


def closed_form_svd_2xN(A: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Closed-form SVD for (..., 2, N) complex matrices.

    For M=2, SVD reduces to eigendecomposition of the 2×2 Hermitian
    matrix A @ A^H, which has an analytical closed-form solution.

    Uses real-arithmetic decomposition of complex matmul so that
    torch.compile (Inductor) can fuse all operations.

    Parameters
    ----------
    A : Tensor
        Complex tensor of shape ``(..., 2, N)`` where N >= 2.

    Returns
    -------
    U : Tensor
        Left singular vectors, shape ``(..., 2, 2)``, complex.
    S : Tensor
        Singular values, shape ``(..., 2)``, real (float32/64).
    Vh : Tensor
        Right singular vectors (conjugate transpose), shape ``(..., 2, N)``, complex.
    """
    A_r = A.real
    A_i = A.imag

    # A @ A^H using real arithmetic (avoids complex ops for torch.compile)
    ArT = A_r.transpose(-2, -1)
    AiT = A_i.transpose(-2, -1)
    AAH_r = A_r @ ArT + A_i @ AiT  # (..., 2, 2) real part
    AAH_i = A_i @ ArT - A_r @ AiT  # (..., 2, 2) imag part

    # Extract 2×2 Hermitian elements
    a = AAH_r[..., 0, 0]  # real (diagonal)
    b_r = AAH_r[..., 0, 1]  # real part of off-diagonal
    b_i = AAH_i[..., 0, 1]  # imag part of off-diagonal
    c = AAH_r[..., 1, 1]  # real (diagonal)

    # Closed-form eigenvalues: λ = (a+c)/2 ± sqrt(((a-c)/2)² + |b|²)
    half_sum = (a + c) * 0.5
    b_abs_sq = b_r * b_r + b_i * b_i
    disc = torch.sqrt(((a - c) * 0.5) ** 2 + b_abs_sq)

    lam1 = half_sum + disc  # larger eigenvalue
    lam2 = half_sum - disc  # smaller eigenvalue

    # Singular values = sqrt(eigenvalues)
    # Clamp to small positive (not zero) to avoid inf gradient from sqrt(0)
    s1 = torch.sqrt(lam1.clamp(min=1e-16))
    s2 = torch.sqrt(lam2.clamp(min=1e-16))
    S = torch.stack([s1, s2], dim=-1)

    # Eigenvector for lam1: [b, lam1 - a] (unnormalized)
    # When |b| ≈ 0 (degenerate case), this vector is ~[0, lam1-a] which
    # has tiny norm and causes numerical issues. In that case eigenvalues
    # are a and c, and eigenvectors are [1,0] and [0,1] (or swapped).
    # We use the alternative formula [lam1 - c, conj(b)] which is stable
    # when |b| is small but |lam1 - c| is large (and vice versa).
    # Pick whichever formula gives a larger norm.
    #
    # Formula A: v = [b, lam1 - a]      (stable when |b| >> |lam1 - a|)
    # Formula B: v = [lam1 - c, conj(b)] (stable when |lam1 - c| >> |b|)

    # Compute both candidate eigenvectors and normalize them BEFORE selecting.
    # This avoids NaN gradients from torch.where evaluating both branches —
    # if one branch has zero norm, dividing by it produces inf in the gradient
    # even when that branch isn't selected.

    vA_0_r = b_r
    vA_0_i = b_i
    vA_1_r = lam1 - a
    normA = torch.sqrt(vA_0_r * vA_0_r + vA_0_i * vA_0_i + vA_1_r * vA_1_r).clamp(min=1e-8)
    nA_0_r = vA_0_r / normA
    nA_0_i = vA_0_i / normA
    nA_1_r = vA_1_r / normA

    vB_0_r = lam1 - c
    vB_1_r = b_r
    vB_1_i = -b_i  # conj(b)
    normB = torch.sqrt(vB_0_r * vB_0_r + vB_1_r * vB_1_r + vB_1_i * vB_1_i).clamp(min=1e-8)
    nB_0_r = vB_0_r / normB
    nB_1_r = vB_1_r / normB
    nB_1_i = vB_1_i / normB

    use_A = (normA >= normB).unsqueeze(-1)  # (..., 1) for broadcasting

    # Select from pre-normalized vectors (both branches are finite)
    # Pack into (..., 4) vectors: [re0, im0, re1, im1]
    vecA = torch.stack([nA_0_r, nA_0_i, nA_1_r, torch.zeros_like(nA_1_r)], dim=-1)
    vecB = torch.stack([nB_0_r, torch.zeros_like(nB_0_r), nB_1_r, nB_1_i], dim=-1)
    vec1 = torch.where(use_A, vecA, vecB)

    u1_0_r = vec1[..., 0]
    u1_0_i = vec1[..., 1]
    u1_1_r = vec1[..., 2]
    u1_1_i = vec1[..., 3]

    # Second eigenvector (orthogonal): [-conj(u1_1), conj(u1_0)]
    u2_0_r = -u1_1_r
    u2_0_i = u1_1_i  # -conj = negate real, keep imag (of negated)
    u2_1_r = u1_0_r
    u2_1_i = -u1_0_i

    # Build U as complex: (..., 2, 2)
    # U = [[u1_0, u2_0], [u1_1, u2_1]] where columns are eigenvectors
    U_r = torch.stack(
        [
            torch.stack([u1_0_r, u2_0_r], dim=-1),
            torch.stack([u1_1_r, u2_1_r], dim=-1),
        ],
        dim=-2,
    )
    U_i = torch.stack(
        [
            torch.stack([u1_0_i, u2_0_i], dim=-1),
            torch.stack([u1_1_i, u2_1_i], dim=-1),
        ],
        dim=-2,
    )
    U = torch.complex(U_r, U_i)

    # Vh = diag(1/σ) @ U^H @ A
    # Clamp S before reciprocal to avoid inf/NaN in backward pass.
    # (torch.where(S > eps, 1/S, 0) leaks NaN gradients from the 1/S branch)
    UH_A = U.conj().transpose(-2, -1) @ A  # (..., 2, N)
    s_safe = S.clamp(min=1e-8)
    s_inv = 1.0 / s_safe
    Vh = s_inv.unsqueeze(-1) * UH_A  # (..., 2, N)

    return U, S, Vh
