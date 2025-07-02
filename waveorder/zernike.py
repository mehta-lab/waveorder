import torch


def noll_to_zern(j: int) -> tuple[int, int]:
    n = 0
    j1 = j - 1
    while j1 > n:
        n += 1
        j1 -= n
    m = -n + 2 * j1
    return m, n


def factorial(n: int) -> int:
    return 1 if n < 2 else n * factorial(n - 1)


def zernike_radial(m: int, n: int, rho: torch.Tensor) -> torch.Tensor:
    R = torch.zeros_like(rho)
    for k in range((n - abs(m)) // 2 + 1):
        num = (-1.0) ** k * factorial(n - k)
        denom = (
            factorial(k)
            * factorial((n + abs(m)) // 2 - k)
            * factorial((n - abs(m)) // 2 - k)
        )
        R += num / denom * rho ** (n - 2 * k)
    return R


def zernike(
    m: int, n: int, rho: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    R = zernike_radial(m, n, rho)
    if m > 0:
        return R * torch.cos(m * theta)
    elif m < 0:
        return R * torch.sin(-m * theta)
    else:
        return R
