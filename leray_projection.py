import numpy as np

def solve_pressure_fft(u_star, dx, dy, dz, dt):
    rho = 1.0
    Nx, Ny, Nz, _ = u_star.shape

    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    kz = 2 * np.pi * np.fft.fftfreq(Nz, d=dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

    K2 = KX**2 + KY**2 + KZ**2
    K2[0,0,0] = 1.0

    U_hat = np.fft.fftn(u_star, axes=(0,1,2))

    div_U_hat = (
        1j * KX * U_hat[:,:,:,0] +
        1j * KY * U_hat[:,:,:,1] +
        1j * KZ * U_hat[:,:,:,2]
    )

    P_hat = -(rho / dt) * div_U_hat / K2
    P_hat[0,0,0] = 0.0

    U_hat[:,:,:,0] -= (dt / rho) * (1j * KX * P_hat)
    U_hat[:,:,:,1] -= (dt / rho) * (1j * KY * P_hat)
    U_hat[:,:,:,2] -= (dt / rho) * (1j * KZ * P_hat)

    return np.real(np.fft.ifftn(U_hat, axes=(0,1,2)))
