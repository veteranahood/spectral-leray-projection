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

if __name__ == "__main__":
    # Grid parameters
    Nx, Ny, Nz = 32, 32, 32
    Lx, Ly, Lz = 2*np.pi, 2*np.pi, 2*np.pi
    dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz
    
    # Create grid
    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)
    z = np.linspace(0, Lz, Nz, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Initialize velocity field
    u_init = np.zeros((Nx, Ny, Nz, 3))
    u_init[:,:,:,0] += 0.1 * np.sin(3*X)
    
    print("Initialization successful - divergence free solver ready")
