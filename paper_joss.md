---
title: 'Spectral Leray Projection: A Minimal FFT-Based Solver for Incompressible Flow'
tags:
  - Python
  - computational fluid dynamics
  - incompressible flow
  - spectral methods
  - Leray projection
  - Navier-Stokes
authors:
  - name: Anthony Scott Hood
    orcid: 0000-0000-0000-0000  # Replace with your ORCID if you have one
    affiliation: 1
affiliations:
 - name: Independent Researcher
   index: 1
date: 16 January 2026
bibliography: paper.bib
---

# Summary

The enforcement of incompressibility (zero divergence) is a fundamental requirement in computational fluid dynamics (CFD) simulations of incompressible flows. This package provides a minimal, mathematically exact implementation of the spectral Leray projection operator, which projects an arbitrary velocity field onto its divergence-free component using Fast Fourier Transforms (FFTs). The method achieves machine-precision accuracy (divergence ~10⁻¹⁶) for periodic domains and is suitable for educational purposes, prototyping, and integration into larger CFD codes.

# Statement of Need

Incompressible flow solvers require a projection step to enforce the divergence-free constraint $\nabla \cdot \mathbf{u} = 0$. While many CFD packages implement this using finite-difference or finite-volume methods, spectral methods offer exact incompressibility enforcement (up to machine precision) for periodic domains. However, most spectral CFD codes are complex and difficult to understand for newcomers.

This package fills the gap by providing:

- A minimal (~200 line) reference implementation
- Clear mathematical documentation linking code to equations
- Verification through multiple test cases
- Machine-precision accuracy demonstration

The target audience includes:
- Students learning computational fluid dynamics
- Researchers prototyping new flow solver algorithms
- Developers needing a reference implementation for spectral projection

# Mathematical Background

The Leray projection operator $\mathbb{P}$ projects a velocity field $\mathbf{u}^*$ onto its divergence-free component:

$$\mathbf{u} = \mathbb{P}(\mathbf{u}^*) = \mathbf{u}^* - \nabla \phi$$

where $\phi$ solves the Poisson equation:

$$\nabla^2 \phi = \nabla \cdot \mathbf{u}^*$$

In Fourier space, this becomes algebraic. For wavenumber vector $\mathbf{k}$:

$$\hat{\mathbf{u}}(\mathbf{k}) = \hat{\mathbf{u}}^*(\mathbf{k}) - \frac{\mathbf{k}(\mathbf{k} \cdot \hat{\mathbf{u}}^*(\mathbf{k}))}{|\mathbf{k}|^2}$$

This implementation uses NumPy's FFT routines to compute this projection efficiently in $O(N \log N)$ time.

# Implementation

The core function `solve_pressure_fft()` implements the spectral Leray projection:

1. Transform velocity field to Fourier space using `numpy.fft.fftn()`
2. Compute divergence in Fourier space: $\hat{d} = i\mathbf{k} \cdot \hat{\mathbf{u}}^*$
3. Solve for pressure: $\hat{p} = -\hat{d}/|\mathbf{k}|^2$
4. Apply correction: $\hat{\mathbf{u}} = \hat{\mathbf{u}}^* - i\mathbf{k}\hat{p}$
5. Transform back to physical space using `numpy.fft.ifftn()`

The package includes verification through three test cases:
- Simple sinusoidal perturbation
- Taylor-Green vortex
- Smooth random velocity field

All tests achieve divergence at machine precision (~10⁻¹⁶ to 10⁻³²).

# Performance

For a 32³ grid:
- Single projection: ~10 milliseconds (MacBook Pro M1)
- Memory usage: ~3 MB for velocity field storage
- Scaling: $O(N^3 \log N)$ for $N^3$ grid points

# Example Usage

```python
import numpy as np
from leray_projection import solve_pressure_fft

# Create a 32³ velocity field
Nx, Ny, Nz = 32, 32, 32
u = np.random.randn(Nx, Ny, Nz, 3)

# Grid spacing
dx = dy = dz = 2*np.pi / 32
dt = 0.01

# Apply projection
u_divergence_free = solve_pressure_fft(u, dx, dy, dz, dt)
```

# Limitations

- Requires periodic boundary conditions
- Limited to Cartesian grids
- Not suitable for production simulations requiring complex geometries
- No time-stepping or viscosity (Navier-Stokes solver would require additional terms)

# Acknowledgments

This work implements the well-established spectral Leray projection method. The mathematical foundations are described in @Chorin1968 and @Temam1979.

# References
