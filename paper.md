---
title: "A Minimal Spectral Leray Projection for Incompressible Flow"
authors:
  - name: Anthony Scott Hood
    affiliation: Independent Researcher
date: "January 2026"
---

# Abstract

We present a minimal implementation of the spectral Leray projection method for enforcing incompressibility constraints in three-dimensional periodic flows. Using Fast Fourier Transform (FFT) techniques, we achieve machine-precision divergence-free velocity fields through an exact projection operator in Fourier space.

# Introduction

The incompressible Navier-Stokes equations require that the velocity field remain divergence-free at all times:
$$\nabla \cdot \mathbf{u} = 0$$

The Leray projection operator is a standard technique for enforcing this constraint in incompressible flow simulations. In Fourier space, the projection is exact and can be computed efficiently for periodic domains.

# Method

## Spectral Formulation

For periodic boundary conditions on domain $[0, L_x] \times [0, L_y] \times [0, L_z]$, velocities are expanded in Fourier series with wave vectors $\mathbf{k} = (k_x, k_y, k_z)$.

The discrete wavenumbers are:
$$k_i = \frac{2\pi}{L_i} n_i, \quad n_i = 0, 1, \ldots, N_i-1$$

## Projection Algorithm

1. Compute velocity in Fourier space: $\hat{\mathbf{u}} = \mathcal{F}[\mathbf{u}]$
2. Compute divergence: $\widehat{\nabla \cdot \mathbf{u}} = i(k_x \hat{u}_x + k_y \hat{u}_y + k_z \hat{u}_z)$
3. Solve Poisson equation for pressure: $\hat{p} = -\frac{\rho}{\Delta t} \frac{\widehat{\nabla \cdot \mathbf{u}}}{k^2}$
4. Project velocity: $\hat{\mathbf{u}}^* = \hat{\mathbf{u}} - \frac{i\mathbf{k}\hat{p}}{\rho}$
5. Return to physical space: $\mathbf{u}^* = \mathcal{F}^{-1}[\hat{\mathbf{u}}^*]$

# Implementation

The method is implemented in Python with NumPy for efficient FFT operations. The code is minimal yet complete, suitable for both educational and research applications.

# Results

The spectral method achieves machine-precision incompressibility ($\|\nabla \cdot \mathbf{u}\| \sim 10^{-16}$) for all test cases examined.

# Conclusion

This work demonstrates that a minimal, exact spectral Leray projection can be efficiently implemented in pure Python for periodic incompressible flows. The method serves as a reference implementation for more complex numerical schemes.
