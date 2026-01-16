# Spectral Leray Projection

**Author:** Anthony Scott Hood  
**License:** MIT (c) 2026

A minimal, exact Fourier-space Leray projection for enforcing incompressibility in periodic 3D flows.

## Overview

This project implements the spectral Leray projection method using FFT to enforce divergence-free constraints on velocity fields. The method is particularly useful for incompressible fluid dynamics simulations where maintaining zero divergence is critical for solution accuracy.

## Features

- **Spectral Method**: Uses Fast Fourier Transform (FFT) for exact incompressibility enforcement
- **3D Periodic Domains**: Designed for periodic boundary conditions
- **Machine Precision**: Achieves incompressibility at machine precision levels
- **Minimal Implementation**: Clean, focused codebase for educational and research purposes

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install numpy
```

## Usage

```bash
python leray_projection.py
```

This will initialize a 32³ velocity field with a sinusoidal perturbation and run the divergence-free solver.

## Project Structure

- `leray_projection.py` - Main implementation of the spectral Leray projection
- `paper.md` / `paper.tex` - Academic paper describing the method
- `LICENSE` - MIT License
- `.gitignore` - Git ignore rules for Python and LaTeX files

## Mathematical Background

The Leray projection operator in Fourier space is defined as:

$$\mathbf{u}^* = \mathbf{\hat{u}} - \frac{\nabla \hat{p}}{\rho}$$

where the pressure is computed from the divergence of the velocity field to ensure incompressibility:

$$\nabla^2 p = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{\hat{u}}$$

## Requirements

- Python 3.7+
- NumPy

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{hood2026leray,
  title={A Minimal Spectral Leray Projection for Incompressible Flow},
  author={Hood, Anthony Scott},
  year={2026}
}
```
