# Geometric Constraints in Flow-Based Models of Single-Cell Dynamics

This repository contains code and experiments for a course project on deterministic flow matching with geometric regularization for single-cell trajectory inference.

## Overview
We study whether enforcing local tangent constraints on flow-matching vector fields improves the recovery of meaningful trajectories from snapshot single-cell data. Using synthetic branching datasets (e.g., the petal dataset), we show that while tangent regularization improves local smoothness and geometric adherence, deterministic trajectories fail in symmetric branching settings due to fundamental identifiability limitations.

## Repository Structure
`01_preprocess_eb_data.ipynb` contains steps followed to download and preprocess the data.
`02_train_model_eb.ipynb` and `02_train_model_synthetic_petal.ipynb` makes use of written utils to train the model. 
`model.py` contains the model.

## Requirements
See `requirements.lock` for Python dependencies.

## Reproducibility
All experiments were run with fixed random seeds. Synthetic datasets are generated programmatically.