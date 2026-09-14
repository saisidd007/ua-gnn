# Robust Traffic Flow Prediction using Graph Neural Networks

This project implements a robust traffic flow prediction system using Graph Neural Networks (GNNs) that can handle sensor failures in traffic networks.

## Project Overview

The system uses PyTorch Geometric (PyG) to implement a GNN-based model that:
- Processes traffic speed data from the PEMS-BAY dataset
- Handles missing sensor readings and sensor failures
- Predicts future traffic flow patterns
- Visualizes results and model performance

## Dataset

The project uses the PEMS-BAY dataset which includes:
- `PEMS-BAY.csv`: Traffic speed measurements from the Bay Area
- `adj_mx_bay.pkl`: Adjacency matrix representing road network connectivity
- `PEMS-BAY-META.csv`: Metadata about sensor locations and properties

## Project Structure

```
traffic-flow-gnn/
├── data/              # Dataset files
├── src/              # Source code
│   ├── models/       # GNN model implementations
│   └── utils/        # Helper functions and utilities
├── notebooks/        # Jupyter notebooks for analysis
├── results/          # Saved models and outputs
└── requirements.txt  # Project dependencies
```

## Setup Instructions

1. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Ensure dataset files are in the `data/` directory:
   - PEMS-BAY.csv
   - adj_mx_bay.pkl
   - PEMS-BAY-META.csv

## Usage

1. To train the model:
```bash
python src/train.py
```

2. To evaluate the model:
```bash
python src/test.py
```

3. For data exploration and visualization, check the notebooks in the `notebooks/` directory.

## Model Architecture

The GNN model implements:
- Graph Convolutional Networks (GCN) layers
- Temporal attention mechanisms
- Sensor failure simulation during training
- Robust prediction under missing data