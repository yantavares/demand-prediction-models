```bash
  _______    _____    _____               __     __                
 |__   __|  / ____|  / ____|              \ \   / /                
    | |    | |      | |         ______     \ \_/ /    __ _   _ __  
    | |    | |      | |        |______|     \   /    / _` | | '_ \ 
    | |    | |____  | |____                  | |    | (_| | | | | |
    |_|     \_____|  \_____|                 |_|     \__,_| |_| |_|  
```

This repository implements and benchmarks a variety of classical and modern approaches to forecast Brazil's electricity demand, including time-series, machine-learning, and hybrid methods.

## Repository Structure

```
demand-prediction-models/
│
├── data/                             # Processed datasets
│   ├── no_exogenous/                 # Demand data without exogenous variables
│   └── with_exogenous/               # Demand data including weather and economic features
│
├── MATLAB/                           # Classical time-series models (MATLAB)
│   ├── ARMA/                         # ARMA (AutoRegressive Moving Average)
│   │   ├── arma.m                    # Training, forecasting, and evaluation script
│   │   └── README.md
│   ├── ARIMA/                        # ARIMA (AutoRegressive Integrated MA)
│   │   ├── arima.m
│   │   └── README.md
│   └── AWT/                          # AWT (ARIMA with Wavelet Transform)
│       ├── awt.m
│       └── README.md
│
├── Python/                           # Machine-learning and hybrid pipelines (Python)
│   ├── PROPHET/                      # Facebook Prophet model
│   │   ├── prophet.ipynb             # Training and evaluation notebook
│   │   └── requirements.txt
│   ├── RF/                           # Random Forest regression
│   │   ├── rf.ipynb
│   │   └── requirements.txt
│   ├── MLP/                          # Multilayer Perceptron
│   │   ├── mlp.ipynb
│   │   └── requirements.txt
│   ├── LSTM/                         # Long Short-Term Memory network
│   │   ├── lstm.ipynb
│   │   └── requirements.txt
│   ├── LSTM-Wavelet-RF/              # Hybrid wavelet + LSTM + Random Forest
│   │   ├── lstm_wavelet_rf.ipynb
│   │   └── requirements.txt
│   └── AFT/                          # AFT (ARIMA with Fourier Transform) in Python
│       ├── aft.ipynb                 # Notebook for Fourier filter and ARIMA modeling
│       └── requirements.txt
│
└── LICENSE                           # MIT License
```

## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/yantavares/demand-prediction-models.git
   cd demand-prediction-models
   ```

2. **Prepare the data**

   * Place cleaned CSV files into `data/` or use the provided datasets.

3. **Running MATLAB models**

   * Requires MATLAB R2021a or later (Signal Processing and Wavelet Toolboxes for AWT).
   * Navigate to the desired model folder (`MATLAB/ARMA/`, `MATLAB/ARIMA/`, or `MATLAB/AWT/`), and run:

     ```matlab
     % Example in MATLAB/AWT/
     awt
     ```
   * Outputs: forecasts, error metrics (MSE, RMSE, MAPE), and saved plots.

4. **Running Python models**

   * Install dependencies:

     ```bash
     pip install -r Python/PROPHET/requirements.txt \
                 -r Python/RF/requirements.txt \
                 -r Python/MLP/requirements.txt \
                 -r Python/LSTM/requirements.txt \
                 -r Python/LSTM-Wavelet-RF/requirements.txt \
                 -r Python/AFT/requirements.txt
     ```
   * Launch Jupyter Lab for your chosen notebook:

     ```bash
     jupyter lab Python/AFT/aft.ipynb  # or another folder of choice
     ```

## Model Overview

* **ARMA (MATLAB)**: Models autocorrelation in stationary series with autoregressive and moving-average components.
* **ARIMA (MATLAB)**: Extends ARMA with differencing to handle non-stationarity.
* **AWT (MATLAB)**: Applies wavelet decomposition before ARIMA for multi-resolution analysis.
* **AFT (Python)**: Uses Fourier decomposition to extract periodic and seasonal components prior to ARIMA modeling.
* **Prophet (Python)**: Additive model handling trends, seasonality, and holiday effects.
* **Random Forest (Python)**: Ensemble tree regressor capturing nonlinear relationships.
* **MLP (Python)**: Feed-forward neural network (multilayer perceptron).
* **LSTM (Python)**: Recurrent neural network capturing long-term dependencies.
* **LSTM-Wavelet-RF (Python)**: Hybrid pipeline combining wavelet decomposition, LSTM forecasting, and Random Forest correction.

## Usage Tips

* **Hyperparameter Tuning**: Adjust parameters in each script or notebook.
* **Cross-Validation**: Use `TimeSeriesSplit` in Python or rolling-origin evaluation in MATLAB.
* **Customization**: Modify data preprocessing and feature sets in `data/` before training.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
