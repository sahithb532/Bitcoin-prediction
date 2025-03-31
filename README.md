
# Bitcoin Price Prediction

## Overview
This project predicts Bitcoin prices using **Support Vector Regression (SVR)** based on historical data.

## Requirements
- Python (>=3.7)
- Pandas
- NumPy
- scikit-learn

### Install Dependencies
```sh
pip install numpy pandas scikit-learn
```

## Usage
1. Place **bitcoin.csv** in the project folder (must contain a **Price** column).
2. Run the script:
   ```sh
   python main.py
   ```
3. Outputs:
   - Model accuracy
   - Predicted Bitcoin prices for the next 30 days

## Structure
```
bitcoin-prediction/
│── main.py        # Script for training and prediction
│── bitcoin.csv    # Historical Bitcoin prices
│── README.md      # Project documentation
```

## License
Open-source project, free to use. 🚀

