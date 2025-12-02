# Car Price Prediction Model

Predicts car prices using Linear Regression based on mileage and age.

## Requirements
- Python 3.7+
- pandas
- matplotlib
- scikit-learn
- joblib

## Installation
```bash
pip install pandas matplotlib scikit-learn joblib
```

## Dataset
- **File**: `carprices.csv`
- **Features**: Mileage, Age(yrs)
- **Target**: Sell Price($)

## Quick Start

### Load Data
```python
import pandas as pd
df = pd.read_csv('carprices.csv')
```

### Prepare Data
```python
x = df[['Mileage', 'Age(yrs)']]
y = df['Sell Price($)']
```

### Train Model
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
```

### Evaluate
```python
score = model.score(x_test, y_test)
print(f"R² Score: {score:.4f}")
```

### Predict
```python
new_car = [[150000, 5]]
price = model.predict(new_car)
print(f"Predicted Price: ${price[0]:,.2f}")
```

## Save & Load
```python
import joblib
joblib.dump(model, 'car_price_model.pkl')
loaded_model = joblib.load('car_price_model.pkl')
```

## Files
- `carprices.csv` - Raw dataset
- `Untitled-1.ipynb` - Complete Jupyter notebook
- `car_price_model.pkl` - Trained model
- `README.md` - Documentation

## Model Details
- **Algorithm**: Linear Regression
- **Train-Test Split**: 60% train, 40% test
- **Evaluation Metric**: R² Score

## Notes
- Ensure `carprices.csv` is in the same directory
- Use `random_state=42` for reproducibility
- Run notebook cells in order
