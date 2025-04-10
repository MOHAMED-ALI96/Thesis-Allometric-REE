import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load your CSV file (update the path accordingly)
df = pd.read_csv("path/to/your/data.csv")

# Define a single combination of organs (example)
organs = ['WM', 'Heart', 'Liver', 'Kidneys', 'Muscles']

# Prepare the column names for independent (X) and dependent (Y) variables
x_columns = [f"Log_V_bmi({organ})" for organ in organs]
y_columns = [f"Log_SUV_A({organ})" for organ in organs]

# Extract the relevant data from the DataFrame
df_x = df[x_columns]
df_y = df[y_columns]

# Lists to hold per-subject regression outputs
results_list = []
r_squared_list = []

# For each subject, extract the organ measurements and fit a linear model
for i, row in df_x.iterrows():
    x_values = pd.to_numeric(row, errors='coerce').dropna().reset_index(drop=True)
    y_values = pd.to_numeric(df_y.iloc[i], errors='coerce').dropna().reset_index(drop=True)

    # Only fit a model if there are enough data points (at least 3)
    if len(x_values) < 3 or len(y_values) < 3:
        continue

    # Add a constant term and fit the OLS model
    X = sm.add_constant(x_values)
    try:
        model = sm.OLS(y_values, X).fit()
    except np.linalg.LinAlgError:
        continue

    # Store the slope (allometric exponent) and R-squared value
    results_list.append(model.params[1])
    r_squared_list.append(model.rsquared)

# Aggregate the results across subjects
avg_allometric_exponent = np.mean(results_list) if results_list else None
avg_r_squared = np.mean(r_squared_list) if r_squared_list else None

print("Average Allometric Exponent:", avg_allometric_exponent)
print("Average R-squared:", avg_r_squared)
