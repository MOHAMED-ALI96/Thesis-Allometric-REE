import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Load the CSV file
df = pd.read_csv(r"E:\Project\CSV\data all_corrected\Log_collected_metrics_Full_Brain_51_t.csv")

# Create a directory to save the comparison plots
output_directory = r"E:\\1Plots\\model_22_2_25"
os.makedirs(output_directory, exist_ok=True)

# Define the columns for the organs
organs = ['WM', 'Heart', 'Liver', 'Kidneys', 'Muscles', 'Bone', 'Fat']
x_columns = [f"V({organ})" for organ in organs]
y_columns = [f"SUV_A({organ})" for organ in organs]
colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta']

# Drop NaNs
df.dropna(subset=x_columns + y_columns, inplace=True)


# Define the four models
def power_law_model(x, a, b, c):
    return a * np.power(x, b) + c


def exponential_model(x, a, b, c):
    return a * np.exp(b * x) + c


def polynomial_model(x, a, b, c):
    return a * x ** 2 + b * x + c


def rational_model(x, a, b, c, d, e):
    return (a * x ** 2 + b * x + c) / (d * x + e)


# General fitting function
def fit_model(model_func, x_values, y_values, p0, bounds):
    try:
        popt, _ = curve_fit(model_func, x_values, y_values, p0=p0, bounds=bounds, maxfev=10000)
        return popt
    except RuntimeError:
        return None


# Plot each model independently
def plot_models(df, x_columns, y_columns, model_name, model_func, p0, bounds):
    combined_x_values = np.array([])
    combined_y_values = np.array([])

    for _, row in df.iterrows():
        x_values = row[x_columns].astype(float).values
        y_values = row[y_columns].astype(float).values
        combined_x_values = np.append(combined_x_values, x_values)
        combined_y_values = np.append(combined_y_values, y_values)

    popt = fit_model(model_func, combined_x_values, combined_y_values, p0, bounds)
    if popt is not None:
        y_pred = model_func(combined_x_values, *popt)
        r2 = r2_score(combined_y_values, y_pred)

        plt.figure(figsize=(12, 8))

        # Plot actual data points
        for i, (x_col, y_col, color) in enumerate(zip(x_columns, y_columns, colors)):
            plt.scatter(df[x_col], df[y_col], color=color, label=organs[i], alpha=0.6)

        # Define an extended x range for plotting
        extended_x_range = np.linspace(np.min(combined_x_values) - 2, np.max(combined_x_values) + 2, 100)
        y_pred_extended = model_func(extended_x_range, *popt)

        # Generate equation text
        if model_name == 'Power-Law':
            equation_text = f'SUV = {popt[0]:.3f} * Volume^{popt[1]:.3f} + {popt[2]:.3f}'
        elif model_name == 'Exponential':
            equation_text = f'SUV = {popt[0]:.3f} * exp({popt[1]:.3f} * Volume) + {popt[2]:.3f}'
        elif model_name == 'Polynomial':
            equation_text = f'SUV = {popt[0]:.3e} * Volume^2 + {popt[1]:.3e} * Volume + {popt[2]:.3f}'
        elif model_name == 'Rational':
            equation_text = f'SUV = ({popt[0]:.3e} * Volume^2 + {popt[1]:.3e} * Volume + {popt[2]:.3f}) / ({popt[3]:.3e} * Volume + {popt[4]:.3f})'

        plt.plot(extended_x_range, y_pred_extended, linewidth=2, color='red',
                 label=f'{model_name}: {equation_text}\nR² = {r2:.3f}')

        plt.xlabel('Volume')
        plt.ylabel('SUV')
        plt.legend()
        plt.title(f'{model_name} Model Fit')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, f'{model_name}_model_fit.png'))
        plt.show()


# Define initial guesses and bounds for each model
model_settings = {
    'Power-Law': (power_law_model, [1, 1, 1], ([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf])),
    'Exponential': (exponential_model, [1, 0.01, 1], ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])),
    'Polynomial': (polynomial_model, [1, 1, 1], ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])),
    'Rational': (rational_model, [1, 1, 1, 1, 1],
                 ([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf]))
}

# Plot each model
for model_name, (model_func, p0, bounds) in model_settings.items():
    plot_models(df, x_columns, y_columns, model_name, model_func, p0, bounds)
