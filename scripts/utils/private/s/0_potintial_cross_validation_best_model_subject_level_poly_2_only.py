import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

# Load the CSV file
print("Loading CSV file...")
df = pd.read_csv("E:\Project\CSV\data all_corrected\collected_metrics_Full_Brain_51_sum_t.csv")
#df = pd.read_csv("E:\Project\CSV\data all_corrected\collected_metrics_All_325_sum_t.csv")
# Define the columns for the organs
organs = ['Muscles', 'Liver', 'Kidneys',
 'Brain',
          'Fat',
          'Bone', 'Heart']
x_columns = [f"Log_V({organ})" for organ in organs]
y_columns = [f"Log_SUV_A({organ})" for organ in organs]

# Fill NaN values with column medians
print("Filling NaN values with column medians...")
df[x_columns] = df[x_columns].apply(lambda col: col.fillna(col.median()), axis=0)
df[y_columns] = df[y_columns].apply(lambda col: col.fillna(col.median()), axis=0)

# Ensure no NaNs remain in Age and Sex columns
print("Dropping rows with NaN values in Age and Sex...")
df = df.dropna(subset=['Age', 'Sex'])

# Create directory for saving plots
output_directory = r"E:\1Plots\plots_feb_24\potintial_cross_validation_best_model_subject_level_poly_2_only"
os.makedirs(output_directory, exist_ok=True)
print(f"Output directory: {output_directory}")

# Define common functions for polynomial regression
def fit_polynomial_model(X_train, y_train, degree=2):
    coefs = np.polyfit(X_train, y_train, degree)
    print(f"Fitted polynomial model of degree {degree}")
    return coefs

def calculate_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def calculate_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def calculate_ase(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def polynomial_predict(coefs, X):
    return np.polyval(coefs, X)

# Function to map organ color
def get_organ_color_map():
    return {
        'Muscles': 'blue',
        'Liver': 'green',
        'Kidneys': 'red',
        'Brain': 'purple',
        'Fat': 'orange',
        'Bone': 'brown',
        'Heart': 'pink'
    }

# K-Fold cross-validation setup
kf = KFold(n_splits=10, shuffle=True, random_state=42)
X_all = df[x_columns].values.flatten()
y_all = df[y_columns].values.flatten()

# Assign colors by organ for plotting
organ_colors = []
for organ in organs:
    organ_colors += [get_organ_color_map()[organ]] * len(df)

organ_colors = np.array(organ_colors)

# Evaluate 2nd-degree polynomial model
degree = 2
best_fold_r2, best_fold_mse, best_fold_ase = None, None, None
best_fold = None
fold_results = []

for fold, (train_index, test_index) in enumerate(kf.split(X_all)):
    X_train, X_test = X_all[train_index], X_all[test_index]
    y_train, y_test = y_all[train_index], y_all[test_index]

    # Fit polynomial model on the training data
    coefs = fit_polynomial_model(X_train, y_train, degree)

    # Predict on the test data using the polynomial model
    y_pred_test = polynomial_predict(coefs, X_test)

    # Calculate R², MSE, and ASE for this fold
    r2, mse, ase = calculate_r2(y_test, y_pred_test), calculate_mse(y_test, y_pred_test), calculate_ase(y_test, y_pred_test)

    # Save the fold results
    fold_results.append((fold + 1, r2, mse, ase, coefs))

    # Track the best fold based on R²
    if best_fold_r2 is None or r2 > best_fold_r2:
        best_fold_r2 = r2
        best_fold_mse = mse
        best_fold_ase = ase
        best_fold = fold + 1

print(f"Best fold for degree {degree}: Fold {best_fold} - R²: {best_fold_r2:.3f}, MSE: {best_fold_mse:.3f}, ASE: {best_fold_ase:.3f}")

# Now, apply the best fold's polynomial equation on the whole dataset
fold, best_r2, best_mse, best_ase, best_coefs = fold_results[best_fold - 1]

# Predict for the whole dataset using the best fold's model
y_pred_all = polynomial_predict(best_coefs, X_all)
r2_whole_data = calculate_r2(y_all, y_pred_all)
mse_whole_data = calculate_mse(y_all, y_pred_all)
ase_whole_data = calculate_ase(y_all, y_pred_all)

# Sort X_all and y_pred_all for smooth plotting
sorted_indices = np.argsort(X_all)
X_all_sorted = X_all[sorted_indices]
y_pred_all_sorted = y_pred_all[sorted_indices]

# Plot prediction vs actual for the whole dataset
plt.figure(figsize=(8, 6))

# Plot true values colored by organ
for organ in organs:
    indices = df.index[df[f'Log_V({organ})'].notnull()]
    plt.scatter(df[f'Log_V({organ})'], df[f'Log_SUV_A({organ})'],
                label=f'{organ}', color=get_organ_color_map()[organ], alpha=0.5)

# Plot the polynomial model prediction line
plt.plot(X_all_sorted, y_pred_all_sorted,
         label=f'Predicted (R²: {r2_whole_data:.3f}, MSE: {mse_whole_data:.3f}, ASE: {ase_whole_data:.3f})',
         color='green', alpha=0.6)

# Add the polynomial equation to the title
equation_text = ' + '.join([f'{coef:.3f}x^{i}' for i, coef in enumerate(best_coefs[::-1])])
plt.title(f'Best Fold Polynomial Model (Degree 2) Applied to Whole Data\nEquation: {equation_text}')

# Add labels and legend
plt.xlabel('Log_V')
plt.ylabel('Log_SUV_A')
plt.legend(loc='best')

# Save the plot
plot_name = 'prediction_vs_actual_best_fold_polynomial_degree_2.png'
plt.tight_layout()
plt.savefig(os.path.join(output_directory, plot_name))
print(f"Whole data plot for degree 2 saved: {plot_name}")
plt.close()

# Plot residuals for the whole dataset
residuals = y_all - y_pred_all
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_all, residuals, alpha=0.5, color='blue')
plt.axhline(0, color='red', linestyle='--')
plt.title(f'Residuals for Polynomial Degree 2')
plt.xlabel('Predicted Log_SUV_A')
plt.ylabel('Residuals (True - Predicted)')
plt.legend([f'Residuals (R²: {r2_whole_data:.3f}, MSE: {mse_whole_data:.3f}, ASE: {ase_whole_data:.3f})', 'Zero Residual Line'], loc='best')

# Save the residuals plot
plot_name = 'whole_data_residuals_polynomial_degree_2.png'
plt.tight_layout()
plt.savefig(os.path.join(output_directory, plot_name))
print(f"Residuals plot saved: {plot_name}")
plt.close()

# Plot predicted vs actual for the whole dataset
plt.figure(figsize=(8, 6))

# Plot predicted values vs actual values colored by organ
for organ in organs:
    indices = df.index[df[f'Log_V({organ})'].notnull()]
    plt.scatter(df[f'Log_SUV_A({organ})'], polynomial_predict(best_coefs, df[f'Log_V({organ})']),
                label=f'{organ}', color=get_organ_color_map()[organ], alpha=0.5)

# Add a diagonal reference line for y=x
min_val = min(y_all.min(), y_pred_all.min())
max_val = max(y_all.max(), y_pred_all.max())
plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', label='Ideal Prediction (y=x)')

# Add labels, title, and legend
plt.xlabel('Actual Log_SUV_A')
plt.ylabel('Predicted Log_SUV_A')
plt.title(f'Predicted vs Actual Log_SUV_A (Degree 2 Model)\nEquation: {equation_text}')
plt.legend(loc='best')

# Save the plot
plot_name = 'predicted_vs_actual_degree_2_polynomial.png'
plt.tight_layout()
plt.savefig(os.path.join(output_directory, plot_name))
print(f"Predicted vs Actual plot saved: {plot_name}")
plt.close()


def perform_10_fold_cv_and_plot(X, y, degree=2):
    """
    Perform 10-fold cross-validation, fit a polynomial model for each fold, and plot the results with the equation, R², MSE, and ASE.

    :param X: Input features (X)
    :param y: Target variable (y)
    :param degree: Degree of the polynomial model to fit (default is 2)
    """
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_num = 1

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit polynomial model on the training data
        coefs = fit_polynomial_model(X_train, y_train, degree)

        # Predict on the test data using the polynomial model
        y_pred_test = polynomial_predict(coefs, X_test)

        # Calculate R², MSE, and ASE for this fold
        r2 = calculate_r2(y_test, y_pred_test)
        mse = calculate_mse(y_test, y_pred_test)
        ase = calculate_ase(y_test, y_pred_test)

        # Sort the test data for smoother plotting
        sorted_indices = np.argsort(X_test)
        X_test_sorted = X_test[sorted_indices]
        y_pred_test_sorted = y_pred_test[sorted_indices]
        y_test_sorted = y_test[sorted_indices]  # Sorted actual values

        # Plot predicted vs actual for this fold
        plt.figure(figsize=(8, 6))

        # Plot only the test data points for this fold, colored by organ
        for organ in organs:
            # Select test data points that belong to this organ
            organ_test_indices = np.where(df[f'Log_V({organ})'].notnull())[0]
            organ_test_indices_in_fold = [i for i in test_index if i in organ_test_indices]

            if len(organ_test_indices_in_fold) > 0:
                plt.scatter(df[f'Log_V({organ})'].iloc[organ_test_indices_in_fold],
                            df[f'Log_SUV_A({organ})'].iloc[organ_test_indices_in_fold],
                            label=f'{organ}', color=get_organ_color_map()[organ], alpha=0.5)

        # Plot the polynomial model prediction line
        plt.plot(X_test_sorted, y_pred_test_sorted,
                 label=f'Predicted (R²: {r2:.3f}, MSE: {mse:.3f}, ASE: {ase:.3f})',
                 color='green', alpha=0.6)

        # Add the polynomial equation to the title
        equation_text = ' + '.join([f'{coef:.3f}x^{i}' for i, coef in enumerate(coefs[::-1])])
        plt.title(f'Fold {fold_num} - Polynomial Degree {degree} Model\nEquation: {equation_text}')

        # Add labels, title, and legend
        plt.xlabel('Log_V')
        plt.ylabel('Log_SUV_A')
        plt.legend(loc='best')

        # Save the plot with the fold number in the filename
        plot_name = f'prediction_vs_actual_fold_{fold_num}_degree_{degree}.png'
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, plot_name))
        print(f"Plot for fold {fold_num} saved: {plot_name}")
        plt.close()

        fold_num += 1


# Perform 10-fold cross-validation and plot results
X_all = df[x_columns].values.flatten()
y_all = df[y_columns].values.flatten()

perform_10_fold_cv_and_plot(X_all, y_all, degree=2)
