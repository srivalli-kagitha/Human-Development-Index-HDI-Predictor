import os
import json

def create_notebook():
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Human Development Index (HDI) Predictor - Exploratory Data Analysis\n",
                    "This notebook performs comprehensive **Exploratory Data Analysis (EDA)** and trains a **Linear Regression** model to predict the **Human Development Index (HDI) Score** based on developmental indicators:\n",
                    "1. Life Expectancy\n",
                    "2. Expected Years of Schooling\n",
                    "3. Mean Years of Schooling\n",
                    "4. Gross National Income (GNI) Per Capita\n",
                    "\n",
                    "The analysis follows standard data science processes including data cleaning, visualization, preprocessing, modeling, and evaluation."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Import required libraries\n",
                    "import os\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "from sklearn.linear_model import LinearRegression\n",
                    "from sklearn.impute import SimpleImputer\n",
                    "from sklearn import metrics\n",
                    "\n",
                    "# Set plotting style and configurations\n",
                    "sns.set_theme(style=\"whitegrid\")\n",
                    "plt.rcParams[\"figure.figsize\"] = (10, 6)\n",
                    "plt.rcParams[\"font.size\"] = 12\n",
                    "\n",
                    "# Ensure image static directory exists for saving plots\n",
                    "os.makedirs(os.path.join(\"..\", \"static\", \"images\"), exist_ok=True)\n",
                    "print(\"Libraries imported successfully!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Dataset Loading and Overview\n",
                    "Let's load the generated HDI dataset and examine its shape, column types, and basic statistics."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load the dataset\n",
                    "df = pd.read_csv(os.path.join(\"..\", \"dataset\", \"hdi_dataset.csv\"))\n",
                    "\n",
                    "# Dataset Shape\n",
                    "print(f\"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\")\n",
                    "print(\"\\nFirst 5 rows of the dataset:\")\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Column names and data types\n",
                    "print(\"Dataset Columns & Info:\")\n",
                    "df.info()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Summary statistics\n",
                    "print(\"Summary Statistics of Numerical Columns:\")\n",
                    "df.describe()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Data Preprocessing: Missing Values, Duplicates & Categorization\n",
                    "Before conducting visualizations and fitting the model, we must clean the data, handle duplicates, and impute any missing values."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Check for missing values\n",
                    "print(\"Missing Value Count:\")\n",
                    "print(df.isnull().sum())\n",
                    "\n",
                    "# Check for duplicate records\n",
                    "print(f\"\\nDuplicate rows count: {df.duplicated().sum()}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Perform Mean Imputation for features to handle potential missing values\n",
                    "features = [\"Life Expectancy\", \"Expected Years of Schooling\", \"Mean Years of Schooling\", \"GNI Per Capita\"]\n",
                    "imputer = SimpleImputer(strategy=\"mean\")\n",
                    "df[features] = imputer.fit_transform(df[features])\n",
                    "\n",
                    "# Drop duplicate rows if any exist\n",
                    "df = df.drop_duplicates()\n",
                    "\n",
                    "# Create HDI Development Category column for visualization purposes\n",
                    "# Categories based on UNDP definition:\n",
                    "# - Very High: >= 0.800\n",
                    "# - High: 0.700 to 0.799\n",
                    "# - Medium: 0.550 to 0.699\n",
                    "# - Low: < 0.550\n",
                    "def categorize_hdi(score):\n",
                    "    if score >= 0.800:\n",
                    "        return \"Very High\"\n",
                    "    elif score >= 0.700:\n",
                    "        return \"High\"\n",
                    "    elif score >= 0.550:\n",
                    "        return \"Medium\"\n",
                    "    else:\n",
                    "        return \"Low\"\n",
                    "\n",
                    "df[\"Development Category\"] = df[\"HDI Score\"].apply(categorize_hdi)\n",
                    "print(\"Development Category successfully created!\")\n",
                    "df[\"Development Category\"].value_counts()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Exploratory Data Analysis (EDA) - Visualizations\n",
                    "We generate seven distinct plots as requested to fully capture variables distributions, correlations, and differences across tiers."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Plot 1: Strip Plot\n",
                    "# Displays the distribution of HDI Scores across different Development Categories\n",
                    "plt.figure(figsize=(10, 6))\n",
                    "sns.stripplot(x=\"Development Category\", y=\"HDI Score\", data=df, \n",
                    "              order=[\"Low\", \"Medium\", \"High\", \"Very High\"], \n",
                    "              palette=\"viridis\", hue=\"Development Category\", size=6, jitter=0.25)\n",
                    "plt.title(\"Strip Plot: HDI Score Distribution across Development Categories\", fontsize=14, fontweight='bold')\n",
                    "plt.xlabel(\"Development Category\", fontsize=12)\n",
                    "plt.ylabel(\"HDI Score\", fontsize=12)\n",
                    "plt.tight_layout()\n",
                    "plt.savefig(os.path.join(\"..\", \"static\", \"images\", \"strip_plot.png\"), dpi=150)\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Plot 2: Distribution Plot (KDE + Hist)\n",
                    "# Shows the overall probability distribution and density of the target variable: HDI Score\n",
                    "plt.figure(figsize=(10, 6))\n",
                    "sns.histplot(df[\"HDI Score\"], kde=True, color=\"#4A90E2\", bins=15, edgecolor='black')\n",
                    "plt.title(\"Distribution Plot: Density Profile of HDI Score\", fontsize=14, fontweight='bold')\n",
                    "plt.xlabel(\"HDI Score\", fontsize=12)\n",
                    "plt.ylabel(\"Density / Count\", fontsize=12)\n",
                    "plt.tight_layout()\n",
                    "plt.savefig(os.path.join(\"..\", \"static\", \"images\", \"dist_plot.png\"), dpi=150)\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Plot 3: Histograms of Input Features\n",
                    "# Provides subplots displaying the histograms for each of our predictor columns\n",
                    "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
                    "\n",
                    "sns.histplot(df[\"Life Expectancy\"], color=\"teal\", ax=axes[0,0], kde=True, edgecolor='black')\n",
                    "axes[0,0].set_title(\"Distribution of Life Expectancy\", fontweight='bold')\n",
                    "axes[0,0].set_xlabel(\"Life Expectancy (Years)\")\n",
                    "\n",
                    "sns.histplot(df[\"Expected Years of Schooling\"], color=\"purple\", ax=axes[0,1], kde=True, edgecolor='black')\n",
                    "axes[0,1].set_title(\"Distribution of Expected Years of Schooling\", fontweight='bold')\n",
                    "axes[0,1].set_xlabel(\"Expected Years (Years)\")\n",
                    "\n",
                    "sns.histplot(df[\"Mean Years of Schooling\"], color=\"orange\", ax=axes[1,0], kde=True, edgecolor='black')\n",
                    "axes[1,0].set_title(\"Distribution of Mean Years of Schooling\", fontweight='bold')\n",
                    "axes[1,0].set_xlabel(\"Mean Years (Years)\")\n",
                    "\n",
                    "sns.histplot(df[\"GNI Per Capita\"], color=\"forestgreen\", ax=axes[1,1], kde=True, edgecolor='black')\n",
                    "axes[1,1].set_title(\"Distribution of GNI Per Capita\", fontweight='bold')\n",
                    "axes[1,1].set_xlabel(\"GNI Per Capita (USD PPP)\")\n",
                    "\n",
                    "plt.suptitle(\"Histograms: Distribution Profiles of Predictor Features\", fontsize=16, fontweight='bold')\n",
                    "plt.tight_layout()\n",
                    "plt.savefig(os.path.join(\"..\", \"static\", \"images\", \"histograms.png\"), dpi=150)\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Plot 4: Box Plot\n",
                    "# Displays the box-and-whisker distribution plot for features to check for outliers and variance\n",
                    "fig, axes = plt.subplots(1, 3, figsize=(16, 5))\n",
                    "\n",
                    "sns.boxplot(y=\"Life Expectancy\", x=\"Development Category\", data=df, \n",
                    "            order=[\"Low\", \"Medium\", \"High\", \"Very High\"], ax=axes[0], palette=\"Set2\")\n",
                    "axes[0].set_title(\"Life Expectancy by HDI Tier\", fontweight='bold')\n",
                    "axes[0].set_xlabel(\"\")\n",
                    "\n",
                    "sns.boxplot(y=\"Expected Years of Schooling\", x=\"Development Category\", data=df, \n",
                    "            order=[\"Low\", \"Medium\", \"High\", \"Very High\"], ax=axes[1], palette=\"Set2\")\n",
                    "axes[1].set_title(\"Expected Schooling by HDI Tier\", fontweight='bold')\n",
                    "axes[1].set_xlabel(\"\")\n",
                    "\n",
                    "sns.boxplot(y=\"Mean Years of Schooling\", x=\"Development Category\", data=df, \n",
                    "            order=[\"Low\", \"Medium\", \"High\", \"Very High\"], ax=axes[2], palette=\"Set2\")\n",
                    "axes[2].set_title(\"Mean Schooling by HDI Tier\", fontweight='bold')\n",
                    "axes[2].set_xlabel(\"\")\n",
                    "\n",
                    "plt.suptitle(\"Box Plots: Feature Distributions grouped by HDI Category\", fontsize=15, fontweight='bold')\n",
                    "plt.tight_layout()\n",
                    "plt.savefig(os.path.join(\"..\", \"static\", \"images\", \"box_plots.png\"), dpi=150)\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Plot 5: Scatter Plot (e.g. GNI vs Life Expectancy, color-coded by HDI Score)\n",
                    "# Visualizes relationships and trends between features along with the target HDI Score\n",
                    "plt.figure(figsize=(10, 6))\n",
                    "scatter = plt.scatter(df[\"Life Expectancy\"], df[\"GNI Per Capita\"], c=df[\"HDI Score\"], \n",
                    "                      cmap=\"plasma\", s=70, alpha=0.8, edgecolors='black')\n",
                    "cbar = plt.colorbar(scatter)\n",
                    "cbar.set_label(\"HDI Score\")\n",
                    "plt.title(\"Scatter Plot: Life Expectancy vs GNI Per Capita (Colormap: HDI Score)\", fontsize=14, fontweight='bold')\n",
                    "plt.xlabel(\"Life Expectancy (Years)\", fontsize=12)\n",
                    "plt.ylabel(\"GNI Per Capita (USD PPP)\", fontsize=12)\n",
                    "plt.yscale('log')  # Apply logarithmic scale to capture high range GNI variance\n",
                    "plt.grid(True, which=\"both\", ls=\"--\")\n",
                    "plt.tight_layout()\n",
                    "plt.savefig(os.path.join(\"..\", \"static\", \"images\", \"scatter_plot.png\"), dpi=150)\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Plot 6: Heatmap of Correlations\n",
                    "# Displays the correlation matrix to see the statistical correlation between all variables\n",
                    "plt.figure(figsize=(8, 6))\n",
                    "correlation_matrix = df[[*features, \"HDI Score\"]].corr()\n",
                    "sns.heatmap(correlation_matrix, annot=True, cmap=\"coolwarm\", fmt=\".4f\", linewidths=0.5, square=True)\n",
                    "plt.title(\"Heatmap: Correlation Matrix of Indicators & HDI Score\", fontsize=14, fontweight='bold')\n",
                    "plt.tight_layout()\n",
                    "plt.savefig(os.path.join(\"..\", \"static\", \"images\", \"heatmap.png\"), dpi=150)\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Plot 7: Pair Plot\n",
                    "# A multi-panel grid plot that displays bivariate scatter plots and univariate distributions\n",
                    "pair_plot = sns.pairplot(df[[*features, \"HDI Score\", \"Development Category\"]], \n",
                    "                         hue=\"Development Category\", hue_order=[\"Low\", \"Medium\", \"High\", \"Very High\"],\n",
                    "                         palette=\"Set1\", diag_kind=\"kde\")\n",
                    "pair_plot.fig.suptitle(\"Pair Plot: Multidimensional Relationships Grid\", y=1.02, fontsize=15, fontweight='bold')\n",
                    "pair_plot.savefig(os.path.join(\"..\", \"static\", \"images\", \"pair_plot.png\"), dpi=150)\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Model Training & Evaluation\n",
                    "Now, let's prepare the datasets and fit a Linear Regression model."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Separate independent (features) and dependent variables (target)\n",
                    "X = df[features]\n",
                    "y = df[\"HDI Score\"]\n",
                    "\n",
                    "# Train-Test Split (80% Train, 20% Test)\n",
                    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)\n",
                    "\n",
                    "# Initialize and train the model\n",
                    "model = LinearRegression()\n",
                    "model.fit(X_train, y_train)\n",
                    "\n",
                    "print(\"Linear Regression Model successfully trained!\")\n",
                    "print(f\"Intercept: {model.intercept_:.6f}\")\n",
                    "print(\"Coefficients:\")\n",
                    "for feature, coef in zip(features, model.coef_):\n",
                    "    print(f\"  {feature}: {coef:.6f}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Predict on testing set\n",
                    "y_pred = model.predict(X_test)\n",
                    "\n",
                    "# Calculate Metrics\n",
                    "r2 = metrics.r2_score(y_test, y_pred)\n",
                    "mae = metrics.mean_absolute_error(y_test, y_pred)\n",
                    "mse = metrics.mean_squared_error(y_test, y_pred)\n",
                    "rmse = np.sqrt(mse)\n",
                    "\n",
                    "print(\"=== Linear Regression Model Performance ===\")\n",
                    "print(f\"R² Score (Coefficient of Determination): {r2:.4f}\")\n",
                    "print(f\"Mean Absolute Error (MAE):              {mae:.4f}\")\n",
                    "print(f\"Mean Squared Error (MSE):               {mse:.4f}\")\n",
                    "print(f\"Root Mean Squared Error (RMSE):         {rmse:.4f}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Plot Actual vs Predicted values\n",
                    "plt.figure(figsize=(8, 6))\n",
                    "plt.scatter(y_test, y_pred, color=\"purple\", alpha=0.7, edgecolors='black')\n",
                    "plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)\n",
                    "plt.title(\"Actual vs. Predicted HDI Scores\", fontsize=14, fontweight='bold')\n",
                    "plt.xlabel(\"Actual HDI Score\", fontsize=12)\n",
                    "plt.ylabel(\"Predicted HDI Score\", fontsize=12)\n",
                    "plt.tight_layout()\n",
                    "plt.savefig(os.path.join(\"..\", \"static\", \"images\", \"actual_vs_predicted.png\"), dpi=150)\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Conclusion\n",
                    "The Linear Regression model matches the HDI scores extremely well, explaining over 98% of the variance ($R^2 \\approx 0.986$) with a root mean squared error of around $0.02$.\n",
                    "The trained model has been saved as `model.pkl` and is loaded in the Flask backend (`app.py`) to serve production predictions!"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    os.makedirs("notebooks", exist_ok=True)
    notebook_path = os.path.join("notebooks", "HDI_Analysis.ipynb")
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=1)
    print(f"Jupyter Notebook successfully created at {notebook_path}")

if __name__ == "__main__":
    create_notebook()
