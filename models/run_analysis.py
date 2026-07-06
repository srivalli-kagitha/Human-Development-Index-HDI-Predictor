import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn import metrics

# Set plotting theme and defaults
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12

def run_eda_pipeline():
    print("=== Initiating Exploratory Data Analysis & Visualization Pipeline ===")
    
    # Define file paths
    dataset_path = os.path.join("dataset", "hdi_dataset.csv")
    output_dir = os.path.join("static", "images")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load dataset
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 2. Categorization for styling
    def get_category(score):
        if score >= 0.800: return "Very High"
        elif score >= 0.700: return "High"
        elif score >= 0.550: return "Medium"
        else: return "Low"
    df["Development Category"] = df["HDI Score"].apply(get_category)
    
    features = ["Life Expectancy", "Expected Years of Schooling", "Mean Years of Schooling", "GNI Per Capita"]
    
    # 3. Create Plots
    
    # Plot 1: Strip Plot
    print("Generating Plot 1: Strip Plot...")
    plt.figure(figsize=(10, 6))
    sns.stripplot(x="Development Category", y="HDI Score", data=df, 
                  order=["Low", "Medium", "High", "Very High"], 
                  palette="viridis", hue="Development Category", size=6, jitter=0.25)
    plt.title("Strip Plot: HDI Score Distribution across Development Categories", fontsize=14, fontweight='bold')
    plt.xlabel("Development Category", fontsize=12)
    plt.ylabel("HDI Score", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "strip_plot.png"), dpi=150)
    plt.close()
    
    # Plot 2: Distribution Plot
    print("Generating Plot 2: Distribution Plot...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df["HDI Score"], kde=True, color="#4A90E2", bins=15, edgecolor='black')
    plt.title("Distribution Plot: Density Profile of HDI Score", fontsize=14, fontweight='bold')
    plt.xlabel("HDI Score", fontsize=12)
    plt.ylabel("Density / Count", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dist_plot.png"), dpi=150)
    plt.close()
    
    # Plot 3: Histograms
    print("Generating Plot 3: Histograms of features...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(df["Life Expectancy"], color="teal", ax=axes[0,0], kde=True, edgecolor='black')
    axes[0,0].set_title("Distribution of Life Expectancy", fontweight='bold')
    axes[0,0].set_xlabel("Life Expectancy (Years)")
    
    sns.histplot(df["Expected Years of Schooling"], color="purple", ax=axes[0,1], kde=True, edgecolor='black')
    axes[0,1].set_title("Distribution of Expected Years of Schooling", fontweight='bold')
    axes[0,1].set_xlabel("Expected Years (Years)")
    
    sns.histplot(df["Mean Years of Schooling"], color="orange", ax=axes[1,0], kde=True, edgecolor='black')
    axes[1,0].set_title("Distribution of Mean Years of Schooling", fontweight='bold')
    axes[1,0].set_xlabel("Mean Years (Years)")
    
    sns.histplot(df["GNI Per Capita"], color="forestgreen", ax=axes[1,1], kde=True, edgecolor='black')
    axes[1,1].set_title("Distribution of GNI Per Capita", fontweight='bold')
    axes[1,1].set_xlabel("GNI Per Capita (USD PPP)")
    
    plt.suptitle("Histograms: Distribution Profiles of Predictor Features", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "histograms.png"), dpi=150)
    plt.close()
    
    # Plot 4: Box Plot
    print("Generating Plot 4: Box Plot...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.boxplot(y="Life Expectancy", x="Development Category", data=df, 
                order=["Low", "Medium", "High", "Very High"], ax=axes[0], palette="Set2")
    axes[0].set_title("Life Expectancy by HDI Tier", fontweight='bold')
    axes[0].set_xlabel("")
    
    sns.boxplot(y="Expected Years of Schooling", x="Development Category", data=df, 
                order=["Low", "Medium", "High", "Very High"], ax=axes[1], palette="Set2")
    axes[1].set_title("Expected Schooling by HDI Tier", fontweight='bold')
    axes[1].set_xlabel("")
    
    sns.boxplot(y="Mean Years of Schooling", x="Development Category", data=df, 
                order=["Low", "Medium", "High", "Very High"], ax=axes[2], palette="Set2")
    axes[2].set_title("Mean Schooling by HDI Tier", fontweight='bold')
    axes[2].set_xlabel("")
    
    plt.suptitle("Box Plots: Feature Distributions grouped by HDI Category", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "box_plots.png"), dpi=150)
    plt.close()
    
    # Plot 5: Scatter Plot
    print("Generating Plot 5: Scatter Plot...")
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df["Life Expectancy"], df["GNI Per Capita"], c=df["HDI Score"], 
                          cmap="plasma", s=70, alpha=0.8, edgecolors='black')
    cbar = plt.colorbar(scatter)
    cbar.set_label("HDI Score")
    plt.title("Scatter Plot: Life Expectancy vs GNI Per Capita (Colormap: HDI Score)", fontsize=14, fontweight='bold')
    plt.xlabel("Life Expectancy (Years)", fontsize=12)
    plt.ylabel("GNI Per Capita (USD PPP)", fontsize=12)
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_plot.png"), dpi=150)
    plt.close()
    
    # Plot 6: Heatmap
    print("Generating Plot 6: Heatmap...")
    plt.figure(figsize=(8, 6))
    correlation_matrix = df[[*features, "HDI Score"]].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".4f", linewidths=0.5, square=True)
    plt.title("Heatmap: Correlation Matrix of Indicators & HDI Score", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap.png"), dpi=150)
    plt.close()
    
    # Plot 7: Pair Plot
    print("Generating Plot 7: Pair Plot...")
    pair_plot = sns.pairplot(df[[*features, "HDI Score", "Development Category"]], 
                             hue="Development Category", hue_order=["Low", "Medium", "High", "Very High"],
                             palette="Set1", diag_kind="kde")
    pair_plot.fig.suptitle("Pair Plot: Multidimensional Relationships Grid", y=1.02, fontsize=15, fontweight='bold')
    pair_plot.savefig(os.path.join(output_dir, "pair_plot.png"), dpi=150)
    plt.close()
    
    # Model Training Plot (Actual vs Predicted)
    print("Generating Actual vs Predicted Plot...")
    X = df[features]
    y = df["HDI Score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color="purple", alpha=0.7, edgecolors='black')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.title("Actual vs. Predicted HDI Scores", fontsize=14, fontweight='bold')
    plt.xlabel("Actual HDI Score", fontsize=12)
    plt.ylabel("Predicted HDI Score", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"), dpi=150)
    plt.close()
    
    print("=== EDA Pipeline Completed Successfully! ===")
    print(f"All images saved in: {output_dir}")

if __name__ == "__main__":
    run_eda_pipeline()
