# Human Development Index (HDI) Predictor

A complete Machine Learning web application that predicts the **Human Development Index (HDI) Score** of a region based on developmental indicators. This project is built using Python, Flask, Scikit-learn, HTML5, CSS3, JavaScript, and Bootstrap 5.

---

## 🌟 Features

- **High-Accuracy ML Model**: Powered by a Scikit-Learn `LinearRegression` model ($R^2 \approx 98.63\%$) trained on realistic international developmental indicators.
- **Interactive UI**: Fully responsive dashboard utilizing **Bootstrap 5**, **Google Fonts (Outfit)**, and **Bootstrap Icons**.
- **Synchronized Controls**: Range sliders synchronized in real-time with numeric input boxes for intuitive simulation.
- **Persistent Dark Mode**: User-controlled Light/Dark mode switcher that persists selections via HTML5 LocalStorage.
- **Session-based Simulation History**: Displays a history card showing the last five simulated results during the browsing session.
- **Report Export (PDF)**: Allows users to download clean, optimized, printable PDF reports of their prediction score.
- **Comprehensive EDA**: Includes a Jupyter Notebook detailing dataset cleaning, descriptive analytics, and 7 visualization configurations.
- **Modular Pipeline**: Clean separation of dataset generation, training models, and web hosting.
- **Custom Error Interfaces**: Custom styled pages for 404 (Not Found) and 500 (Internal Server Error) exceptions.

---

## 🛠️ Technology Stack

- **Backend**: Python 3, Flask
- **Machine Learning**: Scikit-learn, NumPy, SciPy
- **Data Analysis & Visualization**: Pandas, Matplotlib, Seaborn
- **Frontend**: HTML5, CSS3 (Vanilla CSS variables), JavaScript, Bootstrap 5 (CSS & JS CDN)
- **Serialization**: Pickle

---

## 📊 Dataset Structure

The dataset contains simulated metrics for **168 countries** mirroring real-world UNDP ratios:

1. **Country**: Name of the nation.
2. **Life Expectancy**: Average lifetime at birth (Logical range: 20 to 100 years).
3. **Expected Years of Schooling**: Number of years of schooling a child is expected to receive (Logical range: 0 to 25 years).
4. **Mean Years of Schooling**: Average education years completed by adults $\ge 25$ years (Logical range: 0 to 20 years).
5. **GNI Per Capita**: Gross National Income per capita in USD PPP (Logical range: $100 to $150,000).
6. **HDI Score**: The target index score calculated via geometric means of health, education, and wealth dimensions.

---

## 📂 Project Directory Structure

```
HDI-Predictor/
│
├── app.py                      # Core Flask web server & routes
├── model.pkl                   # Serialized Scikit-learn Linear Regression model
├── requirements.txt            # Python library dependencies
├── README.md                   # Project documentation
│
├── dataset/
│   └── hdi_dataset.csv         # Synthesized development records
│
├── notebooks/
│   └── HDI_Analysis.ipynb      # EDA, visualizations, and training notebook
│
├── static/
│   ├── css/
│   │   └── style.css           # UI design tokens, animations, light/dark themes
│   ├── js/
│   │   └── main.js            # Input synchronizer, validators, printing scripts
│   └── images/                 # Jupyter-exported visualization PNGs
│
├── templates/
│   ├── base.html               # Main layout structure & navigation wrapper
│   ├── index.html              # Homepage with objectives & pillars
│   ├── predict.html            # Simulator input form & history tables
│   ├── result.html             # Result gauge, interpretations & printable reports
│   ├── about.html              # Process timeline & technical details
│   ├── 404.html                # Not Found error page
│   └── 500.html                # Server Exception error page
│
└── models/
    ├── generate_data.py        # Procedural script creating dataset
    ├── create_notebook.py      # Code creating the Jupyter Notebook
    └── train_model.py          # Machine learning model pipeline
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Environment Setup
Clone the repository (or extract the folder), open your terminal in the project root, and install the libraries:
```bash
# Verify Python installation
python --version

# Install dependencies
pip install -r requirements.txt
```

### 3. Initialize Dataset & Model (Optional)
The dataset and model are pre-packaged. If you wish to regenerate the dataset and retrain the regression model, run:
```bash
# 1. Generate the CSV dataset
python models/generate_data.py

# 2. Retrain the ML model & save model.pkl
python models/train_model.py

# 3. Create the Jupyter Notebook file
python models/create_notebook.py
```

### 4. Run the Flask Web Application
Launch the Flask development server:
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

---

## 📉 Machine Learning Metrics

- **$R^2$ Score**: `0.9863` (The linear model captures over 98% of the variance in target scores)
- **Mean Absolute Error (MAE)**: `0.0159`
- **Mean Squared Error (MSE)**: `0.0004`
- **Root Mean Squared Error (RMSE)**: `0.0198`

---

## 🔮 Future Enhancements

1. **Polynomial / Multi-Regression Testing**: Compare linear regression metrics with Random Forests or Polynomial regressions to approximate the logarithmic curves of UNDP indices more closely.
2. **Interactive Charting**: Integrate dynamic Javascript charts (such as Chart.js or D3.js) on the results page to show index profiles visually.
3. **Database Integration**: Connect an SQLite backend database to log simulations globally rather than in session storage.
