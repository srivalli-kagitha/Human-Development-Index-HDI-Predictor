import os
import numpy as np
import pandas as pd

# Define directories
os.makedirs("dataset", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Representative real countries from the 4 HDI categories
real_countries = [
    # Very High HDI
    {"Country": "Switzerland", "Life Expectancy": 84.0, "Expected Years of Schooling": 16.5, "Mean Years of Schooling": 13.9, "GNI Per Capita": 66933},
    {"Country": "Norway", "Life Expectancy": 83.2, "Expected Years of Schooling": 18.2, "Mean Years of Schooling": 13.0, "GNI Per Capita": 64660},
    {"Country": "Iceland", "Life Expectancy": 82.7, "Expected Years of Schooling": 19.2, "Mean Years of Schooling": 12.8, "GNI Per Capita": 55782},
    {"Country": "Germany", "Life Expectancy": 80.6, "Expected Years of Schooling": 17.0, "Mean Years of Schooling": 14.1, "GNI Per Capita": 54538},
    {"Country": "United States", "Life Expectancy": 77.2, "Expected Years of Schooling": 16.3, "Mean Years of Schooling": 13.7, "GNI Per Capita": 64765},
    {"Country": "United Kingdom", "Life Expectancy": 80.7, "Expected Years of Schooling": 17.3, "Mean Years of Schooling": 13.4, "GNI Per Capita": 45225},
    {"Country": "Japan", "Life Expectancy": 84.8, "Expected Years of Schooling": 15.2, "Mean Years of Schooling": 13.4, "GNI Per Capita": 42274},
    {"Country": "Canada", "Life Expectancy": 81.2, "Expected Years of Schooling": 16.4, "Mean Years of Schooling": 13.8, "GNI Per Capita": 48521},
    {"Country": "Australia", "Life Expectancy": 83.2, "Expected Years of Schooling": 21.1, "Mean Years of Schooling": 12.7, "GNI Per Capita": 49238},
    {"Country": "Singapore", "Life Expectancy": 82.8, "Expected Years of Schooling": 16.5, "Mean Years of Schooling": 11.9, "GNI Per Capita": 90919},
    
    # High HDI
    {"Country": "Albania", "Life Expectancy": 76.5, "Expected Years of Schooling": 14.4, "Mean Years of Schooling": 11.3, "GNI Per Capita": 14136},
    {"Country": "Brazil", "Life Expectancy": 72.8, "Expected Years of Schooling": 15.6, "Mean Years of Schooling": 8.1, "GNI Per Capita": 14370},
    {"Country": "China", "Life Expectancy": 78.2, "Expected Years of Schooling": 14.2, "Mean Years of Schooling": 7.6, "GNI Per Capita": 17504},
    {"Country": "Mexico", "Life Expectancy": 70.2, "Expected Years of Schooling": 14.9, "Mean Years of Schooling": 9.2, "GNI Per Capita": 17896},
    {"Country": "Sri Lanka", "Life Expectancy": 76.4, "Expected Years of Schooling": 14.1, "Mean Years of Schooling": 10.8, "GNI Per Capita": 12578},
    {"Country": "Ukraine", "Life Expectancy": 71.6, "Expected Years of Schooling": 15.0, "Mean Years of Schooling": 11.1, "GNI Per Capita": 13256},
    {"Country": "Algeria", "Life Expectancy": 76.4, "Expected Years of Schooling": 14.6, "Mean Years of Schooling": 8.0, "GNI Per Capita": 10800},
    {"Country": "Thailand", "Life Expectancy": 78.7, "Expected Years of Schooling": 15.6, "Mean Years of Schooling": 8.7, "GNI Per Capita": 17030},
    {"Country": "Colombia", "Life Expectancy": 72.8, "Expected Years of Schooling": 14.4, "Mean Years of Schooling": 8.9, "GNI Per Capita": 14324},
    
    # Medium HDI
    {"Country": "India", "Life Expectancy": 67.2, "Expected Years of Schooling": 11.9, "Mean Years of Schooling": 6.7, "GNI Per Capita": 6590},
    {"Country": "Bangladesh", "Life Expectancy": 72.4, "Expected Years of Schooling": 12.4, "Mean Years of Schooling": 6.2, "GNI Per Capita": 5472},
    {"Country": "Egypt", "Life Expectancy": 70.2, "Expected Years of Schooling": 13.8, "Mean Years of Schooling": 7.4, "GNI Per Capita": 11757},
    {"Country": "South Africa", "Life Expectancy": 62.3, "Expected Years of Schooling": 13.6, "Mean Years of Schooling": 10.2, "GNI Per Capita": 12948},
    {"Country": "Indonesia", "Life Expectancy": 67.6, "Expected Years of Schooling": 13.7, "Mean Years of Schooling": 8.6, "GNI Per Capita": 11466},
    {"Country": "Iraq", "Life Expectancy": 70.4, "Expected Years of Schooling": 12.1, "Mean Years of Schooling": 8.0, "GNI Per Capita": 9977},
    {"Country": "Morocco", "Life Expectancy": 74.0, "Expected Years of Schooling": 14.2, "Mean Years of Schooling": 5.9, "GNI Per Capita": 7303},
    {"Country": "Philippines", "Life Expectancy": 69.3, "Expected Years of Schooling": 13.1, "Mean Years of Schooling": 9.0, "GNI Per Capita": 8920},
    {"Country": "Vietnam", "Life Expectancy": 74.3, "Expected Years of Schooling": 13.7, "Mean Years of Schooling": 8.4, "GNI Per Capita": 7867},
    
    # Low HDI
    {"Country": "Pakistan", "Life Expectancy": 66.1, "Expected Years of Schooling": 8.7, "Mean Years of Schooling": 4.5, "GNI Per Capita": 4624},
    {"Country": "Nepal", "Life Expectancy": 69.0, "Expected Years of Schooling": 12.2, "Mean Years of Schooling": 5.1, "GNI Per Capita": 3877},
    {"Country": "Kenya", "Life Expectancy": 61.4, "Expected Years of Schooling": 11.3, "Mean Years of Schooling": 6.7, "GNI Per Capita": 4474},
    {"Country": "Nigeria", "Life Expectancy": 52.7, "Expected Years of Schooling": 10.1, "Mean Years of Schooling": 6.2, "GNI Per Capita": 4790},
    {"Country": "Yemen", "Life Expectancy": 63.8, "Expected Years of Schooling": 9.1, "Mean Years of Schooling": 3.2, "GNI Per Capita": 1914},
    {"Country": "Chad", "Life Expectancy": 52.5, "Expected Years of Schooling": 8.0, "Mean Years of Schooling": 2.6, "GNI Per Capita": 1364},
    {"Country": "Niger", "Life Expectancy": 61.6, "Expected Years of Schooling": 7.0, "Mean Years of Schooling": 2.1, "GNI Per Capita": 1240},
    {"Country": "Central African Republic", "Life Expectancy": 53.9, "Expected Years of Schooling": 8.0, "Mean Years of Schooling": 4.3, "GNI Per Capita": 966},
    {"Country": "Mali", "Life Expectancy": 58.9, "Expected Years of Schooling": 7.4, "Mean Years of Schooling": 2.3, "GNI Per Capita": 2133},
    {"Country": "Burundi", "Life Expectancy": 61.7, "Expected Years of Schooling": 11.1, "Mean Years of Schooling": 3.1, "GNI Per Capita": 732},
]

df_real = pd.DataFrame(real_countries)

# Generate synthetic rows to reach ~160 countries
np.random.seed(42)
num_synthetic = 130
synthetic_rows = []

# List of typical country names/placeholders
countries_list = [
    "Sweden", "Netherlands", "Denmark", "Finland", "New Zealand", "Belgium", "Ireland", "Luxembourg",
    "Austria", "Israel", "Slovenia", "United Arab Emirates", "Spain", "France", "South Korea", "Italy",
    "Estonia", "Czech Republic", "Greece", "Poland", "Saudi Arabia", "Lithuania", "Bahrain", "Portugal",
    "Latvia", "Croatia", "Andorra", "Chile", "Qatar", "Slovakia", "Hungary", "Argentina", "Turkey",
    "Montenegro", "Kuwait", "Russia", "Romania", "Oman", "Bahamas", "Kazakhstan", "Uruguay", "Belarus",
    "Panama", "Malaysia", "Georgia", "Costa Rica", "Serbia", "Mauritius", "Seychelles", "Bulgaria",
    "Iran", "Trinidad and Tobago", "Peru", "Azerbaijan", "Ecuador", "Maldives", "Mongolia", "Dominican Republic",
    "Gabon", "Paraguay", "South Africa", "Tunisia", "Suriname", "Botswana", "Uzbekistan", "Dominica",
    "Libya", "Turkmenistan", "Guyana", "Armenia", "Kyrgyzstan", "Namibia", "El Salvador", "Tajikistan",
    "Guatemala", "Honduras", "India", "Micronesia", "Vanuatu", "Timor-Leste", "Eswatini", "Congo",
    "Zambia", "Ghana", "Cameroon", "Syria", "Zimbabwe", "Moldova", "Myanmar", "Cambodia",
    "Angola", "Tanzania", "Madagascar", "Mauritania", "Lesotho", "Rwanda", "Uganda", "Benin",
    "Sudan", "Togo", "Senegal", "Afghanistan", "Eritrea", "Gambia", "Ethiopia", "Malawi",
    "Guinea", "Yemen", "Guinea-Bissau", "Mozambique", "Sierra Leone", "Burkina Faso", "Liberia", "Somalia",
    "South Sudan", "Sudan", "Djibouti", "Comoros", "Solomon Islands", "Vanuatu", "Papua New Guinea", "Laos"
]

# Ensure uniqueness of country names
all_existing_names = set(df_real["Country"])
unique_synth_names = []
for name in countries_list:
    if name not in all_existing_names and name not in unique_synth_names:
        unique_synth_names.append(name)
    if len(unique_synth_names) == num_synthetic:
        break

# If we run out of names, use generic names
while len(unique_synth_names) < num_synthetic:
    unique_synth_names.append(f"State_{len(unique_synth_names)+1}")

# Determine generation parameter bounds per category to keep it realistic
# 0: Very High, 1: High, 2: Medium, 3: Low
categories = np.random.choice([0, 1, 2, 3], size=num_synthetic, p=[0.25, 0.25, 0.25, 0.25])

for i, cat in enumerate(categories):
    c_name = unique_synth_names[i]
    if cat == 0:  # Very High
        life_exp = np.random.uniform(77.0, 85.5)
        eys = np.random.uniform(15.0, 20.0)
        mys = np.random.uniform(11.5, 14.5)
        gni = np.random.uniform(35000, 85000)
    elif cat == 1:  # High
        life_exp = np.random.uniform(70.0, 78.0)
        eys = np.random.uniform(13.0, 16.0)
        mys = np.random.uniform(8.0, 11.5)
        gni = np.random.uniform(10000, 22000)
    elif cat == 2:  # Medium
        life_exp = np.random.uniform(62.0, 74.0)
        eys = np.random.uniform(11.0, 14.0)
        mys = np.random.uniform(5.5, 9.0)
        gni = np.random.uniform(3000, 12000)
    else:  # Low
        life_exp = np.random.uniform(50.0, 64.0)
        eys = np.random.uniform(6.5, 11.5)
        mys = np.random.uniform(1.5, 6.0)
        gni = np.random.uniform(600, 4500)
        
    synthetic_rows.append({
        "Country": c_name,
        "Life Expectancy": round(life_exp, 1),
        "Expected Years of Schooling": round(eys, 1),
        "Mean Years of Schooling": round(mys, 1),
        "GNI Per Capita": int(gni)
    })

df_synth = pd.DataFrame(synthetic_rows)
df_all = pd.concat([df_real, df_synth], ignore_index=True)

# Official UNDP Human Development Index (HDI) Calculation
# Health Index (LEI) = (LE - 20) / (85 - 20)
# Expected Years of Schooling Index (EYSI) = EYS / 18
# Mean Years of Schooling Index (MYSI) = MYS / 15
# Education Index (EI) = (EYSI + MYSI) / 2
# Income Index (II) = (ln(GNI) - ln(100)) / (ln(75000) - ln(100))
# HDI = (LEI * EI * II) ** (1/3)

def calculate_hdi(row):
    le = row["Life Expectancy"]
    eys = row["Expected Years of Schooling"]
    mys = row["Mean Years of Schooling"]
    gni = row["GNI Per Capita"]
    
    # Indices calculation (with clipping)
    lei = np.clip((le - 20) / 65, 0.0, 1.0)
    
    eysi = np.clip(eys / 18, 0.0, 1.0)
    mysi = np.clip(mys / 15, 0.0, 1.0)
    ei = np.clip((eysi + mysi) / 2, 0.0, 1.0)
    
    # Clip GNI at minimum 100 and maximum 75000 for standard calculation bounds
    gni_val = np.clip(gni, 100, 75000)
    ii = np.clip((np.log(gni_val) - np.log(100)) / (np.log(75000) - np.log(100)), 0.0, 1.0)
    
    hdi = (lei * ei * ii) ** (1/3)
    
    # Add a slight realistic random noise, keeping HDI within 0 and 1
    noise = np.random.normal(0, 0.005)
    hdi = np.clip(hdi + noise, 0.0, 1.0)
    
    return round(hdi, 3)

df_all["HDI Score"] = df_all.apply(calculate_hdi, axis=1)

# Drop any duplicate country rows
df_all = df_all.drop_duplicates(subset=["Country"])

# Save dataset
output_path = os.path.join("dataset", "hdi_dataset.csv")
df_all.to_csv(output_path, index=False)
print(f"Dataset saved successfully with {len(df_all)} rows to: {output_path}")
