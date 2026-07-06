import os
import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session

app = Flask(__name__)
app.secret_key = "hdi_predictor_secret_key_12345"

# Load the trained ML model
MODEL_PATH = "model.pkl"
model = None

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("ML Model loaded successfully.")
    else:
        print(f"WARNING: Model file '{MODEL_PATH}' not found. Prediction features will fail until a model is trained.")
except Exception as e:
    print(f"ERROR: Failed to load model. {e}")

# Helper function to categorize HDI score
def get_hdi_category(score):
    score = float(score)
    if score >= 0.800:
        return {
            "name": "Very High HDI",
            "badge_class": "bg-success-subtle text-success border-success",
            "text_color": "text-success",
            "description": "Countries in this tier enjoy high standards of living, advanced healthcare infrastructure, long life expectancies, exceptional education systems, and very strong economies.",
            "recommendation": "Maintain investments in cutting-edge research, tech innovation, sustainable green energy, and social security programs to promote equity and long-term resilience."
        }
    elif score >= 0.700:
        return {
            "name": "High HDI",
            "badge_class": "bg-info-subtle text-info border-info",
            "text_color": "text-info",
            "description": "Countries in this tier show robust progress in economic development and public services, but may face minor inequalities, regional disparities, or transitioning infrastructure.",
            "recommendation": "Focus on strengthening higher education systems, enhancing the coverage of high-quality healthcare in rural sectors, and diversifying the industrial base."
        }
    elif score >= 0.550:
        return {
            "name": "Medium HDI",
            "badge_class": "bg-warning-subtle text-warning border-warning",
            "text_color": "text-warning",
            "description": "Countries in this tier are developing rapidly but continue to struggle with significant gaps in literacy, basic sanitation, healthcare access, and wealth distribution.",
            "recommendation": "Prioritize elementary school completion rates, expand primary healthcare networks, construct resilient public infrastructure, and promote skill development programs."
        }
    else:
        return {
            "name": "Low HDI",
            "badge_class": "bg-danger-subtle text-danger border-danger",
            "text_color": "text-danger",
            "description": "Countries in this tier experience critical deficits across health, education, and national income. High poverty rates, fragile health systems, and limited schooling access are prevalent.",
            "recommendation": "Urgent intervention is needed in maternal-child health, clean water access, subsidized school feeding programs, basic agrarian modernization, and structural economic aid."
        }

# Route: Home Page
@app.route("/")
@app.route("/home")
def home():
    return render_template("index.html", active_page="home")

# Route: Prediction Page Form (GET) and Predict handler (POST)
@app.route("/predict", methods=["GET", "POST"])
def predict():
    global model
    
    if request.method == "POST":
        # Check if model is available
        if model is None:
            # Try reloading it
            if os.path.exists(MODEL_PATH):
                with open(MODEL_PATH, "rb") as f:
                    model = pickle.load(f)
            else:
                flash("Prediction service is currently unavailable: Trained model file not found.", "danger")
                return redirect(url_for("predict"))

        try:
            # Extract form inputs
            life_exp_raw = request.form.get("life_expectancy")
            eys_raw = request.form.get("expected_schooling")
            mys_raw = request.form.get("mean_schooling")
            gni_raw = request.form.get("gni_per_capita")
            
            # 1. Check for empty values
            if not all([life_exp_raw, eys_raw, mys_raw, gni_raw]):
                flash("All input fields are required. Please fill out the form completely.", "warning")
                return redirect(url_for("predict"))
            
            # Convert to float/int
            life_exp = float(life_exp_raw)
            eys = float(eys_raw)
            mys = float(mys_raw)
            gni = float(gni_raw)
            
            # 2. Server-side range validation
            errors = []
            if not (20.0 <= life_exp <= 100.0):
                errors.append("Life Expectancy must be between 20.0 and 100.0 years.")
            if not (0.0 <= eys <= 25.0):
                errors.append("Expected Years of Schooling must be between 0.0 and 25.0 years.")
            if not (0.0 <= mys <= 20.0):
                errors.append("Mean Years of Schooling must be between 0.0 and 20.0 years.")
            if not (100.0 <= gni <= 150000.0):
                errors.append("GNI Per Capita must be between $100.00 and $150,000.00 USD.")
            if mys > eys:
                errors.append("Mean Years of Schooling cannot exceed Expected Years of Schooling.")
                
            if errors:
                for error in errors:
                    flash(error, "danger")
                return render_template("predict.html", 
                                       active_page="predict",
                                       life_expectancy=life_exp_raw,
                                       expected_schooling=eys_raw,
                                       mean_schooling=mys_raw,
                                       gni_per_capita=gni_raw)
            
            # 3. Model Prediction
            import pandas as pd
            features = ["Life Expectancy", "Expected Years of Schooling", "Mean Years of Schooling", "GNI Per Capita"]
            input_df = pd.DataFrame([[life_exp, eys, mys, gni]], columns=features)
            prediction = model.predict(input_df)[0]
            
            # Clip predictions between logical bounds of HDI (0.000 to 1.000)
            prediction = np.clip(prediction, 0.0, 1.0)
            prediction_score = round(float(prediction), 3)
            
            # Get category details
            category_details = get_hdi_category(prediction_score)
            
            # Store in session history (keep last 5 predictions)
            if "history" not in session:
                session["history"] = []
            
            history = session["history"]
            history.insert(0, {
                "life_expectancy": life_exp,
                "expected_schooling": eys,
                "mean_schooling": mys,
                "gni_per_capita": gni,
                "hdi_score": prediction_score,
                "category": category_details["name"]
            })
            # Slice to keep only last 5 items
            session["history"] = history[:5]
            session.modified = True
            
            # Render results page
            return render_template("result.html",
                                   active_page="predict",
                                   life_expectancy=life_exp,
                                   expected_schooling=eys,
                                   mean_schooling=mys,
                                   gni_per_capita=gni,
                                   predicted_score=prediction_score,
                                   category=category_details["name"],
                                   badge_class=category_details["badge_class"],
                                   text_color=category_details["text_color"],
                                   description=category_details["description"],
                                   recommendation=category_details["recommendation"])
            
        except ValueError:
            flash("Invalid input types. Please enter valid numeric values.", "danger")
            return redirect(url_for("predict"))
        except Exception as e:
            flash(f"An unexpected error occurred: {e}", "danger")
            return redirect(url_for("predict"))
            
    # GET request: render the empty form
    return render_template("predict.html", active_page="predict")

# Route: About Page
@app.route("/about")
def about():
    return render_template("about.html", active_page="about")

# Route: Clear Prediction History
@app.route("/clear_history")
def clear_history():
    session.pop("history", None)
    flash("Prediction history cleared successfully.", "success")
    return redirect(url_for("predict"))

# Error Handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html", active_page=""), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template("500.html", active_page=""), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
