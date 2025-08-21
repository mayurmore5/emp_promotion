from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import pickle
import os
from gemini_analysis import analyze_employee_promotion, generate_improvement_plan

app = Flask(__name__, static_folder='Static', template_folder='templates')
app.secret_key = "change-this-secret-key"

# Load the model
model_path = os.path.join(os.path.dirname(__file__), "promotion.pkl")
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template("home.html")

@app.route('/home')
def home():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/submit')
def submit():
    # Legacy route to display the last prediction if available
    last = session.get("history", [])[-1] if session.get("history") else None
    return render_template("submit.html", data=last["message"] if last else None, probability=last.get("probability") if last else None)

@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    # POST: handle form submission
    form = request.form
    department = form.get("dept", "")

    department_mapping = {
        "Sales & Marketing": 7,
        "Operations": 4,
        "Technology": 8,
        "Analytics": 0,
        "R&D": 6,
        "Procurement": 5,
        "Finance": 1,
        "HR": 2,
        "Legal": 3
    }

    try:
        d = department_mapping.get(department, 0)
        num_of_training = int(form.get("not", "").strip())
        pre_yr_rating = float(form.get("pyr", "").strip())
        len_of_service = int(form.get("los", "").strip())
        kpi = int(form.get("kpi", "").strip())
        award = int(form.get("aw", "").strip())
        avg_training_score = float(form.get("ats", "").strip())
    except (ValueError, AttributeError):
        flash("Please enter valid numeric values for all fields.")
        return redirect(url_for("predict"))

    # Basic range validation
    errors = []
    if num_of_training < 0:
        errors.append("Number of trainings must be 0 or more.")
    if not (0.0 <= pre_yr_rating <= 5.0):
        errors.append("Previous year rating must be between 0 and 5.")
    if len_of_service < 0:
        errors.append("Length of service must be 0 or more.")
    if kpi not in (0, 1):
        errors.append("KPIs met must be Yes (1) or No (0).")
    if award not in (0, 1):
        errors.append("Awards won must be Yes (1) or No (0).")
    if not (0.0 <= avg_training_score <= 100.0):
        errors.append("Average training score must be between 0 and 100.")

    if errors:
        for err in errors:
            flash(err)
        return redirect(url_for("predict"))

    data = [[d, num_of_training, pre_yr_rating, len_of_service, kpi, award, avg_training_score]]

    p = model.predict(data)

    probability = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(data)
            # Assuming class 1 is "eligible"
            probability = float(proba[0][1])
    except Exception:
        probability = None

    pred_value = int(p[0]) if hasattr(p, "__iter__") else int(p)
    if pred_value == 0:
        text = 'Sorry, you are not eligible for promotion'
    else:
        text = 'Great, you are eligible for promotion'

    # Prepare employee data for Gemini analysis
    employee_data = {
        'dept': department,
        'not': num_of_training,
        'pyr': pre_yr_rating,
        'los': len_of_service,
        'kpi': kpi,
        'aw': award,
        'ats': avg_training_score
    }

    # Generate Gemini AI analysis
    try:
        gemini_analysis = analyze_employee_promotion(employee_data, text, probability or 0.5)
        improvement_plan = generate_improvement_plan(employee_data, text)
    except Exception as e:
        gemini_analysis = None
        improvement_plan = None
        print(f"Gemini analysis failed: {e}")

    # Store in session history with analysis
    history = session.get("history", [])
    history.append({
        "department": department,
        "inputs": {
            "num_of_training": num_of_training,
            "pre_yr_rating": pre_yr_rating,
            "len_of_service": len_of_service,
            "kpi": kpi,
            "award": award,
            "avg_training_score": avg_training_score
        },
        "prediction": pred_value,
        "probability": probability,
        "message": text,
        "gemini_analysis": gemini_analysis,
        "improvement_plan": improvement_plan
    })
    session["history"] = history[-20:]  # keep last 20

    return render_template("submit.html", 
                         data=text, 
                         probability=probability,
                         gemini_analysis=gemini_analysis,
                         improvement_plan=improvement_plan)

@app.route('/datas', methods=["POST"])
def do():
    # Backward compatibility for existing form action; delegate to /predict
    return predict()

@app.route('/api/predict', methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    department = payload.get("dept", "")

    department_mapping = {
        "Sales & Marketing": 7,
        "Operations": 4,
        "Technology": 8,
        "Analytics": 0,
        "R&D": 6,
        "Procurement": 5,
        "Finance": 1,
        "HR": 2,
        "Legal": 3
    }

    try:
        d = department_mapping.get(department, 0)
        num_of_training = int(payload.get("not", 0))
        pre_yr_rating = float(payload.get("pyr", 0))
        len_of_service = int(payload.get("los", 0))
        kpi = int(payload.get("kpi", 0))
        award = int(payload.get("aw", 0))
        avg_training_score = float(payload.get("ats", 0))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid input types"}), 400

    data = [[d, num_of_training, pre_yr_rating, len_of_service, kpi, award, avg_training_score]]
    p = model.predict(data)

    probability = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(data)
            probability = float(proba[0][1])
    except Exception:
        probability = None

    eligible = int(p[0]) if hasattr(p, "__iter__") else int(p)
    return jsonify({
        "eligible": bool(eligible),
        "probability": probability
    })

@app.route('/history')
def history():
    return render_template("history.html", history=session.get("history", []))

@app.route('/health')
def health():
    return "ok", 200

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404

if __name__ == "__main__":
    app.run(debug=True)
    
