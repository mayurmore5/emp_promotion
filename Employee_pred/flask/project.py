from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open("promotion.pkl", "rb"))

@app.route('/')
def index():
    return render_template("predict.html")

@app.route('/home')
def home():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/submit')
def submit():
    return render_template("submit.html")

@app.route('/predict')
def predict():
    return render_template("predict.html")

@app.route('/datas', methods=["POST"])
def do():
    d = request.form["dept"]
    
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
    d = department_mapping.get(d, 0)
    
    num_of_training = int(request.form["not"])
    pre_yr_rating = float(request.form["pyr"])
    len_of_service = int(request.form["los"])
    kpi = int(request.form["kpi"])
    award = int(request.form["aw"])
    avg_training_score = float(request.form["ats"])
    
    data = [[d, num_of_training, pre_yr_rating, len_of_service, kpi, award, avg_training_score]]
    
    p = model.predict(data)
    
    if p == 0 :
        text = 'Sorry, you are not eligible for promotion'
    else:
        text = 'Great, you are eligible for promotion'

    return render_template("submit.html", data=text)

if __name__ == "__main__":
    app.run(debug=True)
    
