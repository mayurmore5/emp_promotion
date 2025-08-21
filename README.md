HEAD
Demo video Link:
https://drive.google.com/file/d/12wJ0mLWnQ-KvLPJEd5q7YTpI4dM0UcU8/view?usp=drivesdk
=======
# Employee Promotion Prediction

An end-to-end Flask web app that predicts whether an employee is eligible for promotion using a trained ML model.

## Project Structure

- `Employee_pred/flask/project.py`: Flask application
- `Employee_pred/flask/templates/`: Jinja2 templates (`home.html`, `predict.html`, `submit.html`, `about.html`, `history.html`, `404.html`)
- `Employee_pred/flask/Static/`: Static assets (CSS, images)
- `Employee_pred/flask/promotion.pkl`: Trained model
- `Employee_pred/Training/Project.ipynb`: Model training notebook
- `Employee_pred/Dataset/emp_promotion.csv`: Raw dataset

## Setup

1) Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Run the Flask app:

```bash
python Employee_pred/flask/project.py
```

The app will be available at `http://127.0.0.1:5000`.

## Features

- Clean, responsive UI with a shared base layout
- Form validation and friendly error messages
- Prediction result page with optional probability estimate
- View recent predictions on the History page (stored in session)
- Health endpoint and JSON API for integrations

## JSON API

Endpoint: `POST /api/predict`

Request body (JSON):

```json
{
  "dept": "Sales & Marketing",
  "not": 2,
  "pyr": 4.5,
  "los": 7,
  "kpi": 1,
  "aw": 0,
  "ats": 78
}
```

Response:

```json
{ "eligible": true, "probability": 0.83 }
```

## Notes

- The model is loaded from `Employee_pred/flask/promotion.pkl`. If you retrain, overwrite this file.
- The session stores the last 20 predictions for the History page. For production, set a secure secret key via environment variables.
170659b (Trained model and made UI clean)
