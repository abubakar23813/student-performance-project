import pickle
from flask import Flask,request,render_template

print(__name__)

import pandas as pd 
import numpy as np 


from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application

def to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
    


# Route for home page
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    try:
        if request.method == "GET":
            return render_template("home.html")

        data = CustomData(
            gender=str(request.form.get("gender")),
            race_ethnicity=to_int(request.form.get("race_ethnicity")),
            parental_level_of_education=str(request.form.get("parental_level_of_education")),
            lunch=str(request.form.get('lunch')),
            test_preparation_course=str(request.form.get("test_preparation_course")),
            reading_score=to_int(request.form.get("reading_score")),
            writing_score=to_int(request.form.get("writing_score"))
        )

        pred_df = data.get_data_as_data_frame()
        print("DATA:\n", pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template('home.html', results=results[0])

    except Exception as e:
        print("FULL ERROR:", e)
        return f"ERROR: {str(e)}"
    
    
    """route for about page"""
@app.route("/about")
def about():
    return render_template("about.html")

        
    

if __name__=="__main__":
    print("running app")
    app.run(port=5000,debug=True,use_reloader=False)   