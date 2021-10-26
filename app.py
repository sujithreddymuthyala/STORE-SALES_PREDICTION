from flask import Flask, jsonify, render_template, request
import joblib
import os
import numpy
import pickle
from sklearn.ensemble import RandomForestRegressor


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("home.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':

        item_weight = float(request.form['item_weight'])
        item_fat_content = float(request.form['item_fat_content'])
        item_visibility = float(request.form['item_visibility'])
        item_type= float(request.form['item_type'])
        item_mrp = float(request.form['item_mrp'])
        outlet_establishment_year = float(request.form['outlet_establishment_year'])
        outlet_size = float(request.form['outlet_size'])
        outlet_location_type = float(request.form['outlet_location_type'])
        outlet_type = float(request.form['outlet_type'])

        data=[[item_weight, item_fat_content, item_visibility, item_type, item_mrp,
                  outlet_establishment_year , outlet_size, outlet_location_type, outlet_type]]

        rf = pickle.load(open('sales_predict.pkl', 'rb'))

        prediction = rf.predict(data)
    return render_template('home.html', prediction=prediction)




if __name__ == "__main__":
    app.run(debug=True, port=9457)
