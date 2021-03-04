# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 17:06:01 2019

@author: prithvi
"""

from flask import Flask, request , render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('logreg.pkl', 'rb'))

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
        data1 = request.form['a']
        data2 = request.form['b']
        data3 = request.form['c']
        arr = np.array([[data1, data2, data3]])
        pred = model.predict(arr)
        return render_template('after.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)