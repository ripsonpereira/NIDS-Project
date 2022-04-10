from distutils.log import debug
# from turtle import right
from flask import Flask, render_template, request
from matplotlib.transforms import Bbox
import pandas as pd
import csv
import io
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from sklearn.preprocessing import LabelEncoder

# import flask
import pandas as pd
import tensorflow as tf
# from tensorflow import keras

import keras
from keras.models import load_model

import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


application = Flask(__name__)




lstm = load_model('lstm1.h5')
lstm.make_predict_function()


labels={0: 'Benign', 4: 'DDOS attack-HOIC', 1: 'Bot', 8: 'FTP-BruteForce', 10: 'SSH-Bruteforce', 6: 'DoS attacks-GoldenEye', 7: 'DoS attacks-Slowloris', 5: 'DDOS attack-LOIC-UDP', 2: 'Brute Force -Web', 3: 'Brute Force -XSS', 9: 'SQL Injection'}


@application.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html")

#predict for single row
@application.route('/predict_1', methods=['POST', 'GET'])
def predict_1():
    if request.method=='GET':
        return  f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method=='POST':
        
    
        f=request.form['csvfile']
       
        data=[]
        with open(f) as file:
            csvfile=csv.reader(file)
            for row in csvfile:
                data.append(*row)
            data=[float(i) for i in data]
            data.pop(0)
        data=pd.DataFrame(data)
        
        prediction =lstm.predict([data])
        label = int(np.argmax(prediction, axis=-1))
        
        label=(labels[label])
       
        
        return render_template("prediction.html",prediction=label)

    


@application.route("/about")
def about():
    return render_template("about.html")





@application.route('/data', methods=['POST', 'GET'])
def data():
    if request.method=='GET':
        return  f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method=='POST':

    
        f=request.form['csvfile']
        
       
        data=[]
        with open(f) as file:
            csvfile=csv.reader(file)
            for row in csvfile:
                data.append(row)
        data=pd.DataFrame(data)
        
       
        
        return render_template("data.html",data=data.to_html())





# predict for multiple rows
@application.route("/predict", methods=["GET", "POST"])
def predict():
        if request.method=='GET':
            return  f"The URL /data is accessed directly. Try going to '/form' to submit form"
        if request.method=='POST':

        
            f=request.form['csvfile']
            
            
        
            data=[]
            with open(f) as file:
                csvfile=csv.reader(file)
                for row in csvfile:
                    data.append(row)
            data=pd.DataFrame(data)
            X=data
            data = data.iloc[1:31]
            data=data.values.tolist()
            data = [[float(s) for s in row] for row in data]

            prediction = lstm.predict(data)
            label = np.argmax(prediction, axis=-1)
            l=list(labels[i] for i in label)
            sns_bar = sns.countplot(l)
            sns_bar.set_xticklabels(sns_bar.get_xticklabels(), rotation=40, ha="right")
            plt.tight_layout()
          
            fig1 = sns_bar.get_figure()
            fig1.savefig('static/images/bar.png',transparent=True)
            c=Counter(l)
            x=list(c.values())
            y=list(c.keys())
            # print(x)
            # print(y)
            plt.clf()

            plt.pie(x,labels=y,autopct='%1.2f%%')
            plt.pie(x,labels=y)

            plt.savefig("static/images/pie.png",transparent=True)
            d=pd.DataFrame(l)
            # print(dict(c))
            count=dict(c)
            

        return render_template("prediction.html", data=X.to_html(),prediction=d.to_html(),c=count )

   
if __name__=="__main__":
    debug=True
    application.run()

