from turtle import right
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

import flask
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model

import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

app = Flask(__name__)

# we need to redefine our metric function in order 
# to use it when loading the model 
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    keras.backend.get_session().run(tf.local_variables_initializer())
    return auc

# load the model, and pass in the custom metric function
global graph
# graph = tf.get_default_graph()
graph = tf.compat.v1.get_default_graph
# lstm = load_model('lstm.h5', custom_objects={'auc': auc})
lstm = load_model('lstm.h5')
lstm.make_predict_function()

l1=[80.0, 1901423.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1901423.0, 0.0, 1901423.0, 1901423.0, 1901423.0, 1901423.0, 0.0, 1901423.0, 1901423.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 64.0, 0.0, 1.0518438033, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, -1.0, 0.0, 32.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
print(len(l1))
prediction =lstm.predict([l1])
print(prediction)
label = int(np.argmax(prediction, axis=-1))
print(label)
labels={0: 'Benign', 4: 'DDOS attack-HOIC', 1: 'Bot', 8: 'FTP-BruteForce', 10: 'SSH-Bruteforce', 6: 'DoS attacks-GoldenEye', 7: 'DoS attacks-Slowloris', 5: 'DDOS attack-LOIC-UDP', 2: 'Brute Force -Web', 3: 'Brute Force -XSS', 9: 'SQL Injection'}
print(labels[label])

@app.route('/',methods=['GET','POST'])
def index():
    return render_template("index1.html")

# @app.route('/predict',methods=['POST','GET'])
# def predict():
#     if request.method=='GET':
#         return  f"The URL /data is accessed directly. Try going to '/form' to submit form"
#     if request.method=='POST':
        
    
#         f=request.form['csvfile']
       
#         data=[]
#         with open(f) as file:
#             csvfile=csv.reader(file)
#             for row in csvfile:
#                 data.append(*row)
#             data=[float(i) for i in data]
#             data.pop(0)
# #         data=pd.DataFrame(data)
#         print(len(data))
#         prediction =lstm.predict([data])
#         label = int(np.argmax(prediction, axis=-1))
#         print(labels[label])
#         label=(labels[label])
       
        
#         return render_template("prediction.html",prediction=label)

    


@app.route("/show")
def show():
    return render_template("test.html")



@app.route('/graph')
def plot_png():
   fig = Figure()
   axis = fig.add_subplot(1, 1, 1)
   xs = np.random.rand(100)
   ys = np.random.rand(100)
   axis.plot(xs, ys)
   output = io.BytesIO()
   FigureCanvas(fig).print_png(output)
   return Response(output.getvalue(), mimetype='image/png')

@app.route('/data',methods=['POST','GET'])
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
@app.route("/predict", methods=["GET","POST"])
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
            data = data.iloc[1:30]
            data=data.values.tolist()
            data = [[float(s) for s in row] for row in data]

            prediction = lstm.predict(data)
            label = np.argmax(prediction, axis=-1)
            l=list(labels[i] for i in label)
            sns_bar = sns.countplot(l)
          
            fig1 = sns_bar.get_figure()
            fig1.savefig('static/bar.png')
            c=Counter(l)
            x=list(c.values())
            y=list(c.keys())
            print(x)
            print(y)
            plt.clf()

            plt.pie(x,labels=y,autopct='%1.2f%%')
            plt.pie(x,labels=y)

            plt.savefig("static/pie.png")
          
            # plt.show()
            
            
            # plt.savefig('pie.png',bbox_inches="tight",transparent = False,)
            d=pd.DataFrame(l)
            # z=[*zip(data,l)]

            

        return render_template("prediction.html",prediction=d.to_html() )

   
if __name__=="__main__":
    app.run(debug=True)

