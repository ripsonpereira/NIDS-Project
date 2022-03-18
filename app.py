from flask import Flask, render_template, request
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
prediction =lstm.predict(l1)
print(prediction)

@app.route('/',methods=['GET','POST'])
def index():
    return render_template("index1.html")

# @app.route('/predict',methods=['POST','GET'])
# def predict():
#     if request.method=='GET':
#         return  f"The URL /data is accessed directly. Try going to '/form' to submit form"
#     if request.method=='POST':
#         # print("kjnk")
    
#         f=request.form['csvfile']
       
#         data=[]
#         with open(f) as file:
#             csvfile=csv.reader(file)
#             for row in csvfile:
#                 data.append(row)
#         data=pd.DataFrame(data)
#         prediction =lstm.predict(data)
#         labels = np.argmax(prediction, axis=-1)
#         label_encoder = LabelEncoder()
#         labels=label_encoder.inverse_transform(labels)    
#         print(labels)

       
        
#         return render_template("data.html",data=data.to_html())

    


@app.route("/show")
def show():
    return render_template("test.html")

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

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
        # print("kjnk")
    
        f=request.form['csvfile']
       
        data=[]
        with open(f) as file:
            csvfile=csv.reader(file)
            for row in csvfile:
                data.append(row)
        data=pd.DataFrame(data)
        
       
        
        return render_template("data.html",data=data.to_html())





# define a predict function as an endpoint 
# @app.route("/predict", methods=["GET","POST"])
# def predict():
#     data = {"success": False}

#     params = flask.request.json
#     if (params == None):
#         params = flask.request.args

#     # if parameters are found, return a prediction
#     if (params != None):
#         x=pd.DataFrame.from_dict(params, orient='index').transpose()
#         with graph.as_default():
#             data["prediction"] = str(model.predict(x)[0][0])
#             data["success"] = True

#     # return a response in json format 
#     return flask.jsonify(data)    


   
if __name__=="__main__":
    app.run(debug=True)

