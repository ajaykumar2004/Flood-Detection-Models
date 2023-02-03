from flask import Flask,request,jsonify
import joblib
import numpy as np

app = Flask(__name__)


@app.route('/api',methods=['GET'])
def returnPred():
    d = {}
    
    inp  = str(request.args['query'])
    
    w = [[float(x) for x in inp.split(",")]]
    
    
    loaded1=joblib.load('model_tree.joblib')
    # loaded2=joblib.load('model_xgb.joblib') 
    loaded3=joblib.load('model_random_forest.joblib') 
    loaded4=joblib.load('model_logistic_regression.joblib')

    
    d['output_tree'] =str(loaded1.predict(w))
    # d['output_xgb'] =str(loaded2.predict(w))
    d['output_random_forest'] =str(loaded3.predict(w))
    d['output_logistic_regression'] =str(loaded4.predict(w))
    return d

if __name__ =='__main__':
    app.run()
