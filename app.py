
from flask import Flask,request,jsonify
import pandas as pd
import joblib

app = Flask(__name__)



@app.route('/api',methods=['GET'])
def returnPred():
    q = str(request.args['query'])
    d = {}
    if(q[0]=="0"):
        input  = q[:-2]
        answer = [[float(i) for i in input.split(',')]]
        model = joblib.load('model_logistic_regression.joblib')
    
        
        
        if(q[-1]=='1'):
            d['a']=1
            model = joblib.load('model_random_forest.joblib')
        if(q[-1]=='2'):
            d['a']=2
            model = joblib.load('model_xgb.joblib')
        if(q[-1]=='3'):
            d['a']=3
            model = joblib.load('model_logistic_regression.joblib')
        if(q[-1]=='4'):
            d['a']=4
            model = joblib.load('model_tree.joblib')
        
        out = str(model.predict(answer)[0])
        if(out=="1" or out=="YES" ):
            out ="Flood is likely to occur"
        else:
            out="Flood Unlikely"
        
        d['output'] =out
    elif(q[0]=='1'):
        input  = q[1:]
        if(1900<int(input)<2019):
            flag=1
        else:
            flag = 0
            
        df = pd.read_csv('kerala.csv')
        for i in range(2,14):
            d[df.columns[i]] =   df.iloc[int(input)-1901,i] if(flag) else 0
        d['flood'] = df.iloc[int(input)-1901,15] if(flag) else 0        
    return d
if __name__ == "__main__":
    app.run()
