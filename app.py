from flask import Flask,render_template,url_for,request
import pickle, gzip
#from tensorflow import keras
#from tensorflow.keras.preprocessing.sequence import pad_sequences
import contractions
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import time

from multiprocessing import Process, Queue
import sys

var_details = {}
with open('./model/var_details.pkl', 'rb') as f:
    var_details = pickle.load(f)

data_dict = {}
with open('./model/data_dict.pkl', 'rb') as f:
    data_dict = pickle.load(f)


online_order = var_details['online_order'][1]
location = var_details['location'][1]
rest_type = var_details['rest_type'][1]

#model = None
#queue1 = Queue()
#temp = 0
#print('asda ' + str(temp))
print('Model training started!')    
start = time.time()              
model = RandomForestRegressor(random_state=1, n_estimators=1500)
X = data_dict['X']
y = data_dict['y']
model.fit(X,y)
end = time.time()
print('Completed!')
print(end - start)

#with gzip.open('./model/model_compressed.pkl', 'rb') as f:
#    p = pickle.Unpickler(f)
#    model = p.load()


app = Flask(__name__)
def preprocess_model(t,q):
    #global model
    #print(model)
    print('started')
    model = RandomForestRegressor(random_state=1, n_estimators=1500)
    X = data_dict['X']
    y = data_dict['y']
    model.fit(X,y)
    #print(model)
    #temp_dict = {}
    #temp_dict['model'] = model
    print('Completed')
    q.put(model)
    print(q.get())
    print(q.get())
    #return model

def get_output(X, q):
    if not q.empty():
        model = q.get()
        #model =temp_dict['model'] #q.get()
        y_pred = model.predict(X)[0]
        print('prediction completed')

    else:
        print('queue empty')
        y_pred = 1.0
    
    #print(y_pred)
    return y_pred

@app.route('/')
def home():
    #location = var_details['location'][1]
    #print(temp)
    return render_template('home.html',location = location, online_order = online_order, rest_type = rest_type)

@app.route('/predict', methods=['POST'])
def predict():
    #global model
    #print(model)
    if request.method == 'POST':
        #print(request.form) 
        votes = request.form['votes_val' ]
        loc = request.form['loc']
        cost = request.form['cost']
        rest = request.form['rest']
        onl_ord = request.form['onl_ord']

        #print(loc)
        loc_idx = list(location).index(loc)
        rest_idx = list(rest_type).index(rest)
        onl_ord_idx = list(online_order).index(onl_ord)


        X = np.array([int(votes),loc_idx,float(cost),rest_idx,onl_ord_idx]).reshape(1,-1)
        #y_pred = 0
        #if not queue1.empty():
        #    print(queue1.get())
        #    model = queue1.get()
        #    y_pred = model.predict(X)[0]
        #    y_pred = float(str(y_pred)[:3])

        #y_pred = get_output(X, queue1) #model.predict(X)[0]
        y_pred = model.predict(X)[0]
        y_pred = float(str(y_pred)[:3])
        #print(y_pred) 
        #get_output(queue1)             
        #y_pred = 1.0
        #['votes','location','approx_cost(for two people)','rest_type','online_order']

        
    return render_template('result.html',pred = y_pred, votes = votes, loc = loc, cost = cost, rest = rest, onl_ord = onl_ord, loc_idx = loc_idx)


if __name__ == '__main__':
    #temp = 0
    #p1 = Process(target = preprocess_model, args= (temp,queue1))
    #p1.start()
    #print('Completed!')
    app.run(debug=True)