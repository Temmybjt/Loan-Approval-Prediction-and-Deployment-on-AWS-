# import Flask and jsonify
from flask import Flask, jsonify, request

# import Resource, Api and reqparser
from flask_restful import Resource, Api, reqparse
import pandas as pd
import numpy
import pickle

# wrapping our app
app = Flask(__name__)
api = Api(app)  # wrapping app in restful api


# Using own function in Pipeline

num_feats = ['LoanAmount','Loan_Amount_Term','Credit_History','Total_Income', 'Dependents']
cat_feats = ['Gender','Married','Education','Self_Employed','Property_Area']
def numFeat(data):
    return data[num_feats]

def catFeat(data):
    return data[cat_feats]

# To Dense 
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
class ToDenseTransformer(BaseEstimator, TransformerMixin):
    def transform(self, X, y=None):
        return sparse.csr_matrix(X).todense()
    
    def fit(self, X, y=None):
        return self

    
 # load the model   
model = pickle.load( open( "model.p", "rb" ) )   


class Predict(Resource):
    def post(self):
        json_data = request.get_json()
        
        # for 1 observation
        df = pd.DataFrame(json_data.values(), index=json_data.keys()).transpose()
        
        # for multiple observation
        #df = pd.DataFrame(json_data)
        
        # getting predictions from our model.
        # it is much simpler because we used pipelines during development
        result = model.predict(df)
        
        # we cannot send numpt array as a result
        return result.tolist() 
    
# assign endpoint
api.add_resource(Predict, '/predict')    

#running api app
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8000)