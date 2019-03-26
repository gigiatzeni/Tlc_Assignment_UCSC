import flask
from flask import request, jsonify
import pandas as pd
import pickle as pkl

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    return '''<h1>Telco churn API web service</h1>
<p>Predict the probability of churn</p>
<hr>
<h2>Available methods</h2>
<p>
<b>GET</b> /api/v1/predict</p>

<a href="http://127.0.0.1:5000/api/v1/predict?gender=0&SeniorCitizen=0&Partner=1&Dependents=0&tenure=100&PhoneService=1&PaperlessBilling=0&MonthlyCharges=76.2&TotalCharges=76.2&MultipleLines_No=1&MultipleLines_NoPhoneService=0&MultipleLines_Yes=0&InternetService_Dsl=0&InternetService_FiberOptic=1&InternetService_No=0&OnlineSecurity_No=1&OnlineSecurity_NoInternetService=0&OnlineSecurity_Yes=0&OnlineBackup_No=1&OnlineBackup_NoInternetService=0&OnlineBackup_Yes=0&DeviceProtection_No=0&DeviceProtection_NoInternetService=0&DeviceProtection_Yes=1&TechSupport_No=1&TechSupport_NoInternetService=0&TechSupport_Yes=0&StreamingTV_No=1&StreamingTV_NoInternetService=0&StreamingTV_Yes=0&StreamingMovies_No=1&StreamingMovies_NoInternetService=0&StreamingMovies_Yes=0&Contract_Month-to-month=1&Contract_OneYear=0&Contract_TwoYear=0&PaymentMethod_BankTransfer(automatic)=0&PaymentMethod_CreditCard(automatic)=0&PaymentMethod_ElectronicCheck=1&PaymentMethod_MailedCheck=0">
Try the request</a> </br>

'''

@app.route('/api/v1/predict', methods=['GET'])
def api_id():
    X_test=pd.DataFrame.from_dict(request.args, orient='index').transpose()
    
    with open('lr_model.pkl', 'rb') as f:
        lr = pkl.load(f)

    lr_preds = lr.predict(X_test)
    lr_probs = [y for (x, y) in lr.predict_proba(X_test)]
    ret = {"prob": lr_probs[0]}
    return jsonify(ret)

app.run(host='0.0.0.0')