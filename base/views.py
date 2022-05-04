from django.shortcuts import render
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import xgboost

def home(request):
    return render(request, 'index.html')

def preprocess(test_df):
  for col in test_df.columns:
    test_df[col].replace('?', np.nan, inplace=True)
  
  test_df.drop('weight', axis=1, inplace=True)

  constants = ['metformin-pioglitazone', 'metformin-rosiglitazone', 'glimepiride-pioglitazone', 'citoglipton', 'examide', ]

  test_df.drop(constants, axis=1, inplace=True)

  label_encode_cols = ['glyburide-metformin', 'insulin', 'miglitol', 'acarbose', 'rosiglitazone', 'pioglitazone', 'metformin','glyburide', 'glipizide', 'glimepiride', 'chlorpropamide', 'nateglinide', 'repaglinide', 'glipizide-metformin', 'tolazamide', 'troglitazone', 'tolbutamide', 'acetohexamide']

  le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
  temp = np.load('base/encoded_files/four_classes.npy', allow_pickle=True)
  temp = temp.tolist()[0]

  for col in label_encode_cols:
    val = test_df[col].values  
    new = []
    for x in val:
      try:
        new.append(temp.index(x))
      except:
        new.append(-1)
    test_df[col] = new
    


  test_df.diag_1=pd.to_numeric(test_df.diag_1,errors='coerce')
  test_df.diag_2=pd.to_numeric(test_df.diag_2,errors='coerce')
  test_df.diag_3=pd.to_numeric(test_df.diag_3,errors='coerce')

    
  test_df['race'].replace(np.nan, 'Missing', inplace=True)
  test_df['payer_code'].replace(np.nan, 'Missing', inplace=True)


  for col in ['race', 'gender', 'age', 'payer_code', 'medical_specialty', 'max_glu_serum', 'A1Cresult', 'change',	'diabetesMed']:
    name = 'base/encoded_files/'+col+'.npy'
    temp = np.load(name, allow_pickle=True)
    temp = temp.tolist()[0]
    
    val = test_df[col].values  
    new = []
    for x in val:
      try:
        new.append(temp.index(x))
      except:
        new.append(-1)
    test_df[col] = new
  
  return test_df
  


def getPredictions(df):
    xgb_model = xgboost.XGBClassifier() # or which ever sklearn booster you're are using

    xgb_model.load_model("xgb_model.bin")
    scaled = pickle.load(open('scaler.sav', 'rb'))

    prediction = xgb_model.predict(scaled.transform(df))
    
    return prediction

def result(request):
    input_data = request.GET['input']
    data = input_data.split(',')
    cols = ['race', 'gender', 'age', 'weight', 'admission_type_id',\
       'discharge_disposition_id', 'admission_source_id', 'time_in_hospital',\
       'payer_code', 'medical_specialty', 'num_lab_procedures',\
       'num_procedures', 'num_medications', 'number_outpatient',\
       'number_emergency', 'number_inpatient', 'diag_1', 'diag_2', 'diag_3',\
       'number_diagnoses', 'max_glu_serum', 'A1Cresult', 'metformin',\
       'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',\
       'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',\
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',\
       'tolazamide', 'examide', 'citoglipton', 'insulin',\
       'glyburide-metformin', 'glipizide-metformin',\
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',\
       'metformin-pioglitazone', 'change', 'diabetesMed', 'index']

    data = np.array(data)
    df = pd.DataFrame(data.reshape(1,-1), columns=cols)
    df = preprocess(df)
    df.drop('index', axis=1, inplace=True)

    result = getPredictions(df)
    return render(request, 'result.html', {'result': result})
