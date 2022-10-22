
import json
import pandas as pd
from sklearn.linear_model import LinearRegression
import os
import numpy as np
from sklearn.svm import SVR
filename = "input00.txt"

std_rec = []
# with open("./Testcases/input/"+filename,'r') as f:
    
#     N = f.readline()
#     for x in f:
#         std_rec.append(json.loads(x))
# f.close()

# std_rec_df = pd.json_normalize(std_rec)
# std_rec_df = std_rec_df.fillna(0)
# print(std_rec_df['Physics'])

  
data=[]

#training file
################################
f = open('./trainingAndTest/training.json')
N = f.readline()
for x in f:
    data.append(json.loads(x))
f.close()
training_df = pd.json_normalize(data)
training_df = training_df.fillna(0)

Xtrain_df = training_df.drop(['Mathematics','serial'],axis=1)
Ytrain_df = training_df['Mathematics']

#linearmodel
# reg = LinearRegression()
# reg.fit(Xtrain_df,Ytrain_df)

#svm reg
svmModel = SVR(kernel='rbf')
svmModel.fit(Xtrain_df,Ytrain_df)
#svmModel.score(Xtrain_df,Ytrain_df)
print(svmModel.score(Xtrain_df,Ytrain_df))

#neuralnet

#################################

#testfile
#################################
def pred_math_grade(filename):
    data_test = []
    ft = open('./Testcases/input/'+filename)
    Nt = ft.readline()
    for x in ft:
        data_test.append(json.loads(x))
    ft.close()
    test_df = pd.json_normalize(data_test)
    test_df = test_df.fillna(0)

    Xtest_df = test_df.drop(['serial'],axis=1)
    Xtest_df = Xtest_df[Xtrain_df.columns]
    return svmModel.predict(Xtest_df)
#################################

pathi = "./Testcases/input"

dir = os.listdir(pathi)
for f in dir:
    if(f.startswith("input")):
        out = pred_math_grade(f)

        out = np.round(out)
        o_fname = "out" + f[2:]
        fname = open("./Testcases/output/"+o_fname,'w')
        #fname.write(str(out)[1:-1].replace(' ',",\n"))
        fname.write("\n".join(list(map(str,out))))

        fname.close()
