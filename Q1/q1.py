  import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

def is_float(element: str) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False

def pred_stock_pric(filename):
    L = []
    L_date = []
    L_month = []
    L_yr = []
    L_time = []

    L_test_date = []
    L_test_month = []
    L_test_yr = []
    L_test_time = []

    with open('./'+filename,'r') as f:
        N = f.readline()
        for x in f:
            if(not(is_float(x.split()[2]))):
                L_test_date.append(float(x.split()[0].split('/')[1]))
                L_test_month.append(float(x.split()[0].split('/')[0]))
                L_test_yr.append(float(x.split()[0].split('/')[2]))
                L_test_time.append(float(x.split()[1].split(':')[2])+(float(x.split()[1].split(':')[1])*60)+(float(x.split()[1].split(':')[0])*60*60)/(24*60*60))
                continue 
            L.append(float(x.split()[2]))
            L_date.append(float(x.split()[0].split('/')[1]))
            L_month.append(float(x.split()[0].split('/')[0]))
            L_yr.append(float(x.split()[0].split('/')[2]))
            L_time.append(float(x.split()[1].split(':')[2])+(float(x.split()[1].split(':')[1])*60)+(float(x.split()[1].split(':')[0])*60*60)/(24*60*60))

    f.close()
    x1 = np.array(L_date)
    x2 = np.array(L_month)
    x3 = np.array(L_yr)
    x4 = np.array(L_time)

    xt1 = np.array(L_test_date)
    xt2 = np.array(L_test_month)
    xt3 = np.array(L_test_yr)
    xt4 = np.array(L_test_time)

    X_test = np.column_stack((xt2,xt1+xt4))

    y_train = np.array(L)

    X = np.column_stack((x2,x1+x4))
    # polynom = PolynomialFeatures(degree = 6)
    # X = polynom.fit_transform(X)
    # X_test = polynom.fit_transform(X_test)
    # reg = LinearRegression()
    reg = SVR(kernel='rbf')
    reg.fit(X,y_train)
    print(reg.score(X,y_train))
    y_predict = reg.predict(X_test)
    return y_predict

path = "./"
ind = 0
dir = os.listdir(path)
for f in dir:
    if(f.startswith("input")):
        out = pred_stock_pric(f)
        o_fname = "out" + f[2:]
        fname = open(o_fname,'w')
        # fname.write(str(out)[1:-1].replace(' ',"\n"))
        fname.write("\n.".join(list(map(str,out))))
        fname.close()
