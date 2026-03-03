import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,calinski_harabasz_score,davies_bouldin_score
import matplotlib.pyplot as plt

#A1
def training_regression(X_train,y_train):
    reg=LinearRegression().fit(X_train,y_train)
    y_train_predict=reg.predict(X_train)
    return reg,y_train_predict
#A2
def error_metrics(y_true,y_train_predict):
    mse=np.mean((y_true-y_train_predict)**2)
    rmse=np.sqrt(mse)
    mape=np.mean(np.abs((y_true-y_train_predict)/y_true)*100)
    r2=r2_score(y_true,y_train_predict)
    return f"mse:{mse},rmse:{rmse},mape:{mape},r2_score:{r2}"

def test_prediction(X_test,y_test,reg):
    y_test_pred=reg.predict(X_test)
    mse=np.mean((y_test-y_test_pred)**2)
    rmse=np.sqrt(mse)
    mape=np.mean(np.abs((y_test-y_test_pred)/y_test)*100)
    r2=r2_score(y_test,y_test_pred)
    return f"mse:{mse},rmse:{rmse},mape:{mape},r2_score:{r2}"

#A3
def A1(X1_train,y1_train):
    reg1=LinearRegression().fit(X1_train,y1_train)
    y1_train_predict=reg1.predict(X1_train)
    return reg1,y1_train_predict

def A2(y1_train_predict,X1_test,y1_test,reg1):
    print(error_metrics(y1_train,y1_train_predict))
    print(test_prediction(X1_test,y1_test,reg1))
    
#A4
def kmeans_clustering(X_train):
    kmeans=KMeans(n_clusters=2,random_state=0,n_init="auto").fit(X_train)
    kmeans.labels_
    kmeans.cluster_centers_
    
#A5
def clustering_metrics(X_train):
    kmeans=KMeans(n_clusters=2,random_state=42).fit(X_train)
    s=silhouette_score(X_train,kmeans.labels_)
    c=calinski_harabasz_score(X_train,kmeans.labels_)
    d=davies_bouldin_score(X_train,kmeans.labels_)
    return s,c,d 

#A6
def kmeans_different_k(X_train):
    k=range(3,10)
    s=[]
    c=[]
    d=[]
    for i in k:
        kmeans=KMeans(n_clusters=i,random_state=42).fit(X_train)
        s.append(silhouette_score(X_train,kmeans.labels_))
        c.append(calinski_harabasz_score(X_train,kmeans.labels_))
        d.append(davies_bouldin_score(X_train,kmeans.labels_))
        
    plt.figure()
    plt.plot(list(k),s)
    plt.xlabel("k")
    plt.ylabel("silhouette score")
    plt.show()
    
    plt.figure()
    plt.plot(list(k),c)
    plt.xlabel("k")
    plt.ylabel("calinski harabasz score")
    plt.show()
    
    plt.figure()
    plt.plot(list(k),d)
    plt.xlabel("k")
    plt.ylabel("davies bouldin score")
    plt.show()
    
#A7
def elbow_method(X_train):
    k=range(3,10)
    distorsions=[]
    for i in k:
        kmeans=KMeans(n_clusters=i).fit(X_train)
        distorsions.append(kmeans.inertia_)
    plt.figure()
    plt.plot(k,distorsions)
    plt.xlabel("k")
    plt.ylabel("distorsions")
    plt.show()

if __name__ == "__main__":
    df=pd.read_excel("Lab02.xlsx",sheet_name="IRCTC_Stock_Price")
    X=df[["Price"]].values
    y=df["Chg%"].values
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
    training_regression(X_train,y_train)
    
    #A3
    X1=df[["Price","Open","High","Low"]].values
    y1=df["Chg%"].values
    X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.3)
    
    #A1 and #A2
    reg,y_train_predict=training_regression(X_train,y_train)
    print(error_metrics(y_train,y_train_predict))
    test_prediction(X_test,y_test,reg)
    
    #A3
    reg1,y1_train_predict=A1(X1_train,y1_train)
    print(A2(y1_train_predict,X1_test,y1_test,reg1))
    
    #A4
    kmeans_clustering(X1_train)
    #A5
    print(clustering_metrics(X_train))
    #A6
    print(kmeans_different_k(X_train))
    #A7
    print(elbow_method(X_train))