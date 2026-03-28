import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

# A1
def calculate_entropy(y):
    value,counts=np.unique(y,return_counts=True)
    prob=counts/len(y)
    entropy=-np.sum(prob*np.log2(prob))
    return entropy

# A2
def calculate_gini_index(y):
    value,counts=np.unique(y,return_counts=True)
    prob=counts/len(y)
    gini=1-np.sum(prob**2)
    return gini

# A3
def information_gain(df,feature,y):
    total_entropy=calculate_entropy(y)
    value,counts=np.unique(df[feature],return_counts=True)
    weighted_entropy=0
    for v,count in zip(value,counts):
        subset=df[df[feature]==v]
        subset_entropy=calculate_entropy(subset["Response"])  # entropy of that value of the feature. not entropy of whole feature
        weight=count/len(df)
        weighted_entropy+=weight*subset_entropy # multiplying weight and entropy of each value of the feature
    ig=total_entropy-weighted_entropy
    return ig

def bestfeature_for_rootnode(df,y):
    features=df.columns.drop("Response")
    best_feature=None # best feature for root node
    best_ig=-1  # -1 because ig>=0
    for feature in features:
        ig=information_gain(df,feature,y)
        if ig>best_ig:
            best_ig=ig
            best_feature=feature
    return best_feature

# A4
def bining(column,bins=4,method="width"):
    if method=="width":
        max_val=column.max()
        min_val=column.min()
        width=(max_val-min_val)/bins # how much differnce of intervals
        bin_edges=[min_val+i*width for i in range(bins+1)]
        binned=pd.cut(column,bins=bin_edges,labels=False,include_lowest=True)
    elif method=="frequency":
        binned=pd.qcut(column,q=bins,labels=False,duplicates='drop')
    return binned

# A5
def build_tree(df,y):
    if df["Response"].nunique()==1:
        return df["Response"].iloc[0]
    if len(df.columns)==1:
        return df["Response"].mode()[0]
    best=bestfeature_for_rootnode(df,y)
    tree={best: {}}
    for v in df[best].unique():
        sub=df[df[best]==v].drop(columns=[best])
        tree[best][v]=build_tree(sub,y)
    return tree

#A6
def visualize(X,y):
    model=DecisionTreeClassifier(criterion="entropy")
    model.fit(X,y)
    plt.figure(figsize=(20,10))
    plot_tree(model,feature_names=X.columns,filled=True)
    plt.show()
    
def decision_boundary(X1,y):
    model=DecisionTreeClassifier(criterion="entropy")
    model.fit(X1,y)
    plt.figure(figsize=(8,6))
    DecisionBoundaryDisplay.from_estimator(model,X1,response_method="predict",xlabel="Income",ylabel="MntWines")
    plt.scatter(X1.iloc[:,0], X1.iloc[:,1],c=y,edgecolor="black")
    plt.show()

if __name__ == "__main__":
    df=pd.read_excel("Lab02.xlsx",sheet_name="marketing_campaign")
    y=df["Response"]
    df = df.drop(["ID", "Z_CostContact", "Z_Revenue", "Dt_Customer"], axis=1)

    for col in df.columns:
        if col!="Response" and df[col].nunique()>10 and df[col].dtype != 'object':
            df[col]=bining(df[col],bins=4,method="width")
    df=df.astype(str)
    X = pd.get_dummies(df.drop("Response", axis=1)) 
    print(calculate_entropy(y))
    print(calculate_gini_index(y))
    print(visualize(X,y))
    
    df1=pd.read_excel("Lab02.xlsx",sheet_name="marketing_campaign")
    df1=df1.drop(["ID", "Z_CostContact", "Z_Revenue", "Dt_Customer"], axis=1)
    X1=df1[["Income","MntWines"]]
    X1=X1.fillna(X1.mean())
    print(decision_boundary(X1,y))