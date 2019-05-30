#coding=UTF-8
from sklearn.decomposition import PCA
from sklearn import preprocessing ##标准化使用
import pandas as pd
import numpy as np
from functools import reduce
def ready_pca(train,test):
    #Select the argument
    trainX =train.ix[:,['Low', 'Epic', 'Bug', 'Task','Med','comments']].fillna(0)
    print(trainX)
    #colume = trainX.columns.values.tolist()
    testX =test.ix[:,['Low', 'Epic', 'Bug', 'Task','Med','comments']].fillna(0)
    print(testX)
    trainX  = preprocessing.scale(trainX ) #Standardizatin
    #testX  = preprocessing.scale(testX )   #Standardization
    trainX1 = np.array(trainX)
    testX1 = np.array(testX)

    return trainX1,testX1


def pca_train(trainX1,testX1):
    pca=PCA(copy=True, n_components=0.8, whiten=False)
    ## n_component means the accumulated variance contribution
    pca.fit(trainX1)                                 ##将trainX1传入定义好的pca模型
    components = pca.components_                     ##选取的特征向量对应的系数array
    pacTrainX = pca.transform(trainX1)               ##将trainX1在构造好的pca模型上进行映射
    pcaTestX = pca.transform(testX1)                 ##test主成份
    ratio = pca.explained_variance_ratio_            ##选取的主成份分别对应的方差解释率                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                差占比
    #sum_rati0 = reduce(lambda x,y:x+y,ratio)        ##选取主成份的解释方差
    #print("testX",pcaTestX)
    #print('pacTrainX，pca.fit',pca)
    #print('pcaTestX,ratio',ratio)
    #print('components',components)
    #print('sum_rati0',sum_rati0)
    score=(float)(np.dot(pcaTestX,ratio))
    #defen_train = pd.DataFrame(pacTrainX)
    #components_train = pd.DataFrame(components)
    #defen_test = pd.DataFrame(pcaTestX)
    #print(defen_test)
    #print(components_train)
    return pcaTestX,pacTrainX,components,ratio,score
'''
def Linear(pacTrainX,trainy，pcaTestX):
    pca_svc = LinearSVC()
    pca_svc.fit(defen_train,trainy)
    pca_y_predict= pca_svc.predict(defen_test )
    return pca_y_predict
'''

def main(test):
    InList=[]
    #data set
    with open("label.txt") as fin:
        for line in fin:
            val=line.split()
            InList.append([val[0],val[1],val[2],val[3],val[4],val[5]])
    #train = pd.read_csv("/Users/shenlinger/Documents/work/ML Contest/test.csv")
    train=pd.DataFrame(InList,columns=['Low', 'Epic', 'Bug', 'Task','Med','comments'])
    #test=pd.DataFrame({'a': [5.1], 'b': [3.5], 'c': [1.4],'d': [0.2],})

    train, test = ready_pca(train, test)
    pcaTestX,pcaTrainX,components_train, ratio,score = pca_train(train, test)
    #print('score',score)
    return score




