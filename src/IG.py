#coding:utf-8
import time
import sys
import string
import numpy as np
from scipy import sparse

from sklearn.cross_validation import train_test_split 
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from wx import PreDatePickerCtrl

def loadClassData(filename):
    dataList  = []
    for line in open('../data/'+filename,'r').readlines():#读取分类序列
        dataList.append(int(line.strip()))
    return dataList

def loadTrainData(filename):
    dataList  = []
    for line in open('../data/'+filename,'r').readlines():
        dataList.append(line.strip())
    return dataList

def get_term_dict(doc_terms_list):
    term_set_dict = {}
    for doc_terms in doc_terms_list:
        for term in doc_terms.split():
            term_set_dict[term] = 1
    term_set_list = sorted(term_set_dict.keys())       #term set 排序后，按照索引做出字典
    term_set_dict = dict(zip(term_set_list, range(len(term_set_list))))
    return term_set_dict

def get_class_dict(doc_class_list):
    class_set = sorted(list(set(doc_class_list)))
    class_dict = dict(zip(class_set, range(len(class_set))))
    return  class_dict

def stats_term_df(doc_terms_list, term_dict):
    term_df_dict = {}.fromkeys(term_dict.keys(), 0)
    for term in term_dict:
        for doc_terms in doc_terms_list:
            if term in doc_terms_list:
                term_df_dict[term] +=1                
    return term_df_dict

#正负样本数
def stats_class_df(doc_class_list, class_dict):
    class_df_list = [0] * len(class_dict)
    for doc_class in doc_class_list:
        class_df_list[class_dict[doc_class]] += 1
    return class_df_list


def stats_term_class_df(doc_terms_list, doc_class_list, term_dict, class_dict):
    term_class_df_mat = np.zeros((len(term_dict), len(class_dict)), np.float64)
    for k in range(len(doc_class_list)):
        class_index = class_dict[doc_class_list[k]]
        doc_terms = doc_terms_list[k]
        for term in doc_terms.split():
            term_index = term_dict[term]
            term_class_df_mat[term_index][class_index] +=1
    return  term_class_df_mat

def feature_selection_ig(class_df_list, term_set, term_class_df_mat):
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    C = np.tile(class_df_list, (A.shape[0], 1)) - A
    N = sum(class_df_list)
    D = N - A - B - C
    term_df_array = np.sum(A, axis = 1)
    class_set_size = len(class_df_list)
    
    p_t = term_df_array / N
    p_not_t = 1 - p_t
    p_c_t_mat =  (A + 1) / (A + B + class_set_size)
    p_c_not_t_mat = (C+1) / (C + D + class_set_size)
    p_c_t = np.sum(p_c_t_mat  *  np.log(p_c_t_mat), axis =1)
    p_c_not_t = np.sum(p_c_not_t_mat *  np.log(p_c_not_t_mat), axis =1) 
    term_score_array = p_t * p_c_t + p_not_t * p_c_not_t
    sorted_term_score_index = term_score_array.argsort()[: : -1]
    term_set_fs = [term_set[index] for index in sorted_term_score_index]    
    
    return term_set_fs

def feature_selection_mi(class_df_list, term_set, term_class_df_mat):
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    C = np.tile(class_df_list, (A.shape[0], 1)) - A
    N = sum(class_df_list)
    class_set_size = len(class_df_list)
    
    term_score_mat = np.log(((A+1.0)*N) / ((A+C) * (A+B+class_set_size)))
    term_score_max_list = [max(x) for x in term_score_mat]
    term_score_array = np.array(term_score_max_list)
    sorted_term_score_index = term_score_array.argsort()[: : -1]
    term_set_fs = [term_set[index] for index in sorted_term_score_index]
    
    return term_set_fs

def feature_selection_wllr(class_df_list, term_set, term_class_df_mat):
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    C_Total = np.tile(class_df_list, (A.shape[0], 1))
    N = sum(class_df_list)
    C_Total_Not = N - C_Total
    term_set_size = len(term_set)
    
    p_t_c = (A + 1E-6) / (C_Total + 1E-6 * term_set_size)
    p_t_not_c = (B +  1E-6) / (C_Total_Not + 1E-6 * term_set_size)
    term_score_mat = p_t_c  * np.log(p_t_c / p_t_not_c)
    
    term_score_max_list = [max(x) for x in term_score_mat]
    term_score_array = np.array(term_score_max_list)
    sorted_term_score_index = term_score_array.argsort()[: : -1]
    term_set_fs = [term_set[index] for index in sorted_term_score_index]
    
    print term_set_fs[:10]
    return term_set_fs

def feature_selection_sig(class_df_list, term_set, term_class_df_mat):
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    E = [sum(x)  for x in A]
    C = np.tile(class_df_list, (A.shape[0], 1)) - A
    N = sum(class_df_list)
    D = N - A - B - C
    term_df_array = np.sum(A, axis = 1)
    class_set_size = len(class_df_list)
    alphai  = np.array(A)*1.0/np.array(class_df_list)
    alpha = alphai/sum(alphai)
    beta = 1/(np.array(np.log(class_set_size)+1))
    
    p_t = term_df_array / N
    p_not_t = 1 - p_t
    p_c_t_mat =  (A + 1) / (A + B + class_set_size)
    p_c_not_t_mat = (C+1) / (C + D + class_set_size)
    p_c_t = np.sum(p_c_t_mat  *  np.log(p_c_t_mat)*alpha, axis =1)
    p_c_not_t = np.sum(p_c_not_t_mat * np.log(p_c_not_t_mat)*beta, axis =1) 
    term_score_array = p_t * p_c_t + p_not_t * p_c_not_t
    sorted_term_score_index = term_score_array.argsort()[: : -1]
    term_set_fs = [term_set[index] for index in sorted_term_score_index]    
    
    return term_set_fs

def feature_selection(doc_terms_list, doc_class_list,fs_method):
    class_dict = get_class_dict(doc_class_list)
    term_dict = get_term_dict(doc_terms_list) # 字典{'dict':序号}
    class_df_list = stats_class_df(doc_class_list, class_dict)#正负样本数
    term_class_df_mat = stats_term_class_df(doc_terms_list, doc_class_list, term_dict, class_dict) #字典对应的词在不同类别下的词频(len(term_dict),len(class_dict))
    term_set = [term[0] for term in sorted(term_dict.items(), key = lambda x : x[1])]
    
    if fs_method == 'MI':
        print 'MI'
        term_set_fs = feature_selection_mi(class_df_list, term_set, term_class_df_mat)
    elif fs_method == 'IG':
        print 'IG'
        term_set_fs = feature_selection_ig(class_df_list, term_set, term_class_df_mat)
    elif fs_method == 'WLLR':
        print "WLLR"
        term_set_fs = feature_selection_wllr(class_df_list, term_set, term_class_df_mat)
   
    return term_set_fs

#get the feature vector according to the dictionary
def word2vec(words,dictionary):
    voc = dict(zip(dictionary,dictionary)) # 增加访问速度
    msgRow = []
    characterCol = []
    value = []  
    for i in range(len(words)):
         for word in words[i].split():
            if word in voc:
                msgRow.append(i)
                characterCol.append(dictionary.index(word))
                value.append(1)
    fea = sparse.coo_matrix((value,(msgRow,characterCol)),shape=(len(words),len(dictionary))).tocsr()    
    return fea

def svmClassifer(feaTrain,trainLabel,feaTest):
    print '\n LinearSVC \n' + '*************************'
    clf = LinearSVC( C= 0.8)
    clf.fit(feaTrain,np.array(trainLabel))  
    pred = clf.predict(feaTest); 
    return pred
    
def logisticReg(feaTrain,trainLabel,feaTest):
    print '\n LogisticRegression \n'+ '*************************'

    lr =  LogisticRegression()
    lr.fit(feaTrain,np.array(trainLabel)) 
    pred= lr.predict(feaTest)
    return pred


def classifier(trainData, testData, trainLabel, testLabel):
#     method = ['MI','IG','WLLR']
    method = ['IG']
    for m in method:
        t1 = time.time()
        termSet = feature_selection(trainData, trainLabel,m)
        print len(termSet)
        dictionary = termSet[:50000]
        
        feaTrain = word2vec(trainData,dictionary)
        feaTest = word2vec(testData,dictionary)
        print feaTrain.shape
        print feaTest.shape
        pred = logisticReg(feaTrain,trainLabel,feaTest)
        totalScore(pred,testData,testLabel)
        pred = svmClassifer(feaTrain,trainLabel,feaTest)
        totalScore(pred,testData,testLabel)
        t2 = time.time()
        print t2-t1

#计算F值，并找出预测错误的信息保存在文件中   
def totalScore(pred,x_test,y_test):
    A = 0
    C = 0
    B = 0
    D = 0
#     fout01 = open('../data/error01.txt','a+')
#     fout10 = open('../data/error10.txt','a+')
    for i in range(len(pred)):
        if y_test[i] == 0:
            if pred[i] == 0:
                A += 1
            elif pred[i] == 1:
               B += 1
#                fout01.write('0-1\t%s\n' %x_test[i])
        elif y_test[i] == 1:
            if pred[i] == 0:
                C += 1
#                 fout10.write('1-0\t%s\n' %x_test[i])
            elif pred[i] == 1:
                D +=1
#     fout10.close() 
#     fout01.close()
    print  A,B,C,D, A+B+C+D
    
    rb_pr = 1.0*D/(B+D)
    rb_re = 1.0*D/(C+D)
    rt_pr = 1.0*A/(A+C)
    rt_re = 1.0*A/(A+B)
    
    Frb = 0.65*rb_pr + 0.35*rb_re
    Frt = 0.65*rt_pr + 0.35*rt_re
    Ftotal = 0.7*Frb + 0.3*Frt
    print Ftotal


if __name__ == "__main__":
    
    trainCorpus = []
    classLabel = []
    
    classLabel = loadClassData('classLabel.txt')
    trainCorpus = loadTrainData('trainLeft.txt')     #trainleftstop.txt'
    
    trainData, testData, trainLabel, testLabel = train_test_split(trainCorpus, classLabel, test_size = 0.2) 
    classifier(trainData, testData, trainLabel, testLabel)   
    
