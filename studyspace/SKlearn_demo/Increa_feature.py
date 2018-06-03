
import os.path
from glob import glob
from sklearn.datasets import get_data_home
import os.path
from sklearn.feature_extraction.text import HashingVectorizer
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

#SGDClassifier是一个用随机梯度下降算法训练的线性分类器的集合。默认情况下是一个线性（软间隔）支持向量机分类器。
from sklearn.linear_model import SGDClassifier
#线性回归模型
from sklearn.linear_model import PassiveAggressiveClassifier
#线性回归模型
from sklearn.linear_model import Perceptron
#朴素贝叶斯  ,用于处理多项离散数据集
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest ,chi2

data=[]#所有数据集

xtest=[]#测试集文本向量
ytest=[]#测试集类别

xtrain=[]#训练集文本向量
ytrain=[]#训练集类别

TrainDataSize = 1 #训练集个数

all_classes = np.arange(20) #分类器类别上限

printjumpsize = 1 # 输出间隔

FeatureSpaceSize =5000

updatesize=1

vocname='vocubulary/VocubularySave(5000)forone.v'
# 读入数据集 -------------------------------------------------------------------------------
# 读入数据集 -------------------------------------------------------------------------------
def ReadData(path):
    i = 0
    data_path = os.path.join(get_data_home(), path)
    for docname in glob(os.path.join(data_path, "*")):
        doc = []
        for filename in glob(os.path.join(docname, "*.txt")):
            # print(filename)

            s = open(filename, 'r', encoding='gb18030', errors='ignore')
            content = s.read()
            doc.append({'content': content, 'type': i})
            # print(content)
            # print("---------------------------------------------------")
            s.close()

        data.append(doc)
        print("已读入样本集: ", docname)
        i = i + 1

ReadData("FUDAN/train")
# end 读入数据集 -------------------------------------------------------------------------------

# 实际参与训练的类型范围
Num_mintype=0
Num_maxtype=len(data)
type_start =Num_mintype
type_end = 20
typerange = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
             10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

if type_end>Num_maxtype:
    type_end=Num_maxtype
# 划分训练类别成为测试和训练样本集 ---------------------------------------------------------------
def getTEST():
    global data
    for j in range(type_start, type_end):
        for i in range(int(len(data[j]))):
                xtest.append(data[j][i]['content'])
                ytest.append(data[j][i]['type'])

def getTRAIN():
    global data
    for n in range(TrainDataSize):
        xtrain.append([])
        ytrain.append([])

    # print('type_end:',type_end)

    for j in range(type_start, type_end):
        for i in range(len(data[j])):
            for p in range(TrainDataSize):
                if (i in range(int(len(data[j]) / (TrainDataSize) * (p)),
                               int(len(data[j]) / (TrainDataSize)) * (p + 1))):
                    # print(int(len(data[j]) / (TrainDataSize ) * (p)), 'to',
                    #       int(len(data[j]) / (TrainDataSize )) * (p+1 ))
                    xtrain[p].append(data[j][i]['content'])
                    ytrain[p].append(data[j][i]['type'])

getTRAIN()
ReadData("FUDAN/answer")
getTEST()
print('训练样本集 ',TrainDataSize,' 份')
print('测试样本集 ',1,' 份')
print("一份样本集为 %d 条  " % (len(ytest)))
# end 划分训练类别成为测试和训练样本集 ---------------------------------------------------------------

#不放回情况训练集划分
def cutdatafornoreturen():
    global xtrain,ytrain
    Nxtrain=[]
    xx=[]
    yy=[]
    Nytrain=[]
    for dicnum in range(5):
        for cc in range(len(xtrain[dicnum])):
            Nxtrain.append(xtrain[dicnum][cc])
            Nytrain.append(ytrain[dicnum][cc])
    xx.append(Nxtrain)
    xtrain=xx
    yy.append(Nytrain)
    ytrain=yy
    print(len(xtrain))

# 这里有一些支持`partial_fit`方法的分类器
# 新创建分类器容器
partial_fit_classifiers = {
    'SGD': SGDClassifier()

}
# 载入旧的分类器容器
classifiers={
    'SGD': SGDClassifier()
}

cls_stats = {}
for cls_name in partial_fit_classifiers:
    stats = {'n_train': 0, 'n_train_pos': 0,
             'accuracy': 0.0, 'accuracy_history': [(0, 0)], 't0': time.time(),
             'runtime_history': [(0, 0)], 'total_fit_time': 0.0}
    cls_stats[cls_name] = stats

total_vect_time = 0.0

# 载入以往保存下来的模型------------------------------------------------------
def getclassifiers():
    for cls_name, cls in partial_fit_classifiers.items():
        if os.path.exists("modle/Model_" + cls_name + ".m"):
            cls = joblib.load("modle/Model_" + cls_name + ".m")
        classifiers[cls_name]=cls

getclassifiers()

# end 获取以往保存下来的的模型------------------------------------------------------


# Main loop : iterate on mini-batches of examples
# 主循环：迭代小批量的例子-----------------------------------------------


test_stats = {'n_test': 0, 'n_test_pos': 0}
parsing_time=0
vectorizing_time=0


# 测试集的处理------------------------------------------------------------------

def progress(cls_name, stats):
    """Report progress information, return a string.报告进度信息，返回一个字符串。"""
    duration = time.time() - stats['t0']
    s = "%20s 分类器 : \t" % cls_name
    s += "%(n_train)6d 条训练样本  " % stats
    s += "%(n_test)6d 条测试样本  " % test_stats
    s += "准确度: %(accuracy).3f " % stats
    s += "共计 %.2fs (%5d 样本/s)" % (duration, stats['n_train'] / duration)
    return s

def sortbyword(one_list,size,word):
    '''''
    使用排序的方法
    '''
    result_list = []
    result_listname = []
    temp_list=sorted(one_list,key=lambda x:x[word], reverse=True)
    print("sorted!")
    i=0
    j=0
    while i<len(temp_list):
        # print(i)
        if temp_list[i]['name'] not in result_listname:
                result_list.append(temp_list[i])
                result_listname.append(temp_list[i]['name'])
                j+=1
                i+=1
                if(j>=size):
                    return result_list
        else:
            i+=1

    return result_list


oldVocubularysave = []
newVocubularysave = []

# if os.path.exists("newVocubularysave.v"):
#     oldVocubularysave = joblib.load("newVocubularysave.v")


stopwordslist=[]
if os.path.exists("stopwords"):
    print("find stopwords!")
    file_object = open('stopwords')
    try:
        list_of_all_the_lines = file_object.read( )
        for word in list_of_all_the_lines:
            stopwordslist.append(word)
            # print (word)
    finally:
        file_object.close()
# print(stopwordslist[0])

def IncreasingFIT():
    vectorizer = CountVectorizer(stop_words=stopwordslist)
    transformer = TfidfTransformer()
    global total_vect_time,parsing_time,vectorizing_time,oldVocubularysave,newVocubularysave
    # for T in range(TrainDataSize):
    for T in range(updatesize):
        tick = time.time()
        # X_train = vectorizer.transform(xtrain[i])

        count = vectorizer.fit_transform(xtrain[T])
        X_train = transformer.fit_transform(count)

        # ----------------------------------------
        VocubularyList = vectorizer.get_feature_names()

        # vectorizer = CountVectorizer(stop_words=None,vocabulary=VocubularyList)

        # tfidf = X_train.toarray().T

        print(X_train.shape)
        model1=SelectKBest(chi2, k=FeatureSpaceSize)
        X_chi2 = model1.fit_transform(X_train, ytrain[T])
        print(X_chi2.shape)
        print(model1.scores_.shape)


        j = 0
        for i in VocubularyList:
            # print(i,",",vectorizer2.vocabulary_[i],",",max(tfidf[vectorizer2.vocabulary_[i]]))
            newVocubularysave.append(
                {"name": i, 'numb': vectorizer.vocabulary_[i], 'value': model1.scores_[j] })
            j = j + 1

        print("get newVocubularysave!")
        # newVocubularysave=oldVocubularysave
        newVocubularysave=oldVocubularysave+newVocubularysave
        newVocubularysave=sortbyword(newVocubularysave, FeatureSpaceSize, 'value')
        # print(newVocubularysave)
        l=[]
        for numV in newVocubularysave:
           l.append(numV['name'])
        # print(l)
        print("========================================================")
        print("========================================================")
        oldVocubularysave = newVocubularysave


        # /-----------------------------------------

        total_vect_time += time.time() - tick

        # 测试集的处理-----------------


        tick = time.time()
        parsing_time = time.time() - tick
        tick = time.time()


        vectorizing_time = time.time() - tick
        test_stats['n_test'] += len(ytest)
        test_stats['n_test_pos'] += sum(ytest)
        # end 数据集文本向量化 (哈希技巧) -------------------------------------------------------

print('开始增量训练...')
IncreasingFIT()
print('已完成...')
# end 主循环：迭代小批量的例子-----------------------------------------------


print(len(newVocubularysave))
# joblib.dump(newVocubularysave, "ONETIMEVocubularySave.v")
joblib.dump(newVocubularysave, vocname)
# end 保存训练好的模型------------------------------------------------------




