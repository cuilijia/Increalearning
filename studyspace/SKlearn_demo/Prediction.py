
import os.path
from glob import glob
from sklearn.datasets import get_data_home
import os.path
from sklearn.feature_extraction.text import HashingVectorizer
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

#朴素贝叶斯  ,用于处理多项离散数据集
from sklearn.naive_bayes import MultinomialNB

from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer


data=[]#所有数据集

xtest=[]#测试集文本向量
ytest=[]#测试集类别

xtrain=[]#训练集文本向量
ytrain=[]#训练集类别

featurespacesize=15000
modlename='FetureIncModle('+str(featurespacesize)+')'
vocname='vocubulary/VocubularyforFeatureIncre'+str(featurespacesize)+'.v'
# 实际参与训练的类型范围
Num_mintype=0
Num_maxtype=20
type_start =Num_mintype
type_end = Num_maxtype
typerange = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
             10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

if type_end>Num_maxtype:
    type_end=Num_maxtype

# 采用的数据集 -------------------------------------------------------------------------------
#  由复旦大学李荣陆提供。answer.rar为测试语料，共9833篇文档；train.rar为训练语料，共9804篇文档，分为20个类别。
#  训练语料和测试语料基本按照1:1的比例来划分。收集工作花费了不少人力和物力，所以请大家在使用时尽量注明来源
# （复旦大学计算机信息与技术系国际数据库中心自然语言处理小组）。
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

# end 读入数据集 -------------------------------------------------------------------------------

# 划分训练类别成为测试和训练样本集 ---------------------------------------------------------------
def getTEST():
    global data
    for j in range(type_start, type_end):
        for i in range(int(len(data[j]))):
                xtest.append(data[j][i]['content'])
                ytest.append(data[j][i]['type'])


ReadData("FUDAN/answer")
getTEST()

# end 划分训练类别成为测试和训练样本集 ---------------------------------------------------------------


# # 数据集文本向量化 (哈希技巧) -------------------------------------------------------
# vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 18,
#                                alternate_sign=False)
# X_test = vectorizer.transform(xtest)
# # end 数据集文本向量化 (哈希技巧) -------------------------------------------------------
# 数据集文本向量化  -------------------------------------------------------

# 特征空间向量化
oldVocubularysave=[]
if os.path.exists(vocname):
    oldVocubularysave = joblib.load(vocname)
    print('get voc!')
else:
    print('no voc!')

VocubularyList=[]
for numV in oldVocubularysave:
    VocubularyList.append(numV['name'])
vectorizer = CountVectorizer(stop_words=None,vocabulary=VocubularyList)
transformer = TfidfTransformer()

count = vectorizer.fit_transform(xtest)
X_test = transformer.fit_transform(count)
# end 数据集文本向量化  -------------------------------------------------------


# 这里有一些支持`partial_fit`方法的分类器
# 新创建分类器容器
partial_fit_classifiers = {
    modlename: MultinomialNB(alpha=0.01),
}
# 载入旧的分类器容器
classifiers={
    modlename: MultinomialNB(alpha=0.01),
}

cls_stats = {}


# 载入以往保存下来的模型------------------------------------------------------
def getclassifiers():
    for cls_name, cls in partial_fit_classifiers.items():
        if os.path.exists("modle/Model_" + cls_name + ".m"):
            cls = joblib.load("modle/Model_" + cls_name + ".m")
            print('get modle!')
        else:
            print("no modle!")
        classifiers[cls_name]=cls

getclassifiers()

# end 获取以往保存下来的的模型------------------------------------------------------

# 预测类别------------------------------------------------------
def prediction():
    Predicty = []
    for cls_name, cls_useless in partial_fit_classifiers.items():
        cls = classifiers[cls_name]
        # 预测函数
        Predicty=cls.predict(X_test)

    # for i in range(len(Predicty)):
    #     print(Predicty[i].astype('int'),end='')
    #     print("==",end='')
    #     print(ytest[i])

    for i in range(len(Predicty)):
        if Predicty[i]!=ytest[i]:
            print(i,end=':')
            print(Predicty[i],end='')
            print("(",end='')
            print(ytest[i],end=')     ')

prediction()

# end 保存训练好的模型------------------------------------------------------

