
import os.path
from glob import glob
from sklearn.datasets import get_data_home
import os.path
from sklearn.feature_extraction.text import HashingVectorizer
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

#朴素贝叶斯 （多项式分布） ,用于处理多项离散数据集
from sklearn.naive_bayes import MultinomialNB

from sklearn.externals import joblib

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

from sklearn.feature_selection import SelectKBest ,chi2

data=[]#所有数据集

xtest=[]#测试集文本向量
ytest=[]#测试集类别

xtrain=[]#训练集文本向量
ytrain=[]#训练集类别

TrainDataSize = 8 #训练集个数

all_classes = np.arange(20) #分类器类别上限

printjumpsize = 1 # 输出间隔

FeatureSpaceSize = 15000
A_all=0

updatesize = 8

modlename='FetureIncModle('+str(FeatureSpaceSize)+')'

vocname='vocubulary/VocubularyforFeatureIncre'+str(FeatureSpaceSize)+'.v'

# 读入数据集 -------------------------------------------------------------------------------
def ReadData(path):
    i = 0
    data_path = os.path.join(get_data_home(), path)
    for docname in glob(os.path.join(data_path, "*")):
        doc = []
        docnum=0
        for filename in glob(os.path.join(docname, "*.txt")):
            # print(filename)

            s = open(filename, 'r', encoding='gb18030', errors='ignore')
            content = s.read()
            doc.append({'content': content, 'type': i})
            docnum+=1
            # print(content)
            # print("---------------------------------------------------")
            s.close()
        data.append(doc)
        print("已读入样本集: ", docname,'共计',docnum,'份')
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
                    xtrain[p].append(data[j][i]['content'])
                    ytrain[p].append(data[j][i]['type'])

getTRAIN()
ReadData("FUDAN/answer")
getTEST()
print('特征空间增量优化朴素贝叶斯增量学习:')
print('训练样本集 ',len(ytrain[0]),'条',TrainDataSize,'份')
print('测试样本集 ',len(ytest),' 条')
# end 划分训练类别成为测试和训练样本集 ---------------------------------------------------------------


# test data statistics 测试数据的统计
test_stats = {'n_test': 0, 'n_test_pos': 0}
tick = time.time()
parsing_time = time.time() - tick
tick = time.time()

# 这里有一些支持`partial_fit`方法的分类器
# 新创建分类器容器
partial_fit_classifiers = {
    modlename: MultinomialNB(alpha=0.01),
}
# 载入旧的分类器容器
# 载入以往保存下来的模型------------------------------------------------------
def getclassifiers(xclassifiers):
    for cls_name, cls in partial_fit_classifiers.items():
        if os.path.exists("modle/Model_" + cls_name + ".m"):
            cls = joblib.load("modle/Model_" + cls_name + ".m")
        xclassifiers[cls_name]=cls

# getclassifiers(xclassifiers)

# end 获取以往保存下来的的模型------------------------------------------------------



vectorizing_time = time.time() - tick
test_stats['n_test'] += len(ytest)
test_stats['n_test_pos'] += sum(ytest)


def progress(cls_name, stats):
    """Report progress information, return a string.报告进度信息，返回一个字符串。"""
    duration = time.time() - stats['t0']
    s = "%20s 分类器 : \t" % cls_name
    s += "%(n_train)6d 条特征空间训练样本  " % stats
    s += "%(n_test)6d 条测试样本  " % test_stats
    s += "准确度: %(accuracy).4f " % stats
    s += "信息损失率： %.4f " % stats['lost']
    s += "增长率： %.4f " % stats['increaSpeed']
    return s

def progress2(cls_name, stats,no):
    """Report progress information, return a string.报告进度信息，返回一个字符串。"""
    Accuracy=stats['accuracy']
    duration = time.time() - stats['t0']
    s = "%20s 分类器 : \t" % cls_name
    s += "%(n_train)6d 条特征空间训练样本  " % stats
    s += "%(n_test)6d 条测试样本  " % test_stats
    s += "准确度: %.4f " % Accuracy
    s += "信息损失率： %.4f " % stats['lost']
    s += "增长率： %.4f " % stats['increaSpeed']
    # s += "共计 %.2fs (%5d 样本/s)" % (duration, stats['n_train'] / duration)
    return s


cls_stats = {}
AccuracyAverage = {}
for cls_name in partial_fit_classifiers:
    stats = {'n_train': 0, 'n_train_pos': 0,
             'accuracy': 0.0,
             'increaSpeed':0.0,
             'lost': 0.0,
             'accuracy_history': [(0, 0)], 't0': time.time(),
             'runtime_history': [(0, 0)], 'total_fit_time': 0.0}
    cls_stats[cls_name] = stats
    AccuracyAverage[cls_name] = stats


total_vect_time = 0.0

def sortbyword(one_list,size,word):
    '''''
    使用排序的方法
    '''
    result_list = []
    result_listname = []
    temp_list=sorted(one_list,key=lambda x:x[word], reverse=True)
    # print("sorted!")
    i=0
    j=0
    k=0
    while i<len(temp_list) and k==0:
        # print(i)
        if temp_list[i]['name'] not in result_listname:
                result_list.append(temp_list[i])
                result_listname.append(temp_list[i]['name'])
                j+=1
                if(j>=size):
                    k=1

        i+=1
    return result_list


oldVocubularysave = []
newVocubularysave = []


T=0

def IncreasingFIT():
    global total_vect_time
    classifiers = {
        modlename: MultinomialNB(alpha=0.01),
    }

    # 载入保存模型
    # getclassifiers(classifiers)

    Vocubularysave=newVocubularysave
    VocubularyList = []
    for numV in Vocubularysave:
        VocubularyList.append(numV['name'])
    vectorizer = CountVectorizer(stop_words=None, vocabulary=VocubularyList)
    transformer = TfidfTransformer()

    count = vectorizer.fit_transform(xtest)
    X_test = transformer.fit_transform(count)



    for i in range(TrainDataSize):
        tick = time.time()

        count = vectorizer.fit_transform(xtrain[i])
        X_train = transformer.fit_transform(count)

        total_vect_time += time.time() - tick

        for cls_name, cls_useless in partial_fit_classifiers.items():
            cls = classifiers[cls_name]

            tick = time.time()
            # update estimator with examples in the current mini-batch
            # 使用当前最小批次中的示例更新估算器
            # print(X_train)

            cls.partial_fit(X_train, ytrain[i], classes=all_classes)

            # if i % printjumpsize == 0:
            if i == 0:
                FirstScore= cls.score(X_test, ytest)

            if i == (TrainDataSize-1) :
            # if i !=100:
                if T == 0:
                    # accumulate test accuracy stats
                    # 累积测试准确度统计
                    cls_stats[cls_name]['total_fit_time'] += time.time() - tick
                    cls_stats[cls_name]['n_train'] += X_train.shape[0]
                    cls_stats[cls_name]['n_train_pos'] += sum(ytrain[i])
                    tick = time.time()

                    # 测试准确性函数
                    cls_stats[cls_name]['accuracy'] = cls.score(X_test, ytest)

                    cls_stats[cls_name]['lost'] = A_all - cls.score(X_test, ytest)

                    AccuracyAverage[cls_name]['increaSpeed'] = cls.score(X_test, ytest)-FirstScore

                    cls_stats[cls_name]['prediction_time'] = time.time() - tick
                    acc_history = (cls_stats[cls_name]['accuracy'],
                                   cls_stats[cls_name]['n_train'])
                    cls_stats[cls_name]['accuracy_history'].append(acc_history)

                    # 累积测试准确度统计
                    print(progress(cls_name, cls_stats[cls_name]))
                if T != 0:
                    AccuracyAverage[cls_name]['total_fit_time'] += time.time() - tick
                    AccuracyAverage[cls_name]['n_train'] += X_train.shape[0]
                    AccuracyAverage[cls_name]['n_train_pos'] += sum(ytrain[i])
                    tick = time.time()

                    # 测试准确性函数
                    AccuracyAverage[cls_name]['lost'] = A_all - cls.score(X_test, ytest)
                    AccuracyAverage[cls_name]['accuracy'] = cls.score(X_test, ytest)

                    acc_history = (AccuracyAverage[cls_name]['accuracy'],
                                   AccuracyAverage[cls_name]['n_train'])
                    AccuracyAverage[cls_name]['accuracy_history'].append(acc_history)

                    AccuracyAverage[cls_name]['increaSpeed'] = cls.score(X_test, ytest)-FirstScore

                    print(progress2(cls_name, AccuracyAverage[cls_name],T))

    # 保存训练好的模型------------------------------------------------------
    def saveModel():
        for cls_name, cls_useless in partial_fit_classifiers.items():
            cls = classifiers[cls_name]
            joblib.dump(cls, "modle/Model_" + cls_name + ".m")

    saveModel()

    # end 保存训练好的模型------------------------------------------------------


def FeatrueLearning():
    global recordAccuracy
    recordAccuracy=[]
    vectorizer = CountVectorizer(stop_words=None)
    transformer = TfidfTransformer()
    global total_vect_time,parsing_time,vectorizing_time,oldVocubularysave,newVocubularysave

    # 载入保存了的词典
    # if os.path.exists("vocubulary/VocubularyforFeatureIncre.v"):
    #     oldVocubularysave = joblib.load("vocubulary/VocubularyforFeatureIncre.v")

    global T
    # 特征空间更新循环
    for T in range(updatesize):
        tick = time.time()
        # X_train = vectorizer.transform(xtrain[i])

        count = vectorizer.fit_transform(xtrain[T])
        X_train = transformer.fit_transform(count)

        # ----------------------------------------
        VocubularyList = vectorizer.get_feature_names()

        model1=SelectKBest(chi2, k=1)

        X_chi2 = model1.fit_transform(X_train, ytrain[T])
        j = 0
        for i in VocubularyList:
            # print(i,",",vectorizer2.vocabulary_[i],",",max(tfidf[vectorizer2.vocabulary_[i]]))
            newVocubularysave.append({"name": i,
                                      'numb': vectorizer.vocabulary_[i],
                                      'value': model1.scores_[j]})
            j = j + 1

        newVocubularysave=oldVocubularysave+newVocubularysave
        newVocubularysave=sortbyword(newVocubularysave, FeatureSpaceSize, 'value')

        oldVocubularysave = newVocubularysave
        total_vect_time += time.time() - tick

        # 测试集的处理-----------------
        tick = time.time()
        parsing_time = time.time() - tick
        tick = time.time()

        vectorizing_time = time.time() - tick
        test_stats['n_test'] = len(ytest)
        test_stats['n_test_pos'] += sum(ytest)

        #特征空间保存
        joblib.dump(newVocubularysave, vocname)

        # 分类器核模型更新
        IncreasingFIT()


print('开始特征空间增量训练...')
FeatrueLearning()
print('已完成特征空间增量...')


###############################################################################
# Plot results
# 绘制结果
# ------------

def plot_accuracy(x, y, x_legend):
    """Plot accuracy as a function of x."""
    x = np.array(x)
    y = np.array(y)
    plt.title('Classification accuracy as a function of %s' % x_legend)
    plt.xlabel('%s' % x_legend)
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.plot(x, y)

def drawresults():
    rcParams['legend.fontsize'] = 10
    cls_names = list(sorted(cls_stats.keys()))

    # Plot accuracy evolution 绘制准确性演变情况
    plt.figure()
    for _, stats in sorted(cls_stats.items()):
        # Plot accuracy evolution with #examples 用#examples绘制准确性演变图
        accuracy, n_examples = zip(*stats['accuracy_history'])
        plot_accuracy(n_examples, accuracy, "training examples (#)")
        ax = plt.gca()
        ax.set_ylim((0.85, 1))
    plt.legend(cls_names, loc='best')

    plt.show()


drawresults()

# print(len(recordAccuracyList))


