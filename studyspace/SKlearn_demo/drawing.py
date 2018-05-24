
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

from sklearn.externals import joblib

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

data=[]#所有数据集

xtest=[]#测试集文本向量
ytest=[]#测试集类别

xtrain=[]#训练集文本向量
ytrain=[]#训练集类别

TrainDataSize = 8 #训练集个数

all_classes = np.arange(20) #分类器类别上限

printjumpsize=8 # 输出间隔

# 特征空间Vocubulary大小为5000

###############################################################################
# Plot results
# 绘制结果
# ------------

# item1 = joblib.load("ResultPicData/dataitem1.d")
# keys1 = joblib.load("ResultPicData/datakeys1.d")
# item2 = joblib.load("ResultPicData/dataitem2.d")
# keys2 = joblib.load("ResultPicData/datakeys2.d")
# item3 = joblib.load("ResultPicData/dataitem3.d")
# keys3 = joblib.load("ResultPicData/datakeys3.d")
# item4 = joblib.load("ResultPicData/dataitem4.d")
# keys4 = joblib.load("ResultPicData/datakeys4.d")
# item5 = joblib.load("ResultPicData/dataitem5.d")
# keys5 = joblib.load("ResultPicData/datakeys5.d")
# item6 = joblib.load("ResultPicData/dataitem6.d")
# keys6 = joblib.load("ResultPicData/datakeys6.d")
# item7 = joblib.load("ResultPicData/dataitem7.d")
# keys7 = joblib.load("ResultPicData/datakeys7.d")
item1 = joblib.load("featuredataitem.d")
keys1 = joblib.load("featuredatakeys.d")
item3 = joblib.load("featureitem15000.d")
keys3 = joblib.load("featurekeys15000.d")
item4 = joblib.load("featureitem10000.d")
keys4 = joblib.load("featurekeys10000.d")
item5 = joblib.load("featureitem5000.d")
keys5 = joblib.load("featurekeys5000.d")
item2 = joblib.load("hashdataitem.d")
keys2 = joblib.load("hashdatakeys.d")
# keys=keys1+keys2+keys3+keys4+keys5+keys6+keys7
keys=keys1+['NB (hash trick)']+['NB (feature20000)']+['NB (feature15000)']+['NB (feature10000)']+['NB (feature5000)']
###############################################################################
# Plot results
# 绘制结果
# ------------


def plot_accuracy(x, y, x_legend):
    """Plot accuracy as a function of x."""
    x = np.array(x)
    y = np.array(y)
    plt.title('Classification accuracy')
    plt.xlabel('%s' % x_legend)
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.plot(x, y)



def drawresults():
    rcParams['legend.fontsize'] = 10
    cls_names = keys

    # Plot accuracy evolution 绘制准确性演变情况
    plt.figure()
    for _, stats in item1:
        # Plot accuracy evolution with #examples 用#examples绘制准确性演变图
        accuracy, n_examples = zip(*stats['accuracy_history'])
        plot_accuracy(n_examples, accuracy, "Number of training examples")
        # plot_accuracy(n_examples, accuracy, "Number of training examples")
        ax = plt.gca()
        ax.set_ylim((0.70, 1))
    for _, stats in item2:
        # Plot accuracy evolution with #examples 用#examples绘制准确性演变图
        accuracy, n_examples = zip(*stats['accuracy_history'])
        plot_accuracy(n_examples, accuracy, "Number of training examples")
        # plot_accuracy(n_examples, accuracy, "Number of training examples")
        ax = plt.gca()
        ax.set_ylim((0.70, 1))
    for _, stats in item3:
        # Plot accuracy evolution with #examples 用#examples绘制准确性演变图
        accuracy, n_examples = zip(*stats['accuracy_history'])
        plot_accuracy(n_examples, accuracy, "Number of training examples")
        # plot_accuracy(n_examples, accuracy, "Number of training examples")
        ax = plt.gca()
        ax.set_ylim((0.70, 1))
    for _, stats in item4:
        # Plot accuracy evolution with #examples 用#examples绘制准确性演变图
        accuracy, n_examples = zip(*stats['accuracy_history'])
        plot_accuracy(n_examples, accuracy, "Number of training examples")
        # plot_accuracy(n_examples, accuracy, "Number of training examples")
        ax = plt.gca()
        ax.set_ylim((0.70, 1))
    for _, stats in item5:
        # Plot accuracy evolution with #examples 用#examples绘制准确性演变图
        accuracy, n_examples = zip(*stats['accuracy_history'])
        plot_accuracy(n_examples, accuracy, "Number of training examples")
        # plot_accuracy(n_examples, accuracy, "Number of training examples")
        ax = plt.gca()
        ax.set_ylim((0.80, 1))

    # for _, stats in item1:
    #     # Plot accuracy evolution with #examples 用#examples绘制准确性演变图
    #     accuracy, n_examples = zip(*stats['accuracy_history'])
    #     plot_accuracy(n_examples, accuracy, "Number of training examples")
    #     # plot_accuracy(n_examples, accuracy, "Number of training examples")
    #     ax = plt.gca()
    #     ax.set_ylim((0.80, 1))
    #
    # for _, stats in item2:
    #     # Plot accuracy evolution with #examples 用#examples绘制准确性演变图
    #     accuracy, n_examples = zip(*stats['accuracy_history'])
    #     plot_accuracy(n_examples, accuracy, "Number of training examples")
    #     # plot_accuracy(n_examples, accuracy, "Number of training examples")
    #     ax = plt.gca()
    #     ax.set_ylim((0.80, 1))
    #
    # for _, stats in item3:
    #     # Plot accuracy evolution with #examples 用#examples绘制准确性演变图
    #     accuracy, n_examples = zip(*stats['accuracy_history'])
    #     plot_accuracy(n_examples, accuracy, "Number of training examples")
    #     # plot_accuracy(n_examples, accuracy, "Number of training examples")
    #     ax = plt.gca()
    #     ax.set_ylim((0.80, 1))
    #
    # for _, stats in item4:
    #     # Plot accuracy evolution with #examples 用#examples绘制准确性演变图
    #     accuracy, n_examples = zip(*stats['accuracy_history'])
    #     plot_accuracy(n_examples, accuracy, "Number of training examples")
    #     # plot_accuracy(n_examples, accuracy, "Number of training examples")
    #     ax = plt.gca()
    #     ax.set_ylim((0.80, 1))
    #
    # for _, stats in item5:
    #     # Plot accuracy evolution with #examples 用#examples绘制准确性演变图
    #     accuracy, n_examples = zip(*stats['accuracy_history'])
    #     plot_accuracy(n_examples, accuracy, "Number of training examples")
    #     # plot_accuracy(n_examples, accuracy, "Number of training examples")
    #     ax = plt.gca()
    #     ax.set_ylim((0.80, 1))
    #
    # for _, stats in item6:
    #     # Plot accuracy evolution with #examples 用#examples绘制准确性演变图
    #     accuracy, n_examples = zip(*stats['accuracy_history'])
    #     plot_accuracy(n_examples, accuracy, "Number of training examples")
    #     # plot_accuracy(n_examples, accuracy, "Number of training examples")
    #     ax = plt.gca()
    #     ax.set_ylim((0.80, 1))
    #
    # for _, stats in item7:
    #     # Plot accuracy evolution with #examples 用#examples绘制准确性演变图
    #     accuracy, n_examples = zip(*stats['accuracy_history'])
    #     plot_accuracy(n_examples, accuracy, "Number of training examples")
    #     # plot_accuracy(n_examples, accuracy, "Number of training examples")
    #     ax = plt.gca()
    #     ax.set_ylim((0.80, 1))




    plt.legend(cls_names, loc='best')

    plt.show()


drawresults()
