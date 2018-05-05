
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

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.externals import joblib


data=[]#所有数据集

xtest=[]#测试集文本向量
ytest=[]#测试集类别

xtrain=[]#训练集文本向量
ytrain=[]#训练集类别

TrainDataSize = 8 #训练集个数

all_classes = np.arange(20) #分类器类别上限

printjumpsize=2 # 输出间隔

# 读入数据集 -------------------------------------------------------------------------------
def ReadData():
    i = 0
    data_path = os.path.join(get_data_home(), "FUDAN/answer")
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

ReadData()
# end 读入数据集 -------------------------------------------------------------------------------

# 实际参与训练的类型范围
Num_mintype=0
Num_maxtype=len(data)
type_start =Num_mintype
type_end = Num_maxtype
typerange = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
             10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

# 划分训练类别成为测试和训练样本集 ---------------------------------------------------------------
def getTRAINandTEST():

    for n in range(TrainDataSize):
        xtrain.append([])
        ytrain.append([])

    for j in range(type_start, type_end):
        for i in range(len(data[j])):
            if (i in range(0, int(len(data[j]) / (TrainDataSize + 1)))):
                xtest.append(data[j][i]['content'])
                ytest.append(data[j][i]['type'])
            for p in range(TrainDataSize):
                if (i in range(int(len(data[j]) / (TrainDataSize + 1) * (p + 1)),
                               int(len(data[j]) / (TrainDataSize + 1)) * (p + 2))):
                    xtrain[p].append(data[j][i]['content'])
                    ytrain[p].append(data[j][i]['type'])

getTRAINandTEST()
print('训练样本集 ',TrainDataSize,' 份')
print('测试样本集 ',1,' 份')
print("一份样本集为 %d 条  " % (len(ytest)))
# end 划分训练类别成为测试和训练样本集 ---------------------------------------------------------------


# test data statistics 测试数据的统计
test_stats = {'n_test': 0, 'n_test_pos': 0}

tick = time.time()
parsing_time = time.time() - tick
tick = time.time()


tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=1000,
                                   stop_words='none')
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=1000,
                                stop_words='english')
# 数据集文本向量化 (哈希技巧) -------------------------------------------------------
vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 18,
                               alternate_sign=False)
X_test = vectorizer.transform(xtest)
# end 数据集文本向量化 (哈希技巧) -------------------------------------------------------

vectorizing_time = time.time() - tick
test_stats['n_test'] += len(ytest)
test_stats['n_test_pos'] += sum(ytest)


def progress(cls_name, stats):
    """Report progress information, return a string.报告进度信息，返回一个字符串。"""
    duration = time.time() - stats['t0']
    s = "%20s 分类器 : \t" % cls_name
    s += "%(n_train)6d 条训练样本  " % stats
    s += "%(n_test)6d 条测试样本  " % test_stats
    s += "准确度: %(accuracy).3f " % stats
    s += "共计 %.2fs (%5d 样本/s)" % (duration, stats['n_train'] / duration)
    return s

# 这里有一些支持`partial_fit`方法的分类器
# 新创建分类器容器
partial_fit_classifiers = {
    'SGD': SGDClassifier(),
    'Perceptron': Perceptron(),
    'NB Multinomial': MultinomialNB(alpha=0.01),
    'Passive-Aggressive': PassiveAggressiveClassifier(),

}
# 载入旧的分类器容器
classifiers={
    'SGD': SGDClassifier(),
    'Perceptron': Perceptron(),
    'NB Multinomial': MultinomialNB(alpha=0.01),
    'Passive-Aggressive': PassiveAggressiveClassifier(),
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
        if os.path.exists("Train_Model_" + cls_name + ".m"):
            cls = joblib.load("Train_Model_" + cls_name + ".m")
        classifiers[cls_name]=cls

getclassifiers()

# end 获取以往保存下来的的模型------------------------------------------------------


# Main loop : iterate on mini-batches of examples
# 主循环：迭代小批量的例子-----------------------------------------------
def IncreasingFIT():
    global total_vect_time
    for i in range(TrainDataSize):
        tick = time.time()
        X_train = vectorizer.transform(xtrain[i])
        total_vect_time += time.time() - tick

        for cls_name, cls_useless in partial_fit_classifiers.items():
            cls = classifiers[cls_name]

            tick = time.time()
            # update estimator with examples in the current mini-batch
            # 使用当前最小批次中的示例更新估算器

            cls.partial_fit(X_train, ytrain[i], classes=all_classes)

            # accumulate test accuracy stats
            # 累积测试准确度统计
            cls_stats[cls_name]['total_fit_time'] += time.time() - tick
            cls_stats[cls_name]['n_train'] += X_train.shape[0]
            cls_stats[cls_name]['n_train_pos'] += sum(ytrain[i])
            tick = time.time()

            #测试准确性函数
            cls_stats[cls_name]['accuracy'] = cls.score(X_test, ytest)

            cls_stats[cls_name]['prediction_time'] = time.time() - tick
            acc_history = (cls_stats[cls_name]['accuracy'],
                           cls_stats[cls_name]['n_train'])
            cls_stats[cls_name]['accuracy_history'].append(acc_history)
            run_history = (cls_stats[cls_name]['accuracy'],
                           total_vect_time + cls_stats[cls_name]['total_fit_time'])
            cls_stats[cls_name]['runtime_history'].append(run_history)

            if i % printjumpsize == 0:
                print(progress(cls_name, cls_stats[cls_name]))
        if i % printjumpsize == 0:
            print('\n')

print('开始增量训练...')
IncreasingFIT()
print('已完成...')
# end 主循环：迭代小批量的例子-----------------------------------------------

# 保存训练好的模型------------------------------------------------------
def saveModel():
    for cls_name, cls_useless in partial_fit_classifiers.items():
        cls = classifiers[cls_name]
        joblib.dump(cls, "Train_Model_" + cls_name + ".m")

        # 预测函数
        # print(cls.predict(X_test))

saveModel()

# end 保存训练好的模型------------------------------------------------------

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
        ax.set_ylim((0.5, 1))
    plt.legend(cls_names, loc='best')

    plt.figure()
    for _, stats in sorted(cls_stats.items()):
        # Plot accuracy evolution with runtime 用运行时绘制准确度的演变
        accuracy, runtime = zip(*stats['runtime_history'])
        plot_accuracy(runtime, accuracy, 'runtime (s)')
        ax = plt.gca()
        ax.set_ylim((0.5, 1))
    plt.legend(cls_names, loc='best')

    # Plot fitting times 绘制拟合时间
    plt.figure()
    fig = plt.gcf()
    cls_runtime = []
    for cls_name, stats in sorted(cls_stats.items()):
        cls_runtime.append(stats['total_fit_time'])

    cls_runtime.append(total_vect_time)
    cls_names.append('Vectorization')
    bar_colors = ['b', 'g', 'r', 'c', 'm', 'y']

    ax = plt.subplot(111)
    rectangles = plt.bar(range(len(cls_names)), cls_runtime, width=0.5,
                         color=bar_colors)

    ax.set_xticks(np.linspace(0.25, len(cls_names) - 0.75, len(cls_names)))
    ax.set_xticklabels(cls_names, fontsize=10)
    ymax = max(cls_runtime) * 1.2
    ax.set_ylim((0, ymax))
    ax.set_ylabel('runtime (s)')
    ax.set_title('Training Times')

    def autolabel(rectangles):
        """attach some text vi autolabel on rectangles. 在矩形上附加一些文本vi autolabel。"""
        for rect in rectangles:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2.,
                    1.05 * height, '%.4f' % height,
                    ha='center', va='bottom')

    autolabel(rectangles)
    plt.show()

    # Plot prediction times 绘制预测时间
    plt.figure()
    cls_runtime = []
    cls_names = list(sorted(cls_stats.keys()))
    for cls_name, stats in sorted(cls_stats.items()):
        cls_runtime.append(stats['prediction_time'])
    cls_runtime.append(parsing_time)
    cls_names.append('Read/Parse\n+Feat.Extr.')
    cls_runtime.append(vectorizing_time)
    cls_names.append('Hashing\n+Vect.')

    ax = plt.subplot(111)
    rectangles = plt.bar(range(len(cls_names)), cls_runtime, width=0.5,
                         color=bar_colors)

    ax.set_xticks(np.linspace(0.25, len(cls_names) - 0.75, len(cls_names)))
    ax.set_xticklabels(cls_names, fontsize=8)
    plt.setp(plt.xticks()[1], rotation=30)
    ymax = max(cls_runtime) * 1.2
    ax.set_ylim((0, ymax))
    ax.set_ylabel('runtime (s)')
    ax.set_title('Prediction Times ')
    autolabel(rectangles)
    plt.show()


drawresults()


