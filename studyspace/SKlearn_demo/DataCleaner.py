# 数据集加工
# 消去空格 回车,消去停用词
# 进行中文分词

from sklearn.datasets import get_data_home
import os.path
from glob import glob
import jieba

data = []
testdata = []
traindata = []
stopkey=[line.strip() for line in open('stopwords').readlines()]
print(stopkey.__len__())



i = 0
data_path = os.path.join(get_data_home(), "FUDAN/train")
for docname in glob(os.path.join(data_path, "*")):
    doc = []
    for filename in glob(os.path.join(docname, "*.txt")):
        # print(filename)

        s = open(filename, 'r', encoding='gb18030', errors='ignore')
        context = s.read()

        # 消去空格和回车
        for i in range(stopkey.__len__()):
            context = context.replace(stopkey[i], '')
        context = context.replace('\r', '')
        context = context.replace("\n", "")

        # 中文分词
        word = jieba.cut(context, cut_all=False)

        context = " ".join(word)
        # print(context)
        s.close()
        s = open(filename, 'w', encoding='gb18030', errors='ignore')
        s.write(context)
        s.close()
        # print("-", end=' ')

    print("finish! ", docname)