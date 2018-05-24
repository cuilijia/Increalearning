from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?'
]
corpus2 = [
    'Is this first this this first this this the first document?',
]

ls=[ 'document', 'first', 'one', 'second']
vectorizer = CountVectorizer(stop_words=None,vocabulary=ls)
count = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(vectorizer.vocabulary_)
print(count.toarray())

test=vectorizer.vocabulary_
vectorizer2 = CountVectorizer(stop_words=None,vocabulary=test)
transformer = TfidfTransformer()
tfidf_matrix = transformer.fit_transform(count)
print(tfidf_matrix.toarray())
#

# print(tfidf_vec.get_feature_names())
# print(tfidf_vec.vocabulary_)
# print(tfidf_matrix.toarray())
count = vectorizer2.fit_transform(corpus)
print(vectorizer2.get_feature_names())

print(vectorizer2.vocabulary_)
print(count.toarray())

Vocubularysave = []
VocubularyList =vectorizer2.get_feature_names()
j = 0
for i in VocubularyList:
    # print(i,",",vectorizer2.vocabulary_[i],",",max(tfidf[vectorizer2.vocabulary_[i]]))
    if(j<100):
        Vocubularysave.append({"name":i,'numb':vectorizer2.vocabulary_[i]})
        j = j+1
    # print('-',end='')
print(len(Vocubularysave))

newV=Vocubularysave+Vocubularysave+[]

print(sorted(Vocubularysave,key=lambda x:x['numb'], reverse=True))

def sortbyword(one_list,size,word):
    '''''
    使用排序的方法
    '''
    result_list=[]
    result_listname = []
    temp_list=sorted(one_list,key=lambda x:x[word], reverse=True)
    i = 0
    j = 0
    while i<len(temp_list):
        print("  i=:",i)
        if temp_list[i][word] not in result_listname:
                result_list.append(temp_list[i])
                result_listname.append(temp_list[i][word])
                j+=1
                i+=1
                if(j>=size):
                    return result_list
        else:
            i+=1

    return result_list

print('newVbefore4:',newV)
print('newVafter4:',sortbyword(newV,2,'numb'))
xx=sortbyword(newV,3,'numb')
# xx=xx['name']
# print(xx)
#
# tfidf_vec = TfidfVectorizer()
# tfidf_matrix = tfidf_vec.fit_transform(corpus)
# X_chi2 = SelectKBest(chi2, k=5).fit_transform(tfidf_matrix, [1,2,3,4])
# print(X_chi2.toarray())
