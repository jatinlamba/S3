# importing all the models required

import matplotlib as inline
import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
print(string.__file__)
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
from wordcloud import WordCloud, STOPWORDS
import matplotlib as mpl

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from nltk.corpus import stopwords
from sklearn.cross_validation import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import recall_score

# Reading the data from the Sqlite file
# The dataset can be downloaded from this location :https://snap.stanford.edu/data/web-FineFoods.html3

sqlobject = sqlite3.connect('C:/Users/Jets/Downloads/amazon-fine-foods-release-2016-01-08-20-34-54/amazon-fine-foods/database.sqlite')

reviews = pd.read_sql_query("""SELECT Score, Summary FROM Reviews""", sqlobject)

originalRev = reviews.copy()
reviews = originalRev.copy()
print(reviews.shape)
reviews = reviews.dropna()

print(reviews.shape)
print (reviews.head(25))

# Encoding score to Positive or negative based on value of each sample

scores = reviews['Score']
reviews['Score'] = reviews['Score'].apply(lambda x : 'pos' if x > 3 else 'neg')
scores.mean()

# Distribution of labels in the dataset

reviews.groupby('Score')['Summary'].count()
reviews.groupby('Score')['Summary'].count().plot(kind='bar',color=['orangered','lime'],title='Label Distribution',figsize=(10,6))
plt.show()

print ('Negative reviews percentage %.2f %%' % ((reviews.groupby('Score')['Summary'].count()['neg'])*100.0/len(reviews)))
print ('Positive reviews percentage %.2f %%' % ((reviews.groupby('Score')['Summary'].count()['pos'])*100.0/len(reviews)))

# Splitting the dataset based on labels

def distPosNeg(Summaries):
    neg = reviews.loc[Summaries['Score']=='neg']
    pos = reviews.loc[Summaries['Score']=='pos']
    return [pos,neg]

[pos,neg] = distPosNeg(reviews)

# Preprocessing

lemmatizer = nltk.WordNetLemmatizer()
stop = stopwords.words('english')
translation = str.maketrans(string.punctuation,' '*len(string.punctuation))

def preProcessing(line):
    tokens=[]
    line = line.translate(translation)
    line = nltk.word_tokenize(line.lower())
    for j in line:
        stemmer = lemmatizer.lemmatize(j)
        tokens.append(stemmer)
    return ' '.join(tokens)

positiveData = []
negativeData = []
for p in pos['Summary']:
    positiveData.append(preProcessing(p))

for n in neg['Summary']:
    negativeData.append(preProcessing(n))

data = positiveData + negativeData
labels = np.concatenate((pos['Score'].values,neg['Score'].values))

# Splitting the data into train and test using Hold-out evaluation

[dataTrain,dataTest,trainLabels,testLabels] = train_test_split(data,labels , test_size=0.25, random_state=20160121,stratify=labels)

# tokenizing the training data to find frequency of words

j = []
for line in dataTrain:
    s = nltk.word_tokenize(line)
    for l in s:
        j.append(l)

wordFeatures = nltk.FreqDist(j)
print (len(wordFeatures))

# Feature Reduction
# Using PCA

vectorAll = CountVectorizer()
ctr_featuresAll = vectorAll.fit_transform(dataTrain)

tf_vectorAll = TfidfTransformer()
tr_featuresAll = tf_vectorAll.fit_transform(ctr_featuresAll)

cte_featuresAll = vectorAll.transform(dataTest)
te_featuresAll = tf_vectorAll.transform(cte_featuresAll)

svd = TruncatedSVD(n_components=200)
tr_features_truncated = svd.fit_transform(tr_featuresAll)

te_features_truncated = svd.transform(te_featuresAll)

svd = TruncatedSVD(n_components=200)
ctr_features_truncated = svd.fit_transform(ctr_featuresAll)
cte_features_truncated = svd.transform(cte_featuresAll)

# Running classification algorithms

models = {'BernoulliNB':BernoulliNB(binarize=0.5)
          ,'Logistic' : linear_model.LogisticRegression(C=1e5),'Decision Tree' : DecisionTreeClassifier(random_state=20160121, criterion='entropy')}

results_svd = pd.DataFrame()

foldnum = 0
tfprediction = {}
cprediction = {}
for name, model in models.items():
    model.fit(tr_features_truncated, trainLabels)
    tfprediction[name] = model.predict(te_features_truncated)
    tfaccuracy = metrics.accuracy_score(tfprediction[name], testLabels)
    results_svd.loc[foldnum, 'Model'] = name
    results_svd.loc[foldnum, 'TF-IDF Accuracy'] = tfaccuracy
    foldnum = foldnum + 1
print (results_svd)

for name,model in models.items():
    print ("Classification report for ",name)
    print(metrics.classification_report(testLabels, tfprediction[name]))
    print("\n")

results_svd.plot(kind='bar',color=['salmon','dimgrey','gold'],title='Model Performance',figsize=(10,6),x='Model',legend=False)
plt.show()

# comparing the accuracies of negative samples

negcom = pd.DataFrame()
for name,model in models.items():
    p= recall_score(testLabels,tfprediction[name],pos_label='neg')
    negcom.loc[name,'recall'] = p

pl = negcom.plot(kind='bar',color=['salmon','dimgrey','gold'],title='Using PCA',figsize=(10,6),legend=False)
pl.set(xlabel='Models',ylabel='Recall for negative samples')
lim = plt.ylim([0,1])
plt.show()

# Feature Selection

topwords = [fpair[0] for fpair in list(wordFeatures.most_common(5000))]

print (wordFeatures.most_common(25))

wordCnt = pd.DataFrame(wordFeatures.most_common(25),columns=['words','count'])

print (wordCnt)

wordCnt.plot(kind='bar',color= ['red'],x=wordCnt['words'],legend=False,title='Top 20 most freq words',figsize=(10,6))
plt.show()

vec = CountVectorizer()
c_fit = vec.fit_transform([' '.join(topwords)])

tf_vec = TfidfTransformer()
tf_fit = tf_vec.fit_transform(c_fit)

ctr_features = vec.transform(dataTrain)
tr_features = tf_vec.transform(ctr_features)

cte_features = vec.transform(dataTest)
te_features = tf_vec.transform(cte_features)

# Running classification algorithms

models = {'BernoulliNB': BernoulliNB()
    , 'Logistic': linear_model.LogisticRegression(C=1e5),
          'Decision Tree': DecisionTreeClassifier(random_state=20160121, criterion='entropy')}
results = pd.DataFrame()

foldnum = 0
tfprediction = {}
cprediction = {}
for name, model in models.items():
    model.fit(tr_features, trainLabels)
    tfprediction[name] = model.predict(te_features)
    tfaccuracy = metrics.accuracy_score(tfprediction[name], testLabels)
    results.loc[foldnum, 'TF-IDF Accuracy'] = tfaccuracy
    results.loc[foldnum, 'Model'] = name
    foldnum = foldnum + 1
print(results)

for name,model in models.items():
    print ("Classification report for ",name)
    print(metrics.classification_report(testLabels, tfprediction[name]))
    print("\n")

results.plot(kind='bar',color=['salmon','dimgrey','gold'],title='Model Performance',figsize=(10,6),x='Model',legend=False)
plt.show()

# comparing the accuracies of negative samples

negcom = pd.DataFrame()
for name,model in models.items():
    p= recall_score(testLabels,tfprediction[name],pos_label='neg')
    negcom.loc[name,'recall'] = p

pl = negcom.plot(kind='bar',color=['salmon','dimgrey','gold'],title='Most Freq Features',figsize=(10,6),legend=False)
pl.set(xlabel='Models',ylabel='Recall')
lim = plt.ylim([0,1])
plt.show()

# Running algorithms on our entire feature set without feature reduction/ selection.
# Unigram

# using count vectorizer for generation of count vectors for the training dataset.

vectorAll = CountVectorizer()
ctr_featuresAll = vectorAll.fit_transform(dataTrain)

tf_vectorAll = TfidfTransformer()
tr_featuresAll = tf_vectorAll.fit_transform(ctr_featuresAll)

cte_features_all = vectorAll.transform(dataTest)
te_features_all = tf_vectorAll.transform(cte_features_all)

# Running classification algorithms

models = {'BernoulliNB': BernoulliNB()
    , 'Logistic': linear_model.LogisticRegression(C=1e5),
          'Decision Tree': DecisionTreeClassifier(random_state=20160121, criterion='entropy')}

results_all_uni = pd.DataFrame()


foldnum = 0
tfprediction = {}
cprediction = {}
for name, model in models.items():
    model.fit(tr_featuresAll, trainLabels)
    tfprediction[name] = model.predict(te_features_all)
    tfaccuracy = metrics.accuracy_score(tfprediction[name], testLabels)

    results_all_uni.loc[foldnum, 'TF-IDF Accuracy'] = tfaccuracy
    results_all_uni.loc[foldnum, 'Model'] = name
    foldnum = foldnum + 1
print(results_all_uni)

for name,model in models.items():
    print ("Classification report for ",name)
    print(metrics.classification_report(testLabels, tfprediction[name]))

results_all_uni.plot(kind='bar',color=['salmon','dimgrey','gold'],title='Model Performance',figsize=(10,6),x='Model',legend=False)
plt.show()

# Comparing the accuracies of negative samples

negcom = pd.DataFrame()
for name,model in models.items():
    p= recall_score(testLabels,tfprediction[name],pos_label='neg')
    negcom.loc[name,'recall'] = p

pl = negcom.plot(kind='bar',color=['salmon','dimgrey','gold'],title='Unigram',figsize=(10,6),legend=False)
pl.set(xlabel='Models',ylabel='Recall for negative samples')
lim = plt.ylim([0,1])
plt.show()


# Using Bigrams
# Using count vectorizer to generate count vectors for the training data.

vectorAll = CountVectorizer(ngram_range=(1,2))
ctr_featuresAll = vectorAll.fit_transform(dataTrain)

tf_vec_all = TfidfTransformer()
tr_features_all = tf_vec_all.fit_transform(ctr_featuresAll)

cte_features_all = vectorAll.transform(dataTest)
te_features_all = tf_vec_all.transform(cte_features_all)

print ((ctr_featuresAll.shape))

results_all_bi = pd.DataFrame()

tfprediction = {}
cprediction = {}
foldnum = 0
for name, model in models.items():
    model.fit(tr_features_all, trainLabels)
    tfprediction[name] = model.predict(te_features_all)
    tfaccuracy = metrics.accuracy_score(tfprediction[name], testLabels)

    results_all_bi.loc[foldnum, 'TF-IDF Accuracy'] = tfaccuracy
    results_all_bi.loc[foldnum, 'Model'] = name
    foldnum = foldnum + 1
print(results_all_bi)

for name,model in models.items():
    print ("Classification report for ",name)
    print(metrics.classification_report(testLabels, tfprediction[name]))
    print("\n")

results_all_bi.plot(kind='bar',color=['salmon','dimgrey','gold'],title='Model Performance',figsize=(10,6),x='Model',legend=False)
plt.show()

# Comparing the accuracies of negative samples

negcom = pd.DataFrame()
for name,model in models.items():
    p= recall_score(testLabels,tfprediction[name],pos_label='neg')
    negcom.loc[name,'recall'] = p

pl = negcom.plot(kind='bar',color=['salmon','dimgrey','gold'],title='Bigram',figsize=(10,6),legend=False)
pl.set(xlabel='Models',ylabel='Recall for negative samples')
lim = plt.ylim([0,1])
plt.show()

# Using Trigrams
# Using countvectorizer to generate count vectors for the training data.

vectorAll = CountVectorizer(ngram_range=(1, 3))
ctr_featuresAll = vectorAll.fit_transform(dataTrain)

tf_vec_all = TfidfTransformer()
tr_features_all = tf_vec_all.fit_transform(ctr_featuresAll)

cte_features_all = vectorAll.transform(dataTest)
te_features_all = tf_vec_all.transform(cte_features_all)

results_all_tri = pd.DataFrame()

tfprediction = {}
cprediction = {}
foldnum = 0
for name, model in models.items():
    model.fit(tr_features_all, trainLabels)
    tfprediction[name] = model.predict(te_features_all)
    tfaccuracy = metrics.accuracy_score(tfprediction[name], testLabels)
    results_all_tri.loc[foldnum, 'TF-IDF Accuracy'] = tfaccuracy
    results_all_tri.loc[foldnum, 'Model'] = name
    foldnum = foldnum + 1
print(results_all_tri)

results_all_tri.plot(kind='bar',color=['salmon','dimgrey','gold'],title='Model Performance',figsize=(10,6),x='Model',legend=False)
plt.show()

for name,model in models.items():
    print ("Classification report for ",name)
    print(metrics.classification_report(testLabels, tfprediction[name]))
    print("\n")

# Comparing the accuracies of negative samples

negcom = pd.DataFrame()
for name,model in models.items():
    p= recall_score(testLabels,tfprediction[name],pos_label='neg')
    negcom.loc[name,'recall'] = p

pl = negcom.plot(kind='bar',color=['salmon','dimgrey','gold'],title='Trigram',figsize=(10,6),legend=False)
pl.set(xlabel='Models',ylabel='Recall for negative samples')
lim = plt.ylim([0,1])
plt.show()

def plotConfusionMatrix(cm, title='Confusion matrix', cmap=plt.cm.Greens):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(set(labels)))
    plt.xticks(tick_marks, ['neg', 'pos'], rotation=45)
    plt.yticks(tick_marks, ['neg', 'pos'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Computing confusion matrix and normalized confusion matrix

cm = confusion_matrix(testLabels, tfprediction['Logistic'])
np.set_printoptions(precision=2)
plt.figure()
plotConfusionMatrix(cm)

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure()
plotConfusionMatrix(cm_normalized, title='Normalized confusion matrix')

plt.show()


# Word cloud

stopwords = set(STOPWORDS)

mpl.rcParams['font.size'] = 12
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['figure.subplot.bottom'] = .1


def show_wordcloud(data, title=None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40,
        scale=3,
        random_state=1
    ).generate(str(data))

    fig = plt.figure(1, figsize=(8, 8))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

wordCld = pd.DataFrame(wordFeatures.most_common(5000),columns=['words','count'])
show_wordcloud(wordCld['words'])

