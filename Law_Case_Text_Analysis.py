import pickle
# Import the following libraries - Pandas, Numpy, Goose3, re, nltk, sklearn
# Importing required libraries
import numpy as np
import pandas as pd
from goose3 import Goose
import re
from sklearn import metrics
from sklearn.metrics import *
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from scipy.sparse import csr_matrix, lil_matrix

nltk.download('stopwords')
lst_stopwords = nltk.corpus.stopwords.words("english")
lst_stopwords
nltk.download('omw-1.4')
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('wordnet')
data=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
dflabels = pd.read_json('Labels.jsonl',lines=True)
dfl = dflabels.drop(['text'], axis=1).T
dfl=dfl.T
dfl=dfl.values.tolist()
listdfl=[]
j=0
for i in dfl:
    listdfl.append(i[0])

from flask import Flask, jsonify, render_template, request, url_for
from werkzeug.utils import redirect


dftrain = pd.read_csv('TrainSet.csv')


# In[81]:


dftrainjson = pd.read_json('Train-Sent.jsonl',lines=True)


# In[82]:


dftrainjson


# In[83]:


dftrain = dftrainjson
dftrain = dftrain.drop(['sent_labels','factid'],axis=1)

dftest = pd.read_json('Test-Doc.jsonl',lines=True)


# In[87]:


dflabels


# In[88]:


dftesting = dftest
dftest = dftest.drop(['factid'],axis=1)


# In[89]:


dftest


# Function for processing the texts
def preprocessing_text(text, stem=False, lemma=True, lst_stopwords=None):
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

    ## Tokenization (convert from string to list)
    processed_text = text.split()

    ## remove Stopwords
    if lst_stopwords is not None:
        processed_text = [word for word in processed_text if word not in
                          lst_stopwords]

    ## Stemming (remove -ing, -ly, ...)
    if stem == True:
        ps = nltk.stem.porter.PorterStemmer()
        processed_text = [ps.stem(word) for word in processed_text]

    ## Lemmatisation (convert the word into root word)
    if lemma == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        processed_text = [lem.lemmatize(word) for word in processed_text]

    ## back to string from list
    text = " ".join(processed_text)
    return text

dftrain['text_clean'] = dftrain['text'].apply(lambda x:preprocessing_text(x, stem=False, lemma=True, lst_stopwords=lst_stopwords))
#dftrain['label_clean'] = dftrain['doc_labels'].apply(lambda x:preprocessing_text(x, stem=False, lemma=True))
dftest['text_clean'] = dftest['text'].apply(lambda x:preprocessing_text(x, stem=False, lemma=True,lst_stopwords=lst_stopwords))


# In[93]:


dftrain


# In[94]:


dftrain['doc_labels'][0][0]


# In[95]:


labels_x,labels_y=dflabels.shape


# In[96]:


labels_x=labels_x-1


# In[97]:


labels_x


# In[98]:
dff = pd.DataFrame([data],columns=listdfl)
dff = dff.drop([0],axis=0)


listfreq = []
h=0
p=0
listdev=[]
for i in dftrain['doc_labels']:
    dfff = pd.DataFrame([data],columns=listdfl)
    dff = pd.concat([dff, dfff], ignore_index = True)
    dfff.reset_index()
    k=0
    for j in i:
        b=None
        for b in listdfl:
            if(j==b):
                k=1
        if(k==1):
            dff[j][h]=1
            k=0
        else:
            p=p+1
            listdev.append(j)
    h=h+1


# In[112]:


dff


# In[113]:


p


# In[114]:


listdev


# In[115]:


p


# In[116]:


dftrainfinal = pd.concat([dftrain,dff], axis=1)


# In[117]:


dftrainfinal


# In[118]:


dftest


# In[119]:


#everything was fine before this line


# In[120]:


dfft = pd.DataFrame([data],columns=listdfl)
dfft = dfft.drop([0],axis=0)


# In[121]:


dfft


# In[122]:


listdfl


# In[123]:


dftest


# In[124]:


listfreqt = []
h=0
p=0
listdevt=[]
j=None
for i in dftest['doc_labels']:
    dfff2 = pd.DataFrame([data],columns=listdfl)
    dfft = pd.concat([dfft, dfff2], ignore_index = True)
    dfff2.reset_index()
    k=0
    for j in i:
        b=None
        for b in listdfl:
            if(j==b):
                k=1
        if(k==1):
            dfft[j][h]=1
            k=0
        else:
            p=p+1
            listdevt.append(j)
    h=h+1


# In[ ]:





# In[125]:


dfft


# In[126]:


dftestfinal = pd.concat([dftest,dfft], axis=1)


# In[127]:


dftestfinal


# In[128]:


dfall = pd.concat([dftrainfinal,dftestfinal],join="outer",
    ignore_index=True)


# In[129]:


dftrainfinal


# In[130]:


dftestfinal


# In[131]:


dfall


# In[132]:


from sklearn.model_selection import train_test_split
dfally = dfall.drop(['text','doc_labels','text_clean'], axis=1)
dfally
#train, test = train_test_split(dfall['text_clean'],dfall, random_state=42, test_size=0.20, shuffle=True)


# In[133]:


X_train, X_test, Y_train, Y_test = train_test_split(dfall['text_clean'],dfally, random_state=42, test_size=0.20, shuffle=True)




filename = 'finalized_model_LAW.sav'
gb = pickle.load(open(filename, 'rb'))



#X_train, X_test, Y_train, Y_test = train_test_split(dfall['text_clean'],dfally, random_state=42, test_size=0.20, shuffle=True)


#Using Tfidf Vectorizer to form word vectors
from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,5), norm='l2')
Trainvectf = v.fit_transform(X_train)
testvectf = v.transform(X_test)

from goose3 import Goose

app = Flask(__name__, template_folder="C:\\Users\\akash")
hihi=None
hihi = []
print('Yo?')
@app.route('/', methods=['POST'])
def sm():
 # Put article link in front of url in '' single quotes
 if request.method == 'POST':
    h = request.json
    print("hr", h)
    url = h
    # url = 'PUT YOUR LINK HERE'
    trying = h
    better = preprocessing_text(trying, stem=False, lemma=True)
    b = pd.Series(better)
    hh = v.transform(b)
    # predicting the bias of the article towards or against a particular party
    g = gb.predict(hh)
    dd = {}
    i = None
    j = None
    # print(g.size())
    fn = []
    for j in range(0, np.size(g)):
        if (g[0][j] == 1):
            print(listdfl[j])
            fn.append(listdfl[j])
    dd['ans'] = fn
    print('Yo????')
    print(dd)
    global hihi
    hihi = fn
    print(hihi)
    print("POST TRYNA HATE ON US")
    print(dd)
    return redirect(url_for('dt'))
    #return '<h1>The predicted bias is <u></u></h1>'
    return dd
 else:
     print("BHAKKA")
     hihi=0
     return redirect(url_for('dt', ff=hihi))

@app.route('/hi', methods=['GET','POST'])
def dt():
    global b
    d={}
    print(hihi)
    d['ans']= hihi
    return d

if __name__ == '__main__':
    app.run()
