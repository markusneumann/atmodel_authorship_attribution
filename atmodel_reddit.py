
# coding: utf-8

# # Authorship detection with the author-topic model
# 
# Links:  <br />
# [Gensim](https://radimrehurek.com/gensim/index.html)  <br />
# [Gensim author-topic model help page](https://radimrehurek.com/gensim/models/atmodel.html)  <br />
# [Gensim author-topic model tutorial](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/atmodel_tutorial.ipynb)
# 
# Relevant papers:  <br />
# [Blei et al. 2003](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)  (LDA) <br />
# [Rosen-Zvi et al. 2004](https://mimno.infosci.cornell.edu/info6150/readings/398.pdf) (Author-topic models) <br />
# [Rosen-Zvi et al. 2010](https://www.researchgate.net/profile/Michal_Rosen-Zvi/publication/220515711_Learning_author-topic_models_from_text_corpora/links/53fb31000cf27c365cf07efd.pdf) (Author-topic models extension) <br />
# [Seroussi et al. 2011](http://aclweb.org/anthology/W11-0321) (Authorship attribution with LDA) <br />
# [Seroussi et al. 2012](http://anthology.aclweb.org/P/P12/P12-2.pdf#page=292) (Authorship attribution with author-topic models)

# ![alt text](http://img.blog.csdn.net/20170417124825166?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGFmZWVkZmg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

# Plan for this presentation:
# 
# - Read in Reddit data
# - Preprocessing
# - Get a document to author dictionary
# - Transform that dictionary to author to document format
# - Create a 'test set' by anonymizing the author of randomly selected posts (so that each of those documents has a unique author with no other posts)
# - Run an author-topic model with Gensim
# - Get the topic distribution for each author
# - Get the topic distribution of the anonymized posts
# - Use the Hellinger Distance to find the author closest to the anonymized author unique to a test document
# - Check predictive accuracy
# - Compare results to SVM

# ## Reading in the reddit data
# 
# The reddit data consists of .json files for each month, which are compressed as .bz2. First, we find all the .bz2 files in a folder.

# In[2]:

import pandas as pd


# In[3]:

import bz2
import glob
files = glob.glob("data/*.bz2")
files


# For the purposes of this tutorial, we only use one. The file is read in, and the .json file it contains is directly transformed into a pandas data frame.

# In[ ]:

with bz2.open(files[0], 'rt') as f:
    text = f.read()
    
df = pd.read_json(text,  lines=True)


# Due to the size of the data, this can take a lot of time and RAM. Consequently we save the data as .csv file, which can be re-loaded more easily.

# In[7]:

#write to csv
df.to_csv('data/reddit2010-06.csv')


# In[3]:

#load data
df = pd.read_csv('data/reddit2010-06.csv')
#Note: this raises a warning, but setting low_memory=False as recommended crashes Jupyter


# Subset the data to include only what we need.

# In[4]:

#need only author body and subreddit variables
df = df[['author', 'body', 'subreddit']]

#restrict to gaming subreddit
df = df[df['subreddit']=='gaming']

#retain only posts with more than 300 characters
df = df[df['body'].apply(len, )>300]

#remove [deleted] authors
df = df[df['author']!='[deleted]']


# Authors who only have one post can't be predicted, so we will remove them from the dataset.

# In[5]:

#count the number of posts per author (like table() in R)
author_counts = df.author.value_counts()

#remove authors who only posted once
authors = author_counts[author_counts!=1]

#get the axis labels (i.e. the author names) and turn them into a list
authors = authors.axes[0].tolist()

#subset the dataframe
df = df[df['author'].isin(authors)]


# In[6]:

df.shape


# In[7]:

df.head()


# In[8]:

#save (and possibly load) the data again, small enough to fit on GitHub
df.to_csv('data/reddit2010-06_subset.csv')
df = pd.read_csv('data/reddit2010-06_subset.csv')


# ## Preprocessing
# 
# This section is basically just a copy of the [Gensim tutorial](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/atmodel_tutorial.ipynb), with some minor modification to fit data as a pandas dataframe instead of the format used by the author.

# In[8]:

import spacy
nlp = spacy.load('en')


# In[9]:

docs = []    
for doc in nlp.pipe(df['body'], n_threads=11, batch_size=100):
    # Process document using Spacy NLP pipeline.
    
    ents = doc.ents  # Named entities.

    # Keep only words (no numbers, no punctuation).
    # Lemmatize tokens, remove punctuation and remove stopwords.
    doc = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

    # Remove common words from a stopword list.
    #doc = [token for token in doc if token not in STOPWORDS]

    # Add named entities, but only if they are a compound of more than word.
    doc.extend([str(entity) for entity in ents if len(entity) > 1])
    
    docs.append(doc)


# In[12]:

# Compute bigrams.
from gensim.models import Phrases
# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
bigram = Phrases(docs, min_count=20)
for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)


# In[13]:

# Create a dictionary representation of the documents, and filter out frequent and rare words.

from gensim.corpora import Dictionary
dictionary = Dictionary(docs)

# Remove rare and common tokens.
# Filter out words that occur too frequently or too rarely.
max_freq = 0.5
min_wordcount = 20
dictionary.filter_extremes(no_below=min_wordcount, no_above=max_freq)

_ = dictionary[0]  # This sort of "initializes" dictionary.id2token.


# In[14]:

# Vectorize data.

# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in docs]


# Mapping from documents to authors, which is different for this dataset (compared to the Gensim tutorial).

# In[15]:

df['docID'] = range(0,len(df['body']))


# In[16]:

doc2author = pd.Series(df.author.values, index=df.docID).to_dict()


# Let's inspect the dimensionality of our data.

# In[18]:

print('Number of authors: %d' % len(pd.Series.unique(df['author'])))
print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))


# ## Training the author-topic model

# An author-topic model is normally run as following in Gensim.

# In[19]:

from gensim.models import AuthorTopicModel


# In[20]:

#model = AuthorTopicModel(corpus=corpus, num_topics=100, id2word=dictionary.id2token, \
#                    doc2author=doc2author, chunksize=100, passes=100, gamma_threshold=0.001, \
#                    eval_every=0, iterations=1, random_state=1)


# The information about which author wrote which document is stored in the doc2author (or alternatively, in the author2doc) object. To create a test set, we replace randomly sampled names with a numbered identifier.

# In[21]:

import random
random.seed(1)
#size of the test set is a fifth of the dataset
n_test = round(len(corpus)/5)
#randomly sample indices to be 'anonymized' and tested
sample_indices = random.sample( doc2author.keys(), n_test )
#create new dictionary
doc2author_test = doc2author.copy()
#randomly replace entries
for k in (sample_indices):
    doc2author_test[k] = 'test_id_' + str(k)


# Unfortunately the AT model in Gensim doesn't seem to work correctly with doc2author. Re-mapping the dictionary to author2doc instead:

# In[22]:

#re-map test dictionary
import collections
author2doc_test = collections.defaultdict(set)
for k, v in doc2author_test.items():
    author2doc_test[v].add(k)
author2doc_test


# Again, the author-topic model, but this time with randomly selected names anonymized.
# 
# Parameters:
# 
# **num_topics**: The number of topics ion the model. There is no 'correct' value here, and it depends entirely on how many different topics occur in the corpus. 100 is generally a reasonable compromise for a corpus this size.  <br />
# **chunksize**: Controls the size of the mini-batches. This depends entirely on the corpus - 2000 is the default, but this obviously makes no sense if a corpus only contains 1000 documents. <br />
# **passes**: 100 by default <br />
# **iterations**: iterations is the maximum number of times the model loops over each document <br />
# **alpha**: Can be set to 'asymmetric' <br />
# **eta**: Can be set to ‘auto’, which learns an asymmetric prior over words directly from the data

# In[23]:

#get_ipython().run_cell_magic('time', '', 'model = AuthorTopicModel(corpus=corpus, num_topics=100, id2word=dictionary.id2token, \\\n                    author2doc=author2doc_test, chunksize=100, passes=100, gamma_threshold=0.001, \\\n                    eval_every=0, iterations=1, random_state=1)')

model = AuthorTopicModel(corpus=corpus, num_topics=100, id2word=dictionary.id2token, \
                    author2doc=author2doc_test, chunksize=100, passes=100, gamma_threshold=0.001, \
                    eval_every=0, iterations=1, random_state=1)

# In[24]:

# Save model. 
model.save('./results/model_presentation.atmodel')


# In[18]:

#import pandas as pd
#import spacy
#from gensim.models import Phrases
#from gensim.corpora import Dictionary
#from gensim.models import AuthorTopicModel

#Load model
model = AuthorTopicModel.load('./results/model_presentation.atmodel')


# ### Results

# In[25]:

model.top_topics(corpus)


# ## Authorship attribution
# 
# ### Hellinger Distance
# 
# $$
# D(\theta_1, \theta_2) = \frac{1}{\sqrt{2}} \sqrt{\sum_{t=1}^T (\sqrt{\theta_{1,t}} - \sqrt{\theta_{2,t}})^2}
# $$
# 
# where  <br />
# $\theta_i$ is a T-dimensional multinomial topic distribution  <br />
# $\theta_{i,t}$ is the probability of the t-th topic
# 
# Predict authors by making a 'fake' author for the test documents, and then compare that author's topic distribution to those of the real authors via the Hellinger Distance:

# In[26]:

from gensim.similarities import MatrixSimilarity


# Functions to calculate Hellinger distance. Mostly taken from the Gensim AT tutorial, but there are some modifications to make this work for prediction.

# In[27]:

import re
from gensim import matutils

# Make a list of all the author-topic distributions.
author_vecs = [model.get_author_topics(author) for author in model.id2author.values()]

def similarity(vec1, vec2):
    '''Get similarity between two vectors'''
    dist = matutils.hellinger(matutils.sparse2full(vec1, model.num_topics),                               matutils.sparse2full(vec2, model.num_topics))
    sim = 1.0 / (1.0 + dist)
    return sim

def get_sims(vec):
    '''Get similarity of vector to all authors.'''
    sims = [similarity(vec, vec2) for vec2 in author_vecs]
    return sims

def get_bestmatch(name, top_n=10, smallest_author=1):
    '''
    Get table with similarities, author names, and author sizes.
    Return `top_n` authors as a dataframe.
    
    '''
    
    # Get similarities.
    sims = get_sims(model.get_author_topics(name))

    # Arrange author names, similarities, and author sizes in a list of tuples.
    table = []
    for elem in enumerate(sims):
        author_name = model.id2author[elem[0]]
        sim = elem[1]
        author_size = len(model.author2doc[author_name])
        if author_size >= smallest_author:
            table.append((author_name, sim, author_size))
    
    
    #turn similarities table int pd dataframe
    df2 = pd.DataFrame(table, columns=['Author', 'Score', 'Size'])
    #remove the test authors
    df2 = df2[df2['Author'].str.contains("test_id_")==False]
    #sort and get the top 10 predictions
    df2 = df2.sort_values('Score', ascending=False)[:top_n]   
    
    bestmatch = df2.Author.iloc[0]
    
    return bestmatch

def get_table(name, top_n=10, smallest_author=1):
    '''
    Get table with similarities, author names, and author sizes.
    Return `top_n` authors as a dataframe.
    
    '''
    
    # Get similarities.
    sims = get_sims(model.get_author_topics(name))

    # Arrange author names, similarities, and author sizes in a list of tuples.
    table = []
    for elem in enumerate(sims):
        author_name = model.id2author[elem[0]]
        sim = elem[1]
        author_size = len(model.author2doc[author_name])
        if author_size >= smallest_author:
            table.append((author_name, sim, author_size))
            
    df2 = pd.DataFrame(table, columns=['Author', 'Score', 'Size'])
    df2 = df2.sort_values('Score', ascending=False)[:top_n]
    
    return df2


# ### Calculate predicted values:

# In[29]:

#get_ipython().run_cell_magic('time', '', "doc2author_predict = doc2author.copy()\n#randomly replace entries\nfor k in (sample_indices):\n    doc2author_predict[k] = get_bestmatch('test_id_' + str(k))")

doc2author_predict = doc2author.copy()
#randomly replace entries
for k in (sample_indices):
    doc2author_predict[k] = get_bestmatch('test_id_' + str(k))

# In[30]:

#predicted authors
pred = pd.Series(list(doc2author_predict.values()))[sample_indices]


# In[31]:

#real authors
actual = pd.Series(list(doc2author.values()))[sample_indices]


# ## Correctly Predicted:

# In[32]:

sum(pred==actual)/len(pred==actual)


# :(  <br /> <br />
# Only 2.5% prediction accuracy.

# Looking at some authors and texts to find out what went wrong:

# In[34]:

#look at the predictions for a specific post
get_table('test_id_63')


# In[35]:

#look at that post
df.iloc[63]


# In[36]:

#look at all the posts of the real author
df[df.author=="elt"]


# Seems like people don't necessary post in the same topic a lot, so topic models accomplish pretty much the opposite of what we are dealing with here.
