import os
# import pprint
import re
import gensim
import pandas as pd
# from collections import defaultdict
from gensim.corpora import Dictionary
# import gensim.corpora as cp
# import matplotlib.pyplot as plt
from gensim.models import TfidfModel, LsiModel
from gensim.similarities import SparseMatrixSimilarity, MatrixSimilarity

# ------------------------------------------------------------------------

# Cleaning data
def processed_corpus(data):
    # get the stopwords
    file = open("gist_stopwords.txt", "r")
    try:
        content = file.read()
        stopwords = content.split(",")
    finally:
        file.close()

    # I read data.csv and I convert in  lists the columns Name and Comment
    # data = pd.read_csv("data.csv")
    df = data[['Name', 'Comment']].copy()
    # print(df)
    names_list = df['Name'].tolist()
    comments_list = df['Comment'].tolist()

    # remove alphanumeric symbol from comments_list
    comments_list = [re.sub(r'\W+', ' ', comment).strip().lower() for comment in comments_list]

    # remove camel_case
    def camel_case_split(input_str):
        words = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', input_str)
        return words

    # search for camel_case
    # if it returns empty [] I search for "_" in name_list
    # if it returns not empty [], I save the not camel_case string in name_list
    for idx, x in enumerate(names_list):
        camel_case_string = camel_case_split(x)
        if camel_case_string == []:
            names_list[idx] = x.split("_")
        else:
            names_list[idx] = camel_case_string

    # join together name and comment
    name_join_comment = []
    for name, comment in zip(names_list, comments_list):
        name_join_comment.append((str.join(" ", name) + " : " + comment).lower())
        
    # check for stopwords 
    corpus = []
    for entry in name_join_comment:
        corpus.append([word for word in entry.split(" ")[:13] if word not in stopwords])

    # substitute empty string with "INVALID STRING" otherwise I lose the matching with the ground truth for evaluation
    processed_corpus = [[word if word != "" else "INVALID STRING" for word in document] for document in corpus]
    # remove ":" from each list
    processed_corpus = [[item for item in inner_list if item != ":"] for inner_list in processed_corpus]
    
    # I create a dictionary {'key': 'value'}
    temp_dict = {}
    for document in processed_corpus:
        for word in document:
            if word in temp_dict:
                temp_dict[word] += 1
            else:
                temp_dict[word] = 1
    # [print(key, value) for key, value in dict.items()]

    # I create a new dictionary with {'key': 'value'} where value > 1
    frequency_dict = {}
    for key, value in temp_dict.items():
        if value != 1:
            frequency_dict[key] = value
    # [print(key, value) for key, value in dict_frequency_word.items()]   

    # vector of the frequencies
    frequency_vector = []
    for _, value in frequency_dict.items():
        frequency_vector.append(value)

    # print(len(frequency_vector))

    # I create a corpus for the most frequency words
    processed_corpus = [[word for word in document if word in frequency_dict] for document in processed_corpus]

    return processed_corpus

# ------------------------------------------------------------------------

"""FREQ
# search by FT
# create the dictionary and the corpus_bow for both the queries and the processed_corpus
# calculate the similarity between the query and the corpus_bow
Returns:
    _type_: the first 5 most similar results
"""
def search_freq(data, query, processed_corpus):
    dictionary = Dictionary(processed_corpus)
    corpus_bow = [dictionary.doc2bow(text) for text in processed_corpus]
    query_bow = dictionary.doc2bow(query.split())
    corpus_index = MatrixSimilarity(corpus_bow)
    sims = corpus_index[query_bow]
    sims = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    
    # print()
    # print()
    # print("Frequency similarity to: ", query)
    # for idx, score in sims[:5]:
    #     print("Index: ", idx, " Score: ", score, " Phrase: ", processed_corpus[idx])
    #     name, file, line, file_type, _ = data.iloc[idx].to_list()
    #     print("Python" , file_type, ": ", name, "\n File: ", file, " Line: ", line, "\n")

    return sims[:5]

# ------------------------------------------------------------------------

"""TF-IDF
# search by TF-IDF
# create the dictionary and the corpus_bow for both the queries and the processed_corpus
# calculate the similarity between the query and the corpus_bow
Returns:
    _type_: the first 5 most similar results
"""
def search_tfidf(data, query, processed_corpus):
    dictionary = Dictionary(processed_corpus)
    corpus_bow = [dictionary.doc2bow(text) for text in processed_corpus]
    tfidf = TfidfModel(corpus_bow)
    index = SparseMatrixSimilarity(tfidf[corpus_bow], num_features=len(dictionary))
    query_bow = dictionary.doc2bow(query.split())
    sims = index[tfidf[query_bow]]
    sims = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    
    # print()
    # print()
    # print("TF-IDF similarity to: ", query)
    # for idx, score in sims[:5]:
    #     print("Index: ", idx, " Score: ", score, " Phrase: ", str.join(" ", processed_corpus[idx]))
    #     name, file, line, file_type, _ = data.iloc[idx].to_list()
    #     print("Python" , file_type, ": ", name, "\n File: ", file, " Line: ", line, "\n")
    
    return sims[:5]

# ------------------------------------------------------------------------

"""LSI
# search by lsi
# create the dictionary and the corpus_bow for both the queries and the processed_corpus
# create the LsiModel and the corpus_lsi
# compute the similarity between the query and the corpus_lsi
Returns:
    _type_: the first 5 most similar results
"""
def search_lsi(data, query, processed_corpus, return_corpus = False):
    dictionary = Dictionary(processed_corpus)
    corpus_bow = [dictionary.doc2bow(text) for text in processed_corpus]
    tfidf = TfidfModel(corpus_bow)
    corpus_tfidf = tfidf[corpus_bow]

    lsi = LsiModel(corpus_tfidf, id2word=dictionary, num_topics=300)
    corpus_lsi = lsi[corpus_tfidf]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
    
    # return the corpus_lsi if needed for TSNE plots
    if return_corpus:
        return corpus_lsi
    
    vec_bow = dictionary.doc2bow(query.split())
    vec_lsi = lsi[tfidf[vec_bow]]
    
    index = MatrixSimilarity(corpus_lsi) # Create an index for the LSI vectors of your documents
    sims = index[vec_lsi]

    sims = abs(sims)
    sims = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    
    # print()
    # print()
    # print("LSI similarity to: ", query)
    # for idx, score in sims[:5]:
    #     print("Index: ", idx, " Score: ", score, " Phrase: ", str.join(" ", processed_corpus[idx]))
    #     name, file, line, file_type, _ = data.iloc[idx].to_list()
    #     print("Python" , file_type, ": ", name, "\n File: ", file, " Line: ", line, "\n")
    
    return sims[:5]

# ------------------------------------------------------------------------

# Doc2Vec

# create the train_corpus for the doc2vec model 
def read_corpus(fname, tokens_only=False):
        for i, line in enumerate(fname):
            tokens = gensim.utils.simple_preprocess(str.join(" ", line))
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


"""
# search by doc2vec
# build, train and save the model if it doesn't exist
# infer the vector for the query
Returns:
    _type_: the first 5 most similar results
"""
def search_doc2vec(data, query, processed_corpus, return_model = False):
    
    train_corpus = list(read_corpus(processed_corpus))

    # Create the Model -> vectorvector_size=300, min_count=2, epochs=500 if it doesn't exit
    if os.path.exists("doc2vec.model"):
        model = gensim.models.doc2vec.Doc2Vec.load("doc2vec.model")
    else:
        model = gensim.models.doc2vec.Doc2Vec(vector_size=300, min_count=2, epochs=500)
        model.build_vocab(train_corpus)
        model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
        model.save("doc2vec.model")

    # return the model if needed for TSNE plots
    if return_model:
        return model
    
    inferred_vector = model.infer_vector(query.split())
    sims = model.dv.most_similar([inferred_vector], topn=5)
    
    # print to verify the performace of my result
    # print()
    # print()
    # print("Doc2Vec similarity to: ", query)
    # for idx, score in sims[:5]:
    #     print("Index: ", idx, " Score: ", score, " Phrase: ", str.join(" ", processed_corpus[idx]))
    #     name, file, line, file_type, _ = data.iloc[idx].to_list()
    #     print("Python" , file_type, ": ", name, "\n File: ", file, " Line: ", line, "\n")
        
    return sims[:5]

