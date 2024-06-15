import search_data as search_data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# from gensim.models import Doc2Vec
from sklearn.manifold import TSNE

data = pd.read_csv("data.csv")
processed_corpus = search_data.processed_corpus(data)


"""plot the TSNE of the embeddings
# create a dataframe with the embeddings and the labels
# create a target list with the labels to use in the scatterplot
# add the target list to the dataframe
# plot the scatterplot
Returns:
    _type_: save the plot (.png)
"""


def plot_TSNE(embeddings, perplexity=2, n_iter=3000, name="lsi.png"):
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter)
    data_2d = tsne.fit_transform(np.array(embeddings))

    df = pd.DataFrame(data=data_2d, columns=['Dimension 1', 'Dimension 2'])

    labels = []

    for i in range(0, 10):
        labels += [i] * 5

    for i in range(0, 10):
        labels += [i]

    df["Group"] = labels

    sns.scatterplot(data=df[:50], x="Dimension 1",
                    y="Dimension 2", hue="Group", palette="deep")
    sns.scatterplot(data=df[50:], x="Dimension 1",
                    y="Dimension 2", hue="Group", palette="deep", marker="X", legend=False, edgecolor="black")
    if name == "doc2vec.png":
        plt.xlim(left=-200, right=200)
        plt.ylim(bottom=-200, top=200)

    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
    plt.title("TSNE of the embeddings of " + name + "\nperplexity = " +
              str(perplexity) + "\nn_iter = " + str(n_iter))
    plt.tight_layout()

    if name == "lsi.png":
        plt.savefig(name)
    else:
        plt.savefig(name)

    plt.show()

# call search_data.search_lsi and search_data.search_doc2vec to get the models


def give_me_models():
    lsi_corpus = search_data.search_lsi(
        data, " ", processed_corpus, return_corpus=True)

    doc2vec_model = search_data.search_doc2vec(
        data, " ", processed_corpus, return_model=True)

    return lsi_corpus, doc2vec_model

# call search_data.search_freq, search_data.search_tfidf, search_data.search_lsi, search_data.search_doc2vec
# to get the indices of the 5 most similar results for each model


def give_me_indeces(query):

    frequency_sims = search_data.search_freq(
        data, query.lower(), processed_corpus)

    tfidf_sims = search_data.search_tfidf(
        data, query.lower(), processed_corpus)

    lsi_sims = search_data.search_lsi(data, query.lower(), processed_corpus)

    doc2vec_sims = search_data.search_doc2vec(
        data, query.lower(), processed_corpus)

    return frequency_sims, tfidf_sims, lsi_sims, doc2vec_sims


# query, node, path
# read the ground truth file and store it in a list
file = open("ground-truth-unique.txt", "r")
ground_truth = []

with open("ground-truth-unique.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        if line != '\n':
            line = line.strip()
            ground_truth.append(line)

file.close()
# print(ground_truth)

# split the list in sublists of 3 elements
ground_truth = [ground_truth[i:i + 3] for i in range(0, len(ground_truth), 3)]
# [print(elem) for elem in ground_truth]

print()

# to remove ./ in the path of each element in the ground truth
for elem in ground_truth:
    elem[2] = elem[2][2:]
# [print(elem) for elem in ground_truth]

print()

# create a list of indeces
# for each query in the ground truth, find the index of the query in the data.csv
indeces = []
for elem in ground_truth:
    node = elem[1]
    path = elem[2]
    print(elem)
    print("Name: ", node, " Path file: ", path)
    idx = np.where((data['Name'] == node) & (data['Path'] == path))[0][0]
    print(data.loc[idx], "\n")
    indeces.append(idx)

print("Indeces: ", indeces, "\n")


# init matching and precision
matching_frequency_sims = 0
matching_tfidf_sims = 0
matching_lsi_sims = 0
matching_doc2vec_sims = 0

frequency_precision = []
tfidf_precision = []
lsi_precision = []
doc2vec_precision = []

lsi_indeces = []
doc2vec_indeces = []


"""calculate_recall_precision
# get the list of indeces of the most similar documents for the model that calls the function
# check if the index of the query is in the list of indeces
# if it is, then the matching count is incremented by 1 and the precision is calculated as 1/(pos+1)
# otherwise, the precision is 0
Returns:
    _type_: precision, matching_count
"""


def calculate_recall_precision(index, matching_count, sims, precision, model_name=""):
    values = [value[0] for value in sims]
    global lsi_indeces
    global doc2vec_indeces

    # for LSI and Doc2Vec, append every time the indeces of the most similar documents to a global list for plotting
    if model_name == "LSI":
        lsi_indeces += values
    elif model_name == "Doc2Vec":
        doc2vec_indeces += values

    if indeces[index] in values:
        matching_count += 1
        index_in_values = values.index(indeces[index])
        precision.append(1/(index_in_values + 1))
    else:
        precision.append(0)

    return precision, matching_count


# infer the models for each query in the ground truth
for index, elem in enumerate(ground_truth):
    frequency_sims, tfidf_sims, lsi_sims, doc2vec_sims = give_me_indeces(
        elem[0])

    # print the most 5 similar for each model
    print(f"Query: {elem[0]}\nFrequency sims: {frequency_sims}\nTFIDF sims: {tfidf_sims}\nLSI sims: {lsi_sims}\nDoc2Vec sims: {doc2vec_sims}\n")

    # calculate the matching and the precision for each model
    frequency_precision, matching_frequency_sims = calculate_recall_precision(
        index, matching_frequency_sims, frequency_sims, frequency_precision)

    tfidf_precision, matching_tfidf_sims = calculate_recall_precision(
        index, matching_tfidf_sims, tfidf_sims, tfidf_precision)

    lsi_precision, matching_lsi_sims = calculate_recall_precision(
        index, matching_lsi_sims, lsi_sims, lsi_precision, "LSI")

    doc2vec_precision, matching_doc2vec_sims = calculate_recall_precision(
        index, matching_doc2vec_sims, doc2vec_sims, doc2vec_precision, "Doc2Vec")


# print the matching sims and the precision
print("Matching frequency sims: ", matching_frequency_sims)
print("Matching tfidf sims: ", matching_tfidf_sims)
print("Matching lsi sims: ", matching_lsi_sims)
print("Matching doc2vec sims: ", matching_doc2vec_sims)
print()
print(
    f"Precision frequency sims: {frequency_precision}, average: {np.mean(frequency_precision)}")
print(
    f"Precision tfidf sims: {tfidf_precision}, average: {np.mean(tfidf_precision)}")
print(
    f"Precision lsi sims: {lsi_precision}, average: {np.mean(lsi_precision)}")
print(
    f"Precision doc2vec sims: {doc2vec_precision}, average: {np.mean(doc2vec_precision)}")


# write into precision_recall.txt
f = open("precision_recall.txt", "w+")
f.writelines("frequency sims:\n recall: " + str(matching_frequency_sims/10) +
             "\n precision: " + str(np.mean(frequency_precision)) + "\n\n")
f.writelines("tfid sims:\n recall: " + str(matching_tfidf_sims/10) +
             "\n precision: " + str(np.mean(tfidf_precision)) + "\n\n")
f.writelines("lsi sims:\n recall: " + str(matching_lsi_sims/10) +
             "\n precision: " + str(np.mean(lsi_precision)) + "\n\n")
f.writelines("doc2vec sims:\n recall: " + str(matching_doc2vec_sims/10) +
             "\n precision: " + str(np.mean(doc2vec_precision)) + "\n\n")
f.close()

# get the lsi_corpus and doc2vec_model
lsi_corpus, doc2vec_model = give_me_models()

# get the embeddings
lsi_embeddings = [[val[1] for val in lsi_corpus[i]] for i in lsi_indeces]
doc2vec_embeddings = [doc2vec_model.infer_vector(
    processed_corpus[i]) for i in doc2vec_indeces]

# get the embiddings of the ground truth
ground_truth_embeddings_lsi = [[elem[1]
                                for elem in lsi_corpus[idx]] for idx in indeces]
ground_truth_embeddings_doc2vec = [doc2vec_model.infer_vector(
    processed_corpus[idx]) for idx in indeces]

lsi_embeddings = lsi_embeddings + ground_truth_embeddings_lsi
doc2vec_embeddings = doc2vec_embeddings + ground_truth_embeddings_doc2vec
# plot the embeddings
plot_TSNE(lsi_embeddings)
plot_TSNE(doc2vec_embeddings, name="doc2vec.png")
