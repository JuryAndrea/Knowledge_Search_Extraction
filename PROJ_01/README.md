# Project 01: Multi-source code search

### About the Project
**Goal:** Develop a search engine that can query a large Python code repository using multiple sources of information

**Author:** Jury Andrea D'Onofrio 
#### To run extract_data.py
Please make sure to import the libraries `os`, `pandas as pd`, and `ast`. Then you will be able to run the file correctly.

The script will create a `.csv` file called `data.csv`.

#### To run search_data.py
Please make sure to import the libraries `os`, `re`, `gensim`.

Moreover, you need to add:
 - `from gensim.corpora import Dictionary`
 - `from gensim.models import TfidfModel, LsiModel`
 - `from gensim.similarities import SparseMatrixSimilarity, MatrixSimilarity`

 The script will not return anything since it contains all the methods required for `prec_recall.py`.

 #### To run prec_recall.py
Please make sure to import `search_data as search_data`, `pandas as pd`, `numpy as np`, `seaborn as sns` and `matplotlib.pyplot as plt`.

Moreover, you need to add:
- `from sklearn.manifold import TSNE` so you need to install `sklearn`

The script will create:
- the `doc2vec.model` if it doesn't exist (it takes some time to create it)
- `doc2vec.png` and `lsi.png`
