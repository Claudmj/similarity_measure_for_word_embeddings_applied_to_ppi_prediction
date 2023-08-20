# Similarity measure for word embeddings applied to protein-protein interaction prediction

In this study, we propose an alternative approach to PPI prediction that in the form of a simple heuristic. Our approach utilises natural language processing methods, such as word2vec and latent Dirichlet allocation, to feature engineer molecular text data. This approach creates a robust and explainable embedding that is easy to implement and relies solely on chemical data, such as protein sequences. Additionally, we show that these explainable embeddings are meaningful in representing a dataset in vector space and as such offer exceptional performance in predicting PPIs through embedding similarity.

## Installation instructions
An Anaconda environment with Python 3.11 was used. Clone the project and navigate to the project folder:
pip install -r requirements.txt
## Running experiments
There is a script for each of the different experiments in src/experiments.