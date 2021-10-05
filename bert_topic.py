from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
from umap import UMAP
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity


def make_bertopic_model():
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=10, metric='euclidean',
                                    cluster_selection_method='eom',
                                    prediction_data=True)

    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                      metric='cosine', random_state=42)

    vectorizer_model = CountVectorizer(ngram_range=(1, 1), stop_words="english")

    topic_model = BERTopic(embedding_model='paraphrase-MiniLM-L6-v2',
                           umap_model=umap_model,
                           hdbscan_model=hdbscan_model,
                           vectorizer_model=vectorizer_model, nr_topics=15,
                           calculate_probabilities=True
                           )

    return topic_model


# Divide Task category
task1 = ['comp.graphics', 'comp.os.ms-windows.misc',
         'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
         'comp.windows.x', 'alt.atheism', 'misc.forsale'
         ]

task2 = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
         'rec.sport.hockey']

task3 = ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
         'soc.religion.christian']

task4 = ['talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc',
         'talk.religion.misc']


def store_documents(category, model):
    """ Save twenty representative documents for topics

    # Arguments:
        category : Task (List[str])
        model : BERTopic Model

    # Returns
        “store_doc” : representative documents for each topics
    """

    # Load data
    data = fetch_20newsgroups(subset='train', remove=('headers', 'footers',
                                                      'quotes'),
                              categories=category)
    docs = data["data"]

    # load trained model predict, topic_embeddings, umap_embeddings
    topics = model.predict
    topic_embeddings = model._reduce_dimensionality(model.topic_embeddings)
    umap_embeddings = model.umap_embeddings

    doc_vec = defaultdict(list)
    store_doc = []

    # doc_vec : document vector(dim=5)
    for i in range(len(docs)):
        doc_vec[topics[i]].append(umap_embeddings[i].tolist())

    for i in range(15):
        topic_embedding = [topic_embeddings[i]]
        tmp = {}

        # topic vector와 각 document vector cosine_similarity 계산, 정렬
        for j in range(len(doc_vec[i])):
            tmp[j] = cosine_similarity(topic_embedding, [doc_vec[i][j]])[
                0][0]
        res = sorted(tmp.items(), key=lambda x: x[1], reverse=True)

        # topic vector와 가까운 20개의 document 저장 15개 topic * 20개 doc
        for j in range(20):
            store_doc.append(docs[res[j][0]])

    return store_doc


def train_tm(model, category, store_doc: None):
    """ Fit the models (Bert, UMAP and HDBSCAN) on a collection of documents
    and  generate topics

    # Arguments:
        model : BERTopic Model
        category : Task (List[str])
        store_doc : prior task documents

    # Returns
        Topic_model : BERTopic Model
    """

    # Load data
    data = fetch_20newsgroups(subset='train', remove=('headers', 'footers',
                                                      'quotes'),
                              categories=category)
    docs = data["data"]

    # Using Replay Methods
    if store_doc is not None:
        docs = docs + store_doc

    # Data Fitting
    topic_model = model.fit(docs)

    # Print Model Output
    print(topic_model.get_topic_info())
    for i in range(len(topic_model.get_topic_info()) - 1):
        print(topic_model.get_topics()[i])

    return topic_model


def test_tm(model, category):
    """ After having fit a model, use transform to predict new instances

    # Arguments:
        model : BERTopic Model
        category : Task (List[str])

    # Returns
        topics :  topic prediction for each documents
        probs : the topic probability distribution
    """

    # Load data
    data = fetch_20newsgroups(subset='test', remove=('headers', 'footers',
                                                     'quotes'),
                              categories=category)
    docs = data["data"]

    # Predict New Instance
    topics, probs = model.transform(docs)

    return topics, probs

