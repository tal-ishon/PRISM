from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
import pickle
import numpy as np

texts = None
dictionary = None

def load_data(path, to_save=False):
    global texts, dictionary
    # Load the dictionary from a file
    dictionary = corpora.Dictionary.load(f'{path}/lda_dictionary.gensim')

    # Load the BoW corpus from a Matrix Market format file
    bow_corpus = corpora.MmCorpus(f'{path}/lda_corpus.mm')
    doc_term_matrix = [list(doc) for doc in bow_corpus]

    # for i, doc in enumerate(doc_term_matrix):
    #     print(f"Document {i}:")
    #     for word_id, count in doc:
    #         if count > 1:
    #             print(f"  {dictionary[word_id]}: {count}")
    
    if to_save:
        with open(f'{path}/filtered_corpus.pkl', 'rb') as f:
            filtered_corpus = pickle.load(f)
        from sklearn.feature_extraction.text import CountVectorizer
        
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(filtered_corpus)
        import pandas as pd
        
        df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        df.to_csv("doc_term_matrix.csv", index=False)

        exit(0)
    
    texts = [
            list((dictionary[word_id] for word_id, freq in bow))
            for bow in doc_term_matrix
        ]
    
    return dictionary, bow_corpus, doc_term_matrix, texts

def load_octis_data(path, to_save=False):
    global texts, dictionary
    # load json file
    import json
    with open(f"{path}.json", 'r') as f:
        texts = json.load(f)
    # create dictionary
    dictionary = corpora.Dictionary(texts)
    # Only needed if something requires id2token explicitly (e.g., printing topics)
    dictionary.id2token = {id_: token for token, id_ in dictionary.token2id.items()}

    # create bag of words
    bow_corpus = [dictionary.doc2bow(text) for text in texts]
    # create doc term matrix
    doc_term_matrix = [list(doc) for doc in bow_corpus]
    
    return dictionary, bow_corpus, doc_term_matrix, texts
    
def get_topics(LDA):
   return [[word for word, _ in LDA.show_topic(topicid, topn=10)] for topicid in range(LDA.num_topics)]

def compute_cv_coherence(model_name, topics):
    coherence_model_cv = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_cv = coherence_model_cv.get_coherence()
    print(f"{model_name} c_v Coherence: {round(coherence_cv, 5)}")
    # with open(f"20NewsGroup_cv_coherence_{len(topics)}_{model_name}.txt", "w") as f:
    #     f.write(f"{model_name} c_v Coherence: {round(coherence_cv, 5)}\n")
    return coherence_cv

    
# Function to calculate NPMI
def calculate_npmi(topics, texts, model_name):
    npmi_metric = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_npmi') 
    cohenrence_npmi = npmi_metric.get_coherence()
    print(f"{model_name} npmi Coherence: {round(cohenrence_npmi, 5)}")
    return cohenrence_npmi


def compute_umass_coherence(model_name, model):
    from gensim.models.coherencemodel import CoherenceModel
    # Compute u_mass coherence
    coherence_model_umass = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='u_mass')
    coherence_umass = coherence_model_umass.get_coherence()
    print(f"{model_name} u_mass Coherence: {round(coherence_umass, 5)}")
    return coherence_umass
    
    
def compute_topic_diversity_coherence(model_name, model, top_n):
    """
    Calculate topic diversity for a given LDA model.

    Parameters:
    lda_model (gensim.models.LdaModel): Trained LDA model
    top_n (int): Number of top words per topic to consider
    
    """
    if model_name == "BERTopic":
        # Get top N words for each topic
        topics = model.get_topics()
        
        # Exclude the '-1' topic if present, which refers to outliers/noise
        if -1 in topics:
            del topics[-1]

        top_words = [word for topic_id in topics for word, _ in topics[topic_id][:top_n]]

    else:
        # Get top N words for each topic
        topics = model.show_topics(num_topics=-1, num_words=top_n, formatted=False)
        # Flatten the list of words and calculate unique words
        top_words = [word for topic in topics for word, _ in topic[1]]
        

    unique_words = set(top_words)

    # Calculate topic diversity
    total_words = len(top_words)
    unique_words_count = len(unique_words)
    topic_diversity_coherence = unique_words_count / total_words

    print(f"{model_name} Topic - Diversity Coherence: {round(topic_diversity_coherence, 5)}")
    return topic_diversity_coherence


# given a corpus, create a document-term matrix
def create_doc_term_matrix(corpus):
    # Create a dictionary representation of the documents.
    dictionary = corpora.Dictionary(corpus)

    # Create a bag-of-words representation of the documents.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in corpus]

    # Save doc_term_matrix as numpy array
    doc_term_matrix = np.array(doc_term_matrix)
    np.save('doc_term_matrix.npy', doc_term_matrix)
    
    
def create_random_vector(length):
    """
    Create a random vector of given length in range 0.01 to 0.05.
    """
    return np.random.uniform(0.01, 0.05, length).tolist()

def create_fixed_vector(length, value):
    """
    Create a fixed np vector of given length with all values set to the specified value.
    """
    return np.full(length, value).tolist()

def save_vector_to_csv(vector, filename):
    """
    Save a vector to a CSV file.
    
    Parameters:
    vector (list): Vector to save
    filename (str): Name of the output CSV file
    """
    import pandas as pd
    df = pd.DataFrame(vector).T
    df.to_csv(filename, index=False, header=False)
    
# vec = create_random_vector(1696)
# # vec = create_fixed_vector(1612, 0.01)
# save_vector_to_csv(vec, "/home/tal/dev/Mallet/priors/random_M10.csv")