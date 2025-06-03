import os
import numpy as np
import joblib
from gensim.models import LdaModel, CoherenceModel
from sklearn.decomposition import TruncatedSVD, NMF

# Limit threads in each process to avoid oversubscription.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def evaluate_topic_models(num_topics, corpus, corpus_tfidf_sparse, dictionary, processed_texts, save_dir=None, save_models=False):
    results = {}
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # --- Modelo LDA ---
    # Reduced passes for faster convergence on small datasets
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=5, alpha='symmetric', eta='auto')
    coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_texts, dictionary=dictionary, coherence='c_v')
    results["coherence_lda"] = coherence_model_lda.get_coherence()
    if save_models and save_dir:
        joblib.dump(lda_model, os.path.join(save_dir, f"lda_model_{num_topics}.pkl"))

    # --- Modelo LSA ---
    svd = TruncatedSVD(n_components=num_topics)
    _ = svd.fit_transform(corpus_tfidf_sparse)
    lsa_topic_word = svd.components_
    lsa_topics = []
    for topic in lsa_topic_word:
        # Get indices for top 10 words (in descending order)
        top_indices = np.argsort(topic)[-10:][::-1]
        top_words = [dictionary[i] for i in top_indices]
        lsa_topics.append(top_words)
    coherence_model_lsa = CoherenceModel(topics=lsa_topics, texts=processed_texts, dictionary=dictionary, coherence='c_v')
    results["coherence_lsa"] = coherence_model_lsa.get_coherence()
    if save_models and save_dir:
        joblib.dump(svd, os.path.join(save_dir, f"lsa_model_{num_topics}.pkl"))

    # --- Modelo pLSA (via NMF) ---
    nmf = NMF(n_components=num_topics, init='nndsvd', random_state=42, max_iter=200)
    _ = nmf.fit_transform(corpus_tfidf_sparse)
    plsa_topic_word = nmf.components_
    plsa_topics = []
    for topic in plsa_topic_word:
        top_indices = np.argsort(topic)[-10:][::-1]
        top_words = [dictionary[i] for i in top_indices]
        plsa_topics.append(top_words)
    coherence_model_plsa = CoherenceModel(topics=plsa_topics, texts=processed_texts, dictionary=dictionary, coherence='c_v')
    results["coherence_plsa"] = coherence_model_plsa.get_coherence()
    if save_models and save_dir:
        joblib.dump(nmf, os.path.join(save_dir, f"plsa_model_{num_topics}.pkl"))

    return num_topics, results