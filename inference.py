from tensorflow.keras.models import load_model
import pickle
import numpy as np
from loguru import logger

logger.debug("Loading model...")
model = load_model("model-artifacts/model.keras")
logger.debug("Model loaded successfully!")

logger.debug("Loading vectorizer...")
with open("model-artifacts/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
logger.debug("Vectorizer loaded successfully!")

logger.debug("Loading label encoder...")
with open("model-artifacts/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
logger.debug("Label encoder loaded successfully!")


def preprocess_query(queries):
    """
    Pre-process the queries, vectorize them and reshape for model
    :return Returns vectors
    """
    query_vectors = vectorizer.transform(queries).toarray()
    query_vectors = query_vectors.reshape(
        query_vectors.shape[0], query_vectors.shape[1], 1
    )  # Reshape for Conv1D
    return query_vectors


@logger.catch
def get_predictions(queries):
    """
    Process the queries which are in list and return the results after inferencing the model
    :return list of {'query', 'category', 'probability'} after inferencing
    """
    query_vectors = preprocess_query(queries)

    predicted_probs = model.predict(query_vectors)
    predicted_indices = predicted_probs.argmax(axis=1)
    max_probs = predicted_probs.max(axis=1)
    max_probs = np.atleast_1d(max_probs)
    predicted_category = label_encoder.inverse_transform(predicted_indices)
    results = [
        {
            "query": query,
            "category": category,
            "probability": round(float(probability), 4),
        }
        for query, category, probability in zip(queries, predicted_category, max_probs)
    ]

    return results
