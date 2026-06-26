import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

class WebArchiveDataset(Dataset):
    def __init__(self, documents):
        self.documents = documents

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        return self.documents[idx]

class TopicModeling:
    def __init__(self, n_topics=5, max_features=1000):
        self.n_topics = n_topics
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.nmf_model = NMF(n_components=n_topics, random_state=42)

    def fit_transform(self, documents):
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        topic_matrix = self.nmf_model.fit_transform(tfidf_matrix)
        return topic_matrix

    def get_top_words(self, n_top_words=10):
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(self.nmf_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            topics.append(top_words)
        return topics

class EventDetection:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def detect_events(self, topic_matrix):
        normalized_matrix = normalize(topic_matrix, axis=1, norm='l1')
        event_indices = np.where(normalized_matrix.max(axis=1) > self.threshold)[0]
        return event_indices

if __name__ == '__main__':
    # Dummy data: Simulated web archive documents
    documents = [
        "The economy is growing rapidly with new policies.",
        "A major earthquake has struck the city causing widespread damage.",
        "The new technology conference will showcase AI advancements.",
        "Political tensions are rising in the region.",
        "A new sports event has been announced for next summer.",
        "The stock market is experiencing significant volatility.",
        "Scientists have discovered a new species in the Amazon rainforest.",
        "The local community is organizing a charity event for disaster relief."
    ]

    # Step 1: Create a dataset
    dataset = WebArchiveDataset(documents)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Step 2: Topic Modeling
    topic_model = TopicModeling(n_topics=3, max_features=50)
    all_documents = [doc for doc in dataset]
    topic_matrix = topic_model.fit_transform(all_documents)

    # Print topics and top words
    topics = topic_model.get_top_words(n_top_words=5)
    print("Identified Topics and Top Words:")
    for idx, topic in enumerate(topics):
        print(f"Topic {idx + 1}: {', '.join(topic)}")

    # Step 3: Event Detection
    event_detector = EventDetection(threshold=0.4)
    event_indices = event_detector.detect_events(topic_matrix)

    # Print detected events
    print("\nDetected Events (Document Indices):", event_indices)
    print("Event Documents:")
    for idx in event_indices:
        print(f"Document {idx}: {documents[idx]}")