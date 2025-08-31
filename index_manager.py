import threading
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from search_engine import build_general_indices

class IndexManager:
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.dataframe = None
        self.general_indices = None
    @classmethod
    def get_instance(cls):
        with cls._lock: 
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
        
    def initialize(self, df):
        self.dataframe = df.copy()
        embeddings = self.model.encode(df["HS Course Description"].tolist(), convert_to_numpy=True).astype('float32') # Course Descriptions
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings)

        self.general_indices = build_general_indices(df, self.model)
    def search_similar_course(self, description, top_k=10):
        # Convert input course to an embedding
        input_embedding = self.model.encode([description], convert_to_numpy=True).astype('float32')

        #  Search for the most similar courses
        distances, indices = self.index.search(input_embedding, k=top_k)  
        return distances[0], indices[0]
    
    def get_model(self):
        return self.model

    def get_general_indices(self):
        return self.general_indices

    def get_df(self):
        return self.dataframe
