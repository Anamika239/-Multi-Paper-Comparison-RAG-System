import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import hashlib
from typing import List, Dict

class MultiPaperRAG:
    def __init__(self):
        """Initialize the RAG system with embedding model and FAISS index"""
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.persist_dir = "./chroma_db"
        os.makedirs(self.persist_dir, exist_ok=True)
        
        self.dimension = 384
        self.index = None
        self.chunks = []
        self.metadatas = []
        
        self._load_data()
        
    def _save_data(self):
        """Save index and metadata to disk"""
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(self.persist_dir, "index.faiss"))
        
        with open(os.path.join(self.persist_dir, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        with open(os.path.join(self.persist_dir, "metadatas.pkl"), "wb") as f:
            pickle.dump(self.metadatas, f)
    
    def _load_data(self):
        """Load existing index and metadata from disk"""
        index_path = os.path.join(self.persist_dir, "index.faiss")
        chunks_path = os.path.join(self.persist_dir, "chunks.pkl")
        metadatas_path = os.path.join(self.persist_dir, "metadatas.pkl")
        
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
        
        if os.path.exists(chunks_path):
            with open(chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
        if os.path.exists(metadatas_path):
            with open(metadatas_path, "rb") as f:
                self.metadatas = pickle.load(f)
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.split()) > 50:
                chunks.append(chunk)
        
        return chunks
    
    def add_paper(self, text: str, paper_name: str, metadata: Dict = None):
        """Add a paper to the vector store"""
        chunks = self.chunk_text(text)
        
        if not chunks:
            return 0
        
        embeddings = self.model.encode(chunks).astype('float32')
        
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)
        
        self.index.add(embeddings)
        
        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            chunk_meta = {
                "paper_name": paper_name,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            if metadata:
                chunk_meta.update(metadata)
            self.metadatas.append(chunk_meta)
        
        self._save_data()
        return len(chunks)
    
    def remove_paper(self, paper_name: str):
        """Delete all chunks belonging to a specific paper"""
        indices_to_remove = []
        for i, meta in enumerate(self.metadatas):
            if meta.get('paper_name') == paper_name:
                indices_to_remove.append(i)
        
        if not indices_to_remove:
            return False
        
        for idx in sorted(indices_to_remove, reverse=True):
            del self.chunks[idx]
            del self.metadatas[idx]
        
        if self.chunks:
            embeddings = self.model.encode(self.chunks).astype('float32')
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
        
        self._save_data()
        return True
    
    def retrieve_similar_chunks(self, query: str, n_results: int = 10) -> List[Dict]:
        """Retrieve most relevant chunks for a query"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        query_embedding = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_embedding, min(n_results, self.index.ntotal))
        
        retrieved = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                retrieved.append({
                    'id': f"chunk_{idx}",
                    'text': self.chunks[idx],
                    'metadata': self.metadatas[idx],
                    'distance': float(distances[0][i])
                })
        
        return retrieved
    
    def compare_papers(self, query: str, papers: List[str]) -> Dict:
        """Compare multiple papers on a specific aspect"""
        all_chunks = self.retrieve_similar_chunks(query, n_results=20)
        
        paper_chunks = {}
        for chunk in all_chunks:
            paper = chunk['metadata'].get('paper_name', 'Unknown')
            if paper in papers:
                if paper not in paper_chunks:
                    paper_chunks[paper] = []
                paper_chunks[paper].append(chunk)
        
        comparison = {'query': query, 'papers': {}}
        
        for paper, chunks in paper_chunks.items():
            chunks.sort(key=lambda x: x['distance'])
            top_chunks = chunks[:3]
            top_texts = [c['text'] for c in top_chunks]
            
            comparison['papers'][paper] = {
                'relevant_chunks': len(chunks),
                'top_passages': top_texts,
                'avg_relevance': np.mean([c['distance'] for c in chunks]) if chunks else 1.0
            }
        
        return comparison
    
    def find_common_themes(self, papers: List[str], top_k: int = 5) -> List[str]:
        """Find common themes across papers"""
        all_text = []
        for i, meta in enumerate(self.metadatas):
            if meta.get('paper_name') in papers:
                all_text.append(self.chunks[i])
        
        if len(all_text) < 3:
            return ["Not enough text to analyze themes"]
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(all_text)
        
        feature_names = vectorizer.get_feature_names_out()
        scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
        top_indices = scores.argsort()[-top_k:][::-1]
        
        return [feature_names[i] for i in top_indices]
    
    def get_paper_summary_stats(self, paper: str) -> Dict:
        """Get summary statistics for a paper"""
        chunks_for_paper = []
        for i, meta in enumerate(self.metadatas):
            if meta.get('paper_name') == paper:
                chunks_for_paper.append(self.chunks[i])
        
        if not chunks_for_paper:
            return {}
        
        chunk_lengths = [len(chunk.split()) for chunk in chunks_for_paper]
        
        return {
            'paper': paper,
            'num_chunks': len(chunks_for_paper),
            'avg_chunk_length': float(np.mean(chunk_lengths)) if chunk_lengths else 0,
            'total_words': int(np.sum(chunk_lengths)) if chunk_lengths else 0
        }
    
    def get_all_papers(self) -> List[str]:
        """Get list of all papers in the database"""
        papers = set()
        for meta in self.metadatas:
            if meta and 'paper_name' in meta:
                papers.add(meta['paper_name'])
        return list(papers)
