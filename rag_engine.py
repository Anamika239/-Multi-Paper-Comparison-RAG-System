import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Tuple
import hashlib
import os

class MultiPaperRAG:
    def __init__(self):
        """Initialize the RAG system with embedding model and vector store"""
        print("📚 Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create directory for persistence
        persist_dir = "./chroma_db"
        os.makedirs(persist_dir, exist_ok=True)
        
        # Initialize ChromaDB with persistent client
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="research_papers",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.papers_metadata = {}
        
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.split()) > 100:  # Minimum chunk size
                chunks.append(chunk)
        
        return chunks
    
    def add_paper(self, text: str, paper_name: str, metadata: Dict = None):
        """Add a paper to the vector store"""
        # Create chunks
        chunks = self.chunk_text(text)
        
        if not chunks:
            print(f"⚠️ No chunks created for {paper_name}")
            return 0
        
        # Generate embeddings
        embeddings = self.model.encode(chunks).tolist()
        
        # Create IDs
        ids = [f"{paper_name}_{i}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}" 
               for i, chunk in enumerate(chunks)]
        
        # Prepare metadata for each chunk
        metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_meta = {
                "paper_name": paper_name,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            if metadata:
                chunk_meta.update(metadata)
            metadatas.append(chunk_meta)
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        # Store paper metadata
        self.papers_metadata[paper_name] = {
            'num_chunks': len(chunks),
            'metadata': metadata
        }
        
        print(f"✅ Added {len(chunks)} chunks for {paper_name}")
        return len(chunks)
    
    def remove_paper(self, paper_name: str):
        """Delete all chunks belonging to a specific paper from the collection."""
        try:
            # Get all IDs for chunks with this paper_name
            results = self.collection.get(where={"paper_name": paper_name})
            if results and results['ids']:
                self.collection.delete(ids=results['ids'])
                # Also remove from papers_metadata
                if paper_name in self.papers_metadata:
                    del self.papers_metadata[paper_name]
                return True
        except Exception as e:
            print(f"Error removing paper {paper_name}: {e}")
            return False
        return False
    
    def retrieve_similar_chunks(self, query: str, n_results: int = 10) -> List[Dict]:
        """Retrieve most relevant chunks for a query"""
        # Generate query embedding
        query_embedding = self.model.encode([query]).tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        # Format results
        retrieved = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                retrieved.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0
                })
        
        return retrieved
    
    def compare_papers(self, query: str, papers: List[str]) -> Dict:
        """Compare multiple papers on a specific aspect"""
        # Retrieve relevant chunks
        all_chunks = self.retrieve_similar_chunks(query, n_results=20)
        
        # Group by paper
        paper_chunks = {}
        for chunk in all_chunks:
            paper = chunk['metadata'].get('paper_name', 'Unknown')
            if paper in papers:
                if paper not in paper_chunks:
                    paper_chunks[paper] = []
                paper_chunks[paper].append(chunk)
        
        # Generate comparison
        comparison = {
            'query': query,
            'papers': {},
            'similarities': [],
            'differences': []
        }
        
        for paper, chunks in paper_chunks.items():
            # Sort by relevance (lower distance = more relevant)
            chunks.sort(key=lambda x: x['distance'])
            
            # Get top chunks
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
        # Get all chunks from these papers
        all_text = []
        for paper in papers:
            results = self.collection.get(
                where={"paper_name": paper}
            )
            if results and results['documents']:
                all_text.extend(results['documents'])
        
        if len(all_text) < 3:
            return ["Not enough text to analyze themes"]
        
        # Use TF-IDF to find common themes
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(all_text)
        
        # Get top terms
        feature_names = vectorizer.get_feature_names_out()
        scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
        top_indices = scores.argsort()[-top_k:][::-1]
        
        common_themes = [feature_names[i] for i in top_indices]
        return common_themes
    
    def get_paper_summary_stats(self, paper: str) -> Dict:
        """Get summary statistics for a paper"""
        results = self.collection.get(
            where={"paper_name": paper}
        )
        
        if not results or not results['documents']:
            return {}
        
        # Calculate average chunk length
        chunk_lengths = [len(doc.split()) for doc in results['documents']]
        
        return {
            'paper': paper,
            'num_chunks': len(results['documents']),
            'avg_chunk_length': float(np.mean(chunk_lengths)) if chunk_lengths else 0,
            'total_words': int(np.sum(chunk_lengths)) if chunk_lengths else 0
        }
    
    def get_all_papers(self) -> List[str]:
        """Get list of all papers in the database"""
        try:
            results = self.collection.get()
            papers = set()
            if results and results['metadatas']:
                for meta in results['metadatas']:
                    if meta and 'paper_name' in meta:
                        papers.add(meta['paper_name'])
            return list(papers)
        except:
            return []
