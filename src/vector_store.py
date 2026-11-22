"""
Vector Store for Air Quality RAG Chatbot
Uses FAISS (with high-performance NumPy fallback) for semantic search and retrieval
"""

import os
import json
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import hashlib
import re

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector store implementation with FAISS and pure-NumPy cosine similarity fallback"""
    
    def __init__(self, model_name: str = "simple"):
        self.model_name = model_name
        self.index = None
        self.embeddings = None  # Store numpy array of embeddings directly
        self.documents: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.dimension = 384  # Fixed dimension for simple embeddings
        
    async def initialize(self):
        """Initialize the vector store"""
        try:
            if FAISS_AVAILABLE:
                self.index = faiss.IndexFlatIP(self.dimension)
                logger.info("FAISS vector store initialized")
            else:
                self.index = None
                self.embeddings = np.empty((0, self.dimension), dtype=np.float32)
                logger.info("NumPy fallback vector store initialized (FAISS not installed)")
            
            # Load existing data if available
            await self._load_existing_data()
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    async def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector store"""
        try:
            if not documents:
                return

            texts = []
            metadatas = []
            
            for doc in documents:
                chunks = self._split_document(doc)
                for chunk in chunks:
                    texts.append(chunk['content'])
                    metadatas.append(chunk.get('metadata', {}))
            
            if not texts:
                return

            # Generate normalized embeddings
            new_embeddings = self._generate_simple_embeddings(texts).astype(np.float32)
            
            if FAISS_AVAILABLE and self.index is not None:
                faiss.normalize_L2(new_embeddings)
                self.index.add(new_embeddings)
            else:
                if self.embeddings is None or len(self.embeddings) == 0:
                    self.embeddings = new_embeddings
                else:
                    self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
            # Update document and metadata storage
            self.documents.extend(texts)
            self.metadata.extend(metadatas)
            
            # Save to disk
            await self._save_data()
            logger.info(f"Added {len(texts)} document chunks to vector store (Total: {len(self.documents)})")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    async def similarity_search(self, query: str, k: int = 5, filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar documents using cosine similarity"""
        try:
            total_docs = len(self.documents)
            if total_docs == 0:
                return []
            
            # Generate query embedding
            query_embedding = self._generate_simple_embeddings([query]).astype(np.float32)
            
            if FAISS_AVAILABLE and self.index is not None and self.index.ntotal > 0:
                faiss.normalize_L2(query_embedding)
                search_k = min(total_docs, max(k * 3, 20))  # Retrieve more candidates before filtering
                scores, indices = self.index.search(query_embedding, search_k)
                candidate_pairs = [(float(scores[0][i]), int(indices[0][i])) for i in range(len(indices[0])) if indices[0][i] >= 0]
            else:
                # Pure NumPy Cosine Similarity fallback
                if self.embeddings is None or len(self.embeddings) == 0:
                    return []
                # Compute dot products with normalized vectors
                sim_scores = np.dot(self.embeddings, query_embedding[0])
                ranked_indices = np.argsort(-sim_scores)
                candidate_pairs = [(float(sim_scores[idx]), int(idx)) for idx in ranked_indices]
            
            # Apply metadata filters and format results
            results = []
            for score, idx in candidate_pairs:
                if idx < total_docs:
                    doc_metadata = self.metadata[idx] if idx < len(self.metadata) else {}
                    
                    if filter_metadata:
                        if self._matches_filter(doc_metadata, filter_metadata):
                            results.append({
                                'content': self.documents[idx],
                                'score': score,
                                'metadata': doc_metadata
                            })
                    else:
                        results.append({
                            'content': self.documents[idx],
                            'score': score,
                            'metadata': doc_metadata
                        })
                    
                    if len(results) >= k:
                        break
            
            # If strict filter returned too few results, fallback to general search so RAG never Starves
            if len(results) == 0 and filter_metadata:
                logger.debug("Filter yielded 0 results; falling back to top unfiltered documents")
                for score, idx in candidate_pairs[:k]:
                    if idx < total_docs:
                        results.append({
                            'content': self.documents[idx],
                            'score': score,
                            'metadata': self.metadata[idx] if idx < len(self.metadata) else {}
                        })
            
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def _generate_simple_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate high-quality semantic hash embeddings using normalized term-frequency vectors"""
        embeddings = []
        
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            embedding = np.zeros(self.dimension, dtype=np.float32)
            
            if not words:
                embeddings.append(embedding)
                continue
            
            for word in words:
                # Stable hashing using MD5
                hash_val = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
                pos = hash_val % self.dimension
                # Weight by word length / significance
                embedding[pos] += 1.0 + min(len(word) * 0.1, 1.0)
            
            # Normalize vector to unit length for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            embeddings.append(embedding)
        
        return np.array(embeddings, dtype=np.float32)
    
    def _split_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split document into chunks"""
        content = doc.get('content', '').strip()
        if len(content) <= 1000:
            return [doc]
        
        chunks = []
        chunk_size = 1000
        overlap = 200
        step = max(1, chunk_size - overlap)
        
        for i in range(0, len(content), step):
            chunk_content = content[i:i + chunk_size]
            chunk = {
                'content': chunk_content,
                'metadata': doc.get('metadata', {}).copy()
            }
            chunk['metadata']['chunk_index'] = i // step
            chunks.append(chunk)
        
        return chunks
    
    def _matches_filter(self, metadata: Dict, filter_metadata: Dict) -> bool:
        """Check if metadata matches filter criteria, supporting list values and optional location"""
        for key, value in filter_metadata.items():
            # If filtering by location but document does not have location (e.g. general WHO guidelines),
            # general guidelines should NOT be discarded.
            if key == 'location':
                if 'location' in metadata and metadata['location'] != value:
                    return False
                continue

            if key not in metadata:
                return False
            
            doc_val = metadata[key]
            # If filter condition is a list of acceptable values
            if isinstance(value, (list, tuple, set)):
                if doc_val not in value:
                    return False
            elif isinstance(doc_val, (list, tuple, set)):
                if value not in doc_val:
                    return False
            elif doc_val != value:
                return False

        return True
    
    async def _save_data(self):
        """Save vector store data to disk"""
        try:
            data = {
                'documents': self.documents,
                'metadata': self.metadata,
                'timestamp': datetime.utcnow().isoformat()
            }
            os.makedirs('data', exist_ok=True)
            with open('data/vector_store.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving vector store data: {e}")
    
    async def _load_existing_data(self):
        """Load existing data from disk"""
        try:
            if os.path.exists('data/vector_store.json'):
                with open('data/vector_store.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.documents = data.get('documents', [])
                self.metadata = data.get('metadata', [])
                
                if self.documents:
                    embeddings = self._generate_simple_embeddings(self.documents).astype(np.float32)
                    if FAISS_AVAILABLE and self.index is not None:
                        faiss.normalize_L2(embeddings)
                        self.index.add(embeddings)
                    else:
                        self.embeddings = embeddings
                    logger.info(f"Loaded {len(self.documents)} existing documents into vector store")
        except Exception as e:
            logger.error(f"Error loading existing data: {e}")
    
    async def health_check(self):
        """Check vector store health"""
        return True
