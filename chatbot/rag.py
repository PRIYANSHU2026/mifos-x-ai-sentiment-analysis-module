import os
import glob
import logging

logger = logging.getLogger(__name__)

# Fallback basic RAG if chromadb isn't installed
try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("chromadb not installed. RAG will fallback to basic text search. Run: pip install chromadb")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "chatbot", "chroma_db")

class KnowledgeBase:
    def __init__(self):
        self.is_ready = False
        if CHROMA_AVAILABLE:
            try:
                self.client = chromadb.PersistentClient(path=DB_PATH)
                self.collection = self.client.get_or_create_collection(name="mifos_knowledge")
                self.is_ready = True
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB: {e}")

    def index_project_files(self):
        """Indexes all markdown files in the project."""
        if not self.is_ready:
            return

        documents = []
        ids = []
        metadatas = []
        
        # We index README, architecture overview, and all files in reports/
        search_patterns = [
            os.path.join(BASE_DIR, "*.md"),
            os.path.join(BASE_DIR, "reports", "*.md"),
        ]
        
        doc_id = 0
        for pattern in search_patterns:
            for file_path in glob.glob(pattern):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                        # Simple chunking by headers
                        chunks = content.split("\n## ")
                        for idx, chunk in enumerate(chunks):
                            if len(chunk.strip()) > 20:
                                documents.append(chunk)
                                ids.append(f"doc_{doc_id}_{idx}")
                                metadatas.append({"source": os.path.basename(file_path)})
                    doc_id += 1
                except Exception as e:
                    logger.warning(f"Could not read {file_path}: {e}")

        if documents:
            # We let ChromaDB use its default embedding model if no embedding function is provided.
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Indexed {len(documents)} chunks into ChromaDB.")

    def search(self, query: str, n_results=3) -> str:
        """Searches the knowledge base for the query and returns a formatted context string."""
        if not self.is_ready:
            return "Knowledge base offline."
            
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            context = ""
            if results and results['documents'] and len(results['documents'][0]) > 0:
                for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                    context += f"\n--- Source: {meta.get('source')} ---\n{doc}\n"
            return context
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return ""

# Singleton instance
kb = KnowledgeBase()

# If running this file directly, perform indexing
if __name__ == "__main__":
    kb.index_project_files()
    print("Indexing complete.")
