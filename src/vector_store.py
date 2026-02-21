import os
import uuid
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)

from langchain_core.documents import Document


class QdrantVectorStore:
    """
    Qdrant vector store implementation with session-based filtering.
    """

    def __init__(self):
        """Initialize Qdrant client and collection"""

        # Initialize Qdrant client
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )

        # Collection configuration
        self.collection_name = os.getenv("COLLECTION_NAME", "rag_documents")
        self.vector_size = 768  # nomic-embed-text embedding dimension

        # Ensure collection exists
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Create collection and payload indexes if not present"""
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                print(f"Created collection: {self.collection_name}")

            # session_id index
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="session_id",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                print("Created session_id index")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    print(e)

            # source_type index
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="source_type",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                print("Created source_type index")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    print(e)

        except Exception as e:
            print(f"Error ensuring collection exists: {e}")
            raise

    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> bool:
        """Add documents with embeddings to Qdrant"""
        try:
            points = []

            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                if not embedding:
                    continue

                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "content": doc.page_content,
                        "source_type": doc.metadata.get("source_type"),
                        "source_name": doc.metadata.get("source_name"),
                        "session_id": doc.metadata.get("session_id"),
                        "chunk_id": doc.metadata.get("chunk_id", i),
                    },
                )
                points.append(point)

            if not points:
                print("No valid points to upload")
                return False

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            print(f"Successfully added {len(points)} documents to vector store")
            return True

        except Exception as e:
            print(f"Error adding documents: {e}")
            return False

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Similarity search using latest Qdrant API"""

        try:
            if not query_embedding:
                return []

            query_filter = None
            if filter_dict:
                conditions = []
                for key, value in filter_dict.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value),
                        )
                    )
                query_filter = Filter(must=conditions)

            # ✅ NEW API (IMPORTANT FIX)
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=query_filter,
                limit=k,
            )

            documents = []
            for result in search_results.points:
                documents.append(
                    Document(
                        page_content=result.payload["content"],
                        metadata={
                            "source_type": result.payload.get("source_type"),
                            "source_name": result.payload.get("source_name"),
                            "session_id": result.payload.get("session_id"),
                            "chunk_id": result.payload.get("chunk_id"),
                            "score": result.score,
                        },
                    )
                )

            return documents

        except Exception as e:
            print(f"Error performing similarity search: {e}")
            return []

    def get_documents_by_session(self, session_id: str) -> List[dict]:
        """Fetch all documents for a session"""
        try:
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="session_id",
                            match=MatchValue(value=session_id),
                        )
                    ]
                ),
                limit=1000,
            )

            docs = []
            for p in points:
                docs.append(
                    {
                        "id": p.id,
                        "content": p.payload.get("content"),
                        "source_type": p.payload.get("source_type"),
                        "source_name": p.payload.get("source_name"),
                        "chunk_id": p.payload.get("chunk_id"),
                    }
                )

            return docs

        except Exception as e:
            print(f"Error getting documents by session: {e}")
            return []

    def delete_by_session(self, session_id: str) -> bool:
        """Delete all documents of a session"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="session_id",
                            match=MatchValue(value=session_id),
                        )
                    ]
                ),
            )
            print(f"Deleted documents for session: {session_id}")
            return True

        except Exception as e:
            print(f"Error deleting session documents: {e}")
            return False

    def get_collection_info(self) -> dict:
        """Return collection statistics"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vector_size": info.config.params.vectors.size,
                "points_count": info.points_count,
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {}
