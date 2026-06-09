from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings


class ChromaPolicyStore:
    """Chroma-backed vector store cho policy chunks."""

    def __init__(
        self,
        persist_directory: Path,
        embedding_model: Any,
        collection_name: str = "policy_chunks",
    ) -> None:
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def ensure_index(self, markdown_path: Path) -> None:
        """Rebuild index nếu collection rỗng."""
        if self.collection.count() == 0:
            self.rebuild(markdown_path)

    def rebuild(self, markdown_path: Path) -> None:
        """Parse markdown, embed, và add vào Chroma collection."""
        from rag.parser import parse_policy_markdown

        with open(markdown_path, encoding="utf-8") as f:
            text = f.read()

        chunks = parse_policy_markdown(text)
        if not chunks:
            return

        ids = [f"chunk_{i}" for i in range(len(chunks))]
        documents = [c["rendered_text"] for c in chunks]
        metadatas = [
            {
                "section_h2": c["section_h2"],
                "section_h3": c["section_h3"],
                "citation": c["citation"],
            }
            for c in chunks
        ]

        # Embed toàn bộ documents
        embeddings = self.embedding_model.embed_documents(documents)

        # Xoá collection cũ và tạo lại
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Add batch
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        """Tìm kiếm policy chunks liên quan đến query.

        Trả về list các hit: {citation, content, distance}
        """
        query_emb = self.embedding_model.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[dict[str, Any]] = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "citation": results["metadatas"][0][i]["citation"],
                "content": results["documents"][0][i],
                "distance": results["distances"][0][i],
            })
        return hits
