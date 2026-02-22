#!/usr/bin/env python
"""
Example client script for Smart Multi-Doc RAG API

Shows how to:
- Upload documents
- Query the system
- Manage documents
- Get statistics
"""

import requests
import json
from pathlib import Path
from typing import Optional


class RAGClient:
    """Client for interacting with Smart RAG API"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def health_check(self) -> dict:
        """Check if server is healthy"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def upload_document(self, file_path: str) -> dict:
        """
        Upload a document

        Args:
            file_path: Path to document (txt, pdf, or docx)

        Returns:
            Upload response
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as f:
            files = {"file": f}
            response = self.session.post(f"{self.base_url}/upload", files=files)

        response.raise_for_status()
        return response.json()

    def query(self, query_text: str) -> dict:
        """
        Query the documents

        Args:
            query_text: Question to ask

        Returns:
            Query response with context
        """
        params = {"query_text": query_text}
        response = self.session.post(f"{self.base_url}/query", params=params)
        response.raise_for_status()
        return response.json()

    def list_documents(self) -> dict:
        """Get list of all documents"""
        response = self.session.get(f"{self.base_url}/documents")
        response.raise_for_status()
        return response.json()

    def delete_document(self, doc_id: int) -> dict:
        """Delete a document by ID"""
        response = self.session.delete(f"{self.base_url}/documents/{doc_id}")
        response.raise_for_status()
        return response.json()

    def get_stats(self) -> dict:
        """Get system statistics"""
        response = self.session.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()


def print_header(text: str):
    """Print formatted header"""
    print()
    print("=" * 60)
    print(text)
    print("=" * 60)


def print_response(data: dict):
    """Pretty print JSON response"""
    print(json.dumps(data, indent=2))


def main():
    """Example usage of RAG client"""

    # Create client
    client = RAGClient()

    # 1. Health check
    print_header("1. Health Check")
    try:
        health = client.health_check()
        print(f"Server Status: {health['status']}")
        print(f"Service: {health['service']}")
        print(f"Version: {health['version']}")
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to server.")
        print("Make sure server is running: python main.py")
        return

    # 2. List documents (before upload)
    print_header("2. List Documents (Initial)")
    docs = client.list_documents()
    print(f"Total documents: {docs['count']}")
    if docs["documents"]:
        print("\nDocuments:")
        for doc in docs["documents"]:
            print(f"  - {doc['filename']} ({doc['total_chunks']} chunks)")

    # 3. Upload example documents
    print_header("3. Upload Documents")
    example_docs = [
        "Documents/distributed_systems_failure.txt",
        "Documents/latency_and_throughput.txt",
    ]

    for doc_path in example_docs:
        if Path(doc_path).exists():
            print(f"\nUploading: {doc_path}")
            try:
                result = client.upload_document(doc_path)
                print(f"  Status: {result['status']}")
                print(f"  Chunks: {result['chunks']}")
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print(f"\nSkipping: {doc_path} (not found)")

    # 4. List documents (after upload)
    print_header("4. List Documents (After Upload)")
    docs = client.list_documents()
    print(f"Total documents: {docs['count']}")
    for doc in docs["documents"][:3]:  # Show first 3
        print(f"\n  {doc['filename']}")
        print(f"    Type: {doc['file_type']}")
        print(f"    Chunks: {doc['total_chunks']}")
        print(f"    Uploaded: {doc['uploaded_at']}")

    # 5. Query examples
    print_header("5. Sample Queries")

    queries = [
        "What is latency?",
        "How does distributed systems handle failures?",
        "Explain throughput optimization",
    ]

    for i, q in enumerate(queries, 1):
        print(f"\nQuery {i}: {q}")
        print("-" * 40)
        try:
            result = client.query(q)
            print(f"Intent: {result['intent']}")
            print(f"Chunks Retrieved: {result['retrieved_chunks']}")
            print(f"Response Time: {result['response_time_ms']:.1f}ms")
            print(f"\nContext (first 300 chars):")
            context = result["context"]
            print(context[:300] + "..." if len(context) > 300 else context)
        except Exception as e:
            print(f"Error: {e}")

    # 6. Statistics
    print_header("6. System Statistics")
    stats = client.get_stats()
    s = stats["statistics"]
    print(f"Total Documents: {s['total_documents']}")
    print(f"Total Chunks: {s['total_chunks']}")
    print(f"Total Queries: {s['total_queries']}")
    print(f"Average Response Time: {s['average_response_time_ms']:.1f}ms")
    print(f"\nConfiguration:")
    print(f"  Chunk Size: {s['config']['chunk_size']} words")
    print(f"  Overlap: {s['config']['overlap']} words")
    print(f"  Max Final Chunks: {s['config']['max_final_chunks']}")

    # 7. Demonstrate deletion
    print_header("7. Document Management")
    docs = client.list_documents()
    if docs["documents"]:
        # Note: Uncomment to actually delete a document
        # doc_to_delete = docs["documents"][0]["id"]
        # print(f"Deleting document ID {doc_to_delete}...")
        # result = client.delete_document(doc_to_delete)
        # print(f"Result: {result['message']}")
        print("(Deletion example - uncomment in code to use)")
    else:
        print("No documents to delete")

    print_header("Complete!")
    print("Explore more at: http://localhost:8000/docs")


if __name__ == "__main__":
    main()
