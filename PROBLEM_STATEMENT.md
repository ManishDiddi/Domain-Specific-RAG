Build a Domain Specific RAG
XYZ AI Solutions manages an extensive ecosystem of developer tools. Currently, our internal support engineers and community managers spend over 40% of their day navigating fragmented documentation to answer technical questions about HuggingFace-based workflows. The existing keyword search is insufficient, often failing to capture the nuance of complex machine learning queries, leading to high support ticket latency and inconsistent developer advice.

The Goal: Your goal is to implement a robust, end-to-end RAG pipeline. In an industry setting, success is measured not just by code execution, but by the reliability and relevance of the retrieved information.

The Solution: As part of the RAG (Retrieval-Augmented Generation) task force, you are tasked with building a Production-Grade Knowledge Retrieval System. This system must ingest documentation, store it in a high-performance vector database, and provide context-aware answers to complex queries using an LLM.


Evaluation Criteria

Data Processing & Chunking Logic: Quality of the sliding-window implementation, specifically the precision of chunk_size and overlap to maintain semantic continuity while preserving metadata lineage.

Vectorization & Memory Management: Efficiency of the batched embedding pipeline and the correct application of vector normalization to optimize for high-performance similarity search.

Vector DB Architecture & Persistence: Technical accuracy in configuring the Milvus collection schema, metric types (IP), and the robustness of the data ingestion workflow.

Retrieval Precision: Ability to accurately map natural language queries to high-dimensional vector space and isolate the most relevant context snippets using top_k parameters.

Grounded Synthesis & Hallucination Control: Effectiveness of prompt engineering in restricting the LLM to provided context and triggering "insufficient information" guardrails for out-of-scope queries.

Pipeline Integration & Reasoning: Logical consistency of the end-to-end RAG workflow and the technical accuracy of generated answers relative to the source documentation.

Automated QA & Metric Interpretation: Depth of analysis regarding Opik evaluation scores (Relevance and Hallucination) and the ability to diagnose pipeline failures.

Technical Rigor & Reproducibility: Clarity and efficiency of code execution, with a clear justification for design trade-offs regarding model selection and system latency.

  

Your role
Expert in creating RAG pipelines

Dataset
RAG BCS Dataset