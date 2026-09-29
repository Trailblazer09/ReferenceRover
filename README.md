<h1 align="center"> Reference Rover - A PDF RAG Evaluation Harness 
</h1>

```mermaid
flowchart TD

subgraph group_ingestion["Ingestion and indexing"]
  node_pdfs["Source PDFs"]
  node_pdf_loader["PDF extraction<br/>[rag_evaluation.py]"]
  node_chunking["Text chunking<br/>[rag_evaluation.py]"]
  node_embedding["Embeddings<br/>[rag_evaluation.py]"]
  node_indexing["Document indexing<br/>[rag_evaluation.py]"]
end

subgraph group_answering["Answering and evaluation"]
  node_runtime["CLI workflow<br/>[rag_evaluation.py]"]
  node_questions["Evaluation questions"]
  node_evaluation["Question evaluation<br/>[rag_evaluation.py]"]
  node_search["Semantic search<br/>[rag_evaluation.py]"]
  node_grounding["Grounding check<br/>[rag_evaluation.py]"]
  node_llm_call["LLM call<br/>[rag_evaluation.py]"]
  node_results["Answers and metrics"]
end

subgraph group_services["External services"]
  node_qdrant[("Qdrant")]
  node_providers{{"Gemini or Groq"}}
end

node_user(("User"))
node_cli["CLI entry<br/>[main.py]"]

node_user -->|"runs"| node_cli
node_cli -->|"invokes"| node_runtime
node_runtime -->|"loads PDFs"| node_pdf_loader
node_pdfs -->|"provides files"| node_pdf_loader
node_pdf_loader -->|"extracts text"| node_chunking
node_runtime -->|"indexes documents"| node_indexing
node_indexing -->|"embeds chunks"| node_embedding
node_indexing -->|"writes vectors"| node_qdrant
node_runtime -->|"evaluates questions"| node_evaluation
node_questions -->|"supplies questions"| node_runtime
node_evaluation -->|"retrieves context"| node_search
node_search -->|"embeds query"| node_embedding
node_search -->|"searches vectors"| node_qdrant
node_evaluation -->|"requests answer"| node_llm_call
node_llm_call -->|"sends prompt"| node_providers
node_evaluation -->|"checks context"| node_grounding
node_evaluation -->|"returns answer and metrics"| node_results
node_runtime -->|"prints references"| node_results

click node_cli "https://github.com/trailblazer09/referencerover/blob/main/main.py"
click node_runtime "https://github.com/trailblazer09/referencerover/blob/main/rag_evaluation.py"
click node_pdfs "https://github.com/trailblazer09/referencerover/tree/main/pdfs"
click node_pdf_loader "https://github.com/trailblazer09/referencerover/blob/main/rag_evaluation.py"
click node_chunking "https://github.com/trailblazer09/referencerover/blob/main/rag_evaluation.py"
click node_embedding "https://github.com/trailblazer09/referencerover/blob/main/rag_evaluation.py"
click node_indexing "https://github.com/trailblazer09/referencerover/blob/main/rag_evaluation.py"
click node_questions "https://github.com/trailblazer09/referencerover/blob/main/evaluation_questions.json"
click node_evaluation "https://github.com/trailblazer09/referencerover/blob/main/rag_evaluation.py"
click node_search "https://github.com/trailblazer09/referencerover/blob/main/rag_evaluation.py"
click node_grounding "https://github.com/trailblazer09/referencerover/blob/main/rag_evaluation.py"
click node_llm_call "https://github.com/trailblazer09/referencerover/blob/main/rag_evaluation.py"

classDef toneNeutral fill:#f8fafc,stroke:#334155,stroke-width:1.5px,color:#0f172a
classDef toneBlue fill:#dbeafe,stroke:#2563eb,stroke-width:1.5px,color:#172554
classDef toneAmber fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#78350f
classDef toneMint fill:#dcfce7,stroke:#16a34a,stroke-width:1.5px,color:#14532d
classDef toneRose fill:#ffe4e6,stroke:#e11d48,stroke-width:1.5px,color:#881337
classDef toneIndigo fill:#e0e7ff,stroke:#4f46e5,stroke-width:1.5px,color:#312e81
classDef toneTeal fill:#ccfbf1,stroke:#0f766e,stroke-width:1.5px,color:#134e4a
class node_pdfs,node_pdf_loader,node_chunking,node_embedding,node_indexing,node_user toneBlue
class node_runtime,node_questions,node_evaluation,node_search,node_grounding,node_llm_call,node_results toneAmber
class node_qdrant,node_providers toneMint
class node_cli toneTeal
```
---
This project is a command-line PDF RAG evaluation harness. A user supplies a question or requests an evaluation run. It:

- extracts text from one or more PDF files,
- chunks and embeds the content,
- stores the vectors in Qdrant,
- answers questions using the document as the sole source of truth,
- shows references and page numbers for each answer,
- benchmarks one-turn tokens and latency for the LLM call.
---
<div align="center">
<h2/>Scenario 1: The harness explains that it cannot answer when there is no relevant information inside the attached document. </h2>
  <img width="600" height="350" alt="Screenshot 2026-09-29 210241" src="https://github.com/user-attachments/assets/c243879e-1fa6-4dcf-aebe-bcb1018facaf" />
</div>

<div align="center">
<h2/>Scenario 2: The harness returns grounded responses along with source references and token usage metrics. </h2>
<img width="600" height="350" alt="Screenshot 2026-09-29 210508" src="https://github.com/user-attachments/assets/b3c2e0b5-e811-49f6-b8e1-d4d2e86abb66" />
   </div>


---
## Setup

1. Create a virtual environment.
2. Install dependencies:
   python -m pip install -r requirements.txt
3. Copy `.env.example` to `.env` and add your values:
   - `QDRANT_URL` (local or cloud)
   - `QDRANT_API_KEY` if using Qdrant Cloud
   - either `GEMINI_API_KEY` for Gemini or `GROQ_API_KEY` for Groq
   - optional `LLM_MODEL` to override the default model
4. Put your source PDF files in the `pdfs/` directory.

## Usage

Ask a single question:
python main.py --pdf-dir pdfs --question "What are the key requirements for the role?"

Run the evaluation set:
python main.py --pdf-dir pdfs --evaluate

## Notes

- The app is designed to answer only from the source document.
- If the answer is not present in the PDF, it clearly states that the document does not provide enough information.
- Token and latency metrics are reported for each single LLM turn.
- `thinking_tokens` is reported as `0` when the provider does not expose reasoning-token metadata.

## Files

- `main.py` — CLI entry point
- `rag_evaluation.py` — indexing, retrieval, answer generation, and benchmarking logic
- `evaluation_questions.json` — example evaluation set
- `pdfs/` — place your source PDFs here
