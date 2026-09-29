from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

load_dotenv()

DEFAULT_COLLECTION = 'rag_eval_docs'
DEFAULT_GEMINI_MODEL = 'gemini-1.5-flash'
DEFAULT_GROQ_MODEL = 'openai/gpt-oss-20b'
DEFAULT_TOP_K = 5
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 120


def classify_missing_info(question: str, context_text: str) -> dict[str, Any]:
    normalized_context = (context_text or '').lower()
    if not normalized_context or 'not available in the document' in normalized_context or 'does not mention' in normalized_context:
        return {
            'is_answerable': False,
            'reason': 'The information is not available in the document, so it cannot be answered accurately.',
        }
    return {
        'is_answerable': True,
        'reason': 'The answer is grounded in the retrieved document context.',
    }


def build_reference_block(references: list[dict[str, Any]]) -> str:
    if not references:
        return 'No references available in the retrieved context.'

    blocks: list[str] = []
    for item in references:
        source = item.get('source', 'Unknown source')
        page = item.get('page')
        snippet = (item.get('text') or '').strip()
        label = f'{source} — Page {page}' if page is not None else source
        blocks.append(f'- {label}: {snippet[:220]}...')
    return '\n'.join(blocks)


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    cleaned = ' '.join(text.split())
    if len(cleaned) <= chunk_size:
        return [cleaned] if cleaned else []

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunk = cleaned[start:end]
        if end < len(cleaned):
            last_space = chunk.rfind(' ')
            if last_space > 0:
                chunk = chunk[:last_space]
                end = start + len(chunk)
        chunks.append(chunk.strip())
        if end >= len(cleaned):
            break
        start = max(0, end - overlap)
    return [item for item in chunks if item]


def extract_pdf_text(pdf_path: Path) -> list[dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    chunks: list[dict[str, Any]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ''
        for chunk_index, chunk in enumerate(chunk_text(page_text), start=1):
            chunks.append(
                {
                    'source': pdf_path.name,
                    'page': page_number,
                    'chunk_index': chunk_index,
                    'text': chunk,
                }
            )
    return chunks


def load_pdf_documents(pdf_dir: Path) -> list[dict[str, Any]]:
    if not pdf_dir.exists():
        return []

    pdf_files = sorted(pdf_dir.glob('*.pdf'))
    if not pdf_files:
        return []

    documents: list[dict[str, Any]] = []
    for pdf_file in pdf_files:
        documents.extend(extract_pdf_text(pdf_file))
    return documents


def get_qdrant_client() -> QdrantClient:
    qdrant_url = os.getenv('QDRANT_URL') or 'http://localhost:6333'
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    if qdrant_api_key:
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    return QdrantClient(url=qdrant_url)


def build_embedding_model() -> SentenceTransformer:
    return SentenceTransformer('all-MiniLM-L6-v2')


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int, reset: bool = False) -> None:
    exists = client.collection_exists(collection_name)
    if exists and reset:
        client.delete_collection(collection_name)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def index_documents(documents: list[dict[str, Any]], collection_name: str = DEFAULT_COLLECTION, reset: bool = False) -> dict[str, Any]:
    if not documents:
        return {'collection_name': collection_name, 'chunks_indexed': 0, 'sources': []}

    client = get_qdrant_client()
    embedder = build_embedding_model()
    vector_size = 384
    ensure_collection(client, collection_name, vector_size, reset=reset)

    texts = [item['text'] for item in documents]
    embeddings = embedder.encode(texts)
    points: list[PointStruct] = []
    for idx, (doc, embedding) in enumerate(zip(documents, embeddings), start=1):
        points.append(
            PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload={
                    'source': doc['source'],
                    'page': doc['page'],
                    'chunk_index': doc['chunk_index'],
                    'text': doc['text'],
                },
            )
        )

    for start in range(0, len(points), 50):
        client.upsert(collection_name=collection_name, points=points[start:start + 50], timeout=120)

    sources = sorted({doc['source'] for doc in documents})
    return {'collection_name': collection_name, 'chunks_indexed': len(points), 'sources': sources}


def semantic_search(question: str, collection_name: str = DEFAULT_COLLECTION, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
    client = get_qdrant_client()
    embedder = build_embedding_model()
    query_vector = embedder.encode(question).tolist()
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    ).points

    matches: list[dict[str, Any]] = []
    for result in results:
        payload = result.payload or {}
        matches.append(
            {
                'source': str(payload.get('source', 'unknown')),
                'page': payload.get('page'),
                'chunk_index': payload.get('chunk_index'),
                'text': str(payload.get('text', '')),
                'score': round(float(result.score), 4) if getattr(result, 'score', None) is not None else None,
            }
        )
    return matches


def get_llm_config(model_name: str | None = None) -> dict[str, str]:
    configured_model = (model_name or os.getenv('LLM_MODEL') or '').strip()
    gemini_key = os.getenv('GEMINI_API_KEY')
    groq_key = os.getenv('GROQ_API_KEY')

    if configured_model:
        normalized = configured_model.lower()
        if normalized.startswith('gemini'):
            return {'provider': 'gemini', 'model': configured_model}
        if normalized.startswith('llama') or normalized.startswith('meta-llama') or normalized.startswith('deepseek') or 'groq' in normalized:
            return {'provider': 'groq', 'model': configured_model}

    if gemini_key:
        return {'provider': 'gemini', 'model': configured_model or DEFAULT_GEMINI_MODEL}
    if groq_key:
        return {'provider': 'groq', 'model': configured_model or DEFAULT_GROQ_MODEL}

    if configured_model:
        return {'provider': 'gemini', 'model': configured_model}
    return {'provider': 'gemini', 'model': DEFAULT_GEMINI_MODEL}


def call_llm(question: str, context: list[dict[str, Any]], model_name: str | None = None) -> tuple[str, dict[str, Any]]:
    config = get_llm_config(model_name)
    provider = config['provider']
    model = config['model']

    references_text = '\n'.join(
        f'Source: {item["source"]} | Page {item["page"]} | Snippet: {item["text"]}' for item in context
    )

    prompt = f'''
You are answering using only the provided source context.

Question: {question}

Context:
{references_text}

Rules:
1. Answer using only the provided context.
2. If the information is not present in the document, say exactly: 'The information is not available in the document, so it cannot be answered accurately.' Then explain why the answer is missing from the source.
3. Do not invent facts or answer beyond the document.
4. Include a 'References' section at the end with source names and page numbers in the exact context you used.
5. Keep the final answer concise but grounded in the source text.
'''

    start_time = time.perf_counter()
    if provider == 'groq':
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            raise RuntimeError('GROQ_API_KEY is missing. Add it to your .env file before running an LLM query.')

        client = Groq(api_key=groq_api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.1,
            max_tokens=1024,
        )
        elapsed = time.perf_counter() - start_time
        usage = getattr(response, 'usage', None)
        usage_payload = {
            'input_tokens': getattr(usage, 'prompt_tokens', 0) or 0,
            'output_tokens': getattr(usage, 'completion_tokens', 0) or 0,
            'thinking_tokens': getattr(usage, 'reasoning_tokens', 0) or 0,
            'total_tokens': getattr(usage, 'total_tokens', 0) or 0,
            'time_seconds': round(elapsed, 3),
        }
        answer_text = response.choices[0].message.content or ''
        return answer_text, usage_payload

    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        raise RuntimeError('GEMINI_API_KEY is missing. Add it to your .env file before running an LLM query.')

    genai.configure(api_key=gemini_api_key)
    llm_model = genai.GenerativeModel(model)
    response = llm_model.generate_content(prompt)
    elapsed = time.perf_counter() - start_time

    usage = getattr(response, 'usage_metadata', None)
    usage_payload = {
        'input_tokens': getattr(usage, 'prompt_token_count', 0) or 0,
        'output_tokens': getattr(usage, 'candidates_token_count', 0) or 0,
        'thinking_tokens': getattr(usage, 'thoughts_token_count', 0) or 0,
        'total_tokens': getattr(usage, 'total_token_count', 0) or 0,
        'time_seconds': round(elapsed, 3),
    }

    answer_text = getattr(response, 'text', '') or ''
    return answer_text, usage_payload


def evaluate_question(question: str, collection_name: str = DEFAULT_COLLECTION, top_k: int = DEFAULT_TOP_K, model_name: str | None = None) -> dict[str, Any]:
    matches = semantic_search(question, collection_name=collection_name, top_k=top_k)
    context = matches if matches else []

    if not context:
        answer_text = 'The information is not available in the document, so it cannot be answered accurately. No relevant passages were found in the indexed source material.'
        return {
            'question': question,
            'answer': answer_text,
            'references': [],
            'benchmark': {'input_tokens': 0, 'output_tokens': 0, 'thinking_tokens': 0, 'total_tokens': 0, 'time_seconds': 0.0},
            'grounded': False,
        }

    answer_text, benchmark = call_llm(question, context, model_name=model_name)
    answer_text = answer_text.strip()
    classification = classify_missing_info(question, '\n'.join(item['text'] for item in context))

    result = {
        'question': question,
        'answer': answer_text,
        'references': context,
        'benchmark': benchmark,
        'grounded': classification['is_answerable'],
    }
    return result


def load_default_questions(path: Path | str | None = None) -> list[dict[str, str]]:
    default_path = Path(__file__).resolve().parent / 'evaluation_questions.json'
    question_path = Path(path) if path is not None else default_path
    if not question_path.exists():
        return [
            {'question': 'What working model does the company follow?'},
            {'question': 'What skills are explicitly required for the role?'},
            {'question': 'What is the company expected to deliver from day one?'},
            {'question': 'What should a candidate know about AI tools or automation?'},
            {'question': 'What is the capital of France?'},
        ]
    with question_path.open('r', encoding='utf-8') as fh:
        data = json.load(fh)
    return data


def run_evaluation(pdf_dir: str | Path | None = None, collection_name: str = DEFAULT_COLLECTION, top_k: int = DEFAULT_TOP_K, model_name: str | None = None) -> list[dict[str, Any]]:
    docs_dir = Path(pdf_dir) if pdf_dir else Path(__file__).resolve().parent / 'pdfs'
    documents = load_pdf_documents(docs_dir)
    if not documents:
        raise FileNotFoundError(f'No PDF files were found in {docs_dir}. Add a PDF document there or pass --pdf-dir with the folder containing the source PDF.')

    index_result = index_documents(documents, collection_name=collection_name, reset=True)
    questions = load_default_questions()
    results: list[dict[str, Any]] = []
    for item in questions:
        question = item.get('question', '')
        if not question:
            continue
        result = evaluate_question(question, collection_name=collection_name, top_k=top_k, model_name=model_name)
        result['index_summary'] = index_result
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description='Grounded PDF Q&A with Qdrant and Gemini.')
    parser.add_argument('--pdf-dir', default=str(Path(__file__).resolve().parent / 'pdfs'), help='Directory containing source PDFs.')
    parser.add_argument('--question', help='Ask a single question against the indexed PDF content.')
    parser.add_argument('--evaluate', action='store_true', help='Run the evaluation set and print benchmark results.')
    parser.add_argument('--collection-name', default=DEFAULT_COLLECTION, help='Qdrant collection name.')
    parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K, help='Number of retrieved chunks to include as context.')
    parser.add_argument('--model', default=None, help='Override the LLM model. If omitted, the app auto-selects Gemini or Groq based on available API keys.')
    parser.add_argument('--reset', action='store_true', help='Reset the collection before indexing.')
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    documents = load_pdf_documents(pdf_dir)
    if not documents:
        print(f'No PDF files were found in {pdf_dir}. Add a PDF or pass a valid --pdf-dir path.')
        return

    index_summary = index_documents(documents, collection_name=args.collection_name, reset=args.reset)
    print(f"Indexed {index_summary['chunks_indexed']} chunks from: {', '.join(index_summary['sources'])}")

    if args.question:
        result = evaluate_question(args.question, collection_name=args.collection_name, top_k=args.top_k, model_name=args.model)
        print('\nAnswer:')
        print(result['answer'])
        print('\nReferences:')
        print(build_reference_block(result['references']))
        print('\nBenchmark:')
        print(json.dumps(result['benchmark'], indent=2))
        return

    if args.evaluate:
        questions = load_default_questions()
        for item in questions:
            question = item.get('question', '')
            if not question:
                continue
            result = evaluate_question(question, collection_name=args.collection_name, top_k=args.top_k, model_name=args.model)
            print(f'\nQ: {question}')
            print(f"A: {result['answer']}")
            print(f"Refs: {build_reference_block(result['references'])}")
            print(f"Benchmark: {json.dumps(result['benchmark'], indent=2)}")
        return

    print("Project ready. Use --question 'your question here' or --evaluate to run a benchmark.")


if __name__ == '__main__':
    main()
