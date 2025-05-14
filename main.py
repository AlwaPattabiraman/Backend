from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer,util
import json
from transformers import pipeline, AutoTokenizer, AutoModel
import re
import spacy
from rank_bm25 import BM25Okapi
from transformers import pipeline
from fastapi.responses import JSONResponse
import numpy as np
from typing import List
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import uvicorn
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import SystemMessage, HumanMessage
from langchain.chat_models import ChatOpenAI
from typing import Literal, List
from whoosh.qparser import MultifieldParser, OrGroup
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in, open_dir
from whoosh.qparser import QueryParser
from whoosh.analysis import StemmingAnalyzer
import os, shutil, re
app = FastAPI()
api_key = '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'
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL, e.g., "http://localhost:4200"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

nlp = spacy.load("en_core_web_sm")
# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')
# Create a Pydantic model for the request
class Query5(BaseModel):
    text: str
    chunks: int

# Create a Pydantic model for the response
class SimilarityResult(BaseModel):
    topic_path: str
    content: str
    similarity_score: float
    rank: int

class SimilarityResultFuzzy(BaseModel):
    topic_path: str
    content: str
    similarity_q1: float
    similarity_q2: float
    fuzzy_intersection: float
    fuzzy_union: float
    rank: int

class SimilarityResponse(BaseModel):
    results: List[SimilarityResult]


# Load embeddings data
def load_shakespeare_data():
    try:
        with open('version1_Shakes_final.json', 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        raise Exception("Shakespeare embeddings file not found")
    
def AI_data():
    try:
        with open('Ai_colors.json', 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        raise Exception("Shakespeare embeddings file not found")

def Adam_data():
    try:
        with open('adam-with-embeddings-3.json', 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        raise Exception("Shakespeare embeddings file not found")       

def load_shakespeare_data2():
    try:
        with open('version2_Shakes_final.json', 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        raise Exception("Shakespeare embeddings file not found")
# Ex
# tract all content and embeddings

def concatenate_values(data):
    return " ".join(item["value"] for item in data)

def transform_data(data):
    transformed = {"embeddings": []}
    
    for cluster in data["clusters"].values():
        for item in cluster:
            transformed["embeddings"].extend(item["embeddings"])
    
    return transformed

def extract_content_and_embeddings(data, parent_path=""):
    results = []
    
    for item in data:
        current_path = f"{parent_path}/{item['name']}" if parent_path else item['name']
        
        # Check if content and embeddings are present
        if 'content' in item and item['content'] and 'embeddings' in item:
            content = ''
            if 'content' in item:
                content = item['content']
            elif 'children' in item:
                content = concatenate_values(item['children'])
            else:
                content = item.get('value', '')  # Safely get 'value' if it exists
            
            results.append({
                'path': current_path,
                'content': content,
                'embeddings': item['embeddings']
            })
        else:
            # Check for 'value' key and handle it safely using .get()
            content = item.get('value', '')  # Safely get 'value' if it exists
            
            results.append({
                'path': current_path,
                'content': content,
                'embeddings': item.get('embeddings', None)  # Default to None if 'embeddings' is missing
            })
        
        # Recursively process children if they exist
        if 'children' in item:
            results.extend(extract_content_and_embeddings(item['children'], current_path))
    
    return results

class QueryRequest(BaseModel):
    query: str
    source: str




class Query2(BaseModel):
    query1: str
    query2: str
    chunks: int = 10
    
class SimilarityResponseFuzzy(BaseModel):
    results: List[SimilarityResultFuzzy]



with open("ag_news_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

with open("filtered_file.json", "r", encoding="utf-8") as f:
    metadata2 = json.load(f)

with open("ag_news_embeddings.json", "r", encoding="utf-8") as f:
    embeddings = np.array(json.load(f))  # shape: (n, dim) 

with open("embeddings_output.json", "r", encoding="utf-8") as f:
    embeddings2 = np.array(json.load(f))  # shape: (n, dim) 

class SearchResponse(BaseModel):
    text: str
    label: int
    score: float
    umap_x: float
    query: int
    umap_y: float
    cluster: str

class SearchResponse2(BaseModel):
    text: str
    label: int
    score2: float
    umap_x: float
    query: int
    umap_y: float
    cluster: str    

class search_Sparse(BaseModel):
    text: str
    label: str
    score: float
    umap_x: float
    umap_y: float
    cluster: int

class CombinedSearchResponse(BaseModel):
    text: str
    title: str
    label: int
    score_sparse: float
    score_dense: float
    umap_x: float
    umap_y: float
    cluster: str

class CombinedSearchResponse3(BaseModel):
    text: str
    label: int
    score_sparse: float
    score_dense: float
    umap_x: float
    umap_y: float
    cluster: str

class CombinedSearchResponse2(BaseModel):
    text: str
    label: int
    score_sparse: float
    score_dense: float
    score_sparse2: float
    score_dense2: float
    umap_x: float
    umap_y: float
    cluster: str

class CombinedSearchResponse2(BaseModel):
    title: str
    text: str
    label: int
    score_sparse: float
    score_dense: float
    score_sparse2: float
    score_dense2: float
    umap_x: float
    umap_y: float
    cluster: str
tokenized_corpus = [doc["text"].lower().split() for doc in metadata]
tokenized_corpus2 = [doc["text"].lower().split() for doc in metadata2]
documents = [doc["text"] for doc in metadata]
# Initialize BM25
bm25 = BM25Okapi(tokenized_corpus)

bm25t2 = BM25Okapi(tokenized_corpus2)

INDEX_DIR = "whoosh_index"
schema = Schema(id=ID(stored=True), content=TEXT(stored=True, analyzer=StemmingAnalyzer()))

if os.path.exists(INDEX_DIR):
    shutil.rmtree(INDEX_DIR)
os.mkdir(INDEX_DIR)

ix = create_in(INDEX_DIR, schema)
writer = ix.writer()
for i, doc in enumerate(documents):
    writer.add_document(id=str(i), content=doc)
writer.commit()

def clean_query(query):
    # Remove punctuation (except *) and lowercase
    query = re.sub(r"[^\w\s*]", "", query)
    return query.lower()

@app.get("/search", response_model=List[CombinedSearchResponse])
def search(
    query: str = Query(..., min_length=3),
    top_k: int = 7900,
):
    results = []  # To store the combined results

    # --- Sparse: BM25 ---
    tokenized_query = query.lower().split()
    sparse_scores = bm25.get_scores(tokenized_query) / 100
    sparse_sorted_indices = np.argsort(sparse_scores)[::-1][:top_k]

    # --- Dense: Cosine Similarity ---
    query_embedding = model.encode([query])
    dense_scores = cosine_similarity(query_embedding, embeddings)[0]
    dense_sorted_indices = np.argsort(dense_scores)[::-1][:top_k]

    # --- Full-text BM25 (Whoosh) ---
    fulltext_scores = []
    fulltext_sorted_indices = []
    news = ['World','Sci/Tech','Business','Sports']  
    with ix.searcher() as searcher:
        qp = QueryParser("content", schema=ix.schema)
        q = qp.parse(clean_query(query))
        hits = searcher.search(q, limit=top_k)
        for hit in hits:
            fulltext_scores.append(float(hit.score) / 100)
            fulltext_sorted_indices.append(int(hit["id"]))

    # --- Combine Results from All Types ---
    all_sorted_indices = list(set(sparse_sorted_indices) | set(dense_sorted_indices) | set(fulltext_sorted_indices))
    for idx in all_sorted_indices:
        item = metadata[idx]
        sparse = sparse_scores[idx] if idx in sparse_sorted_indices else 0.0
        dense = dense_scores[idx] if idx in dense_sorted_indices else 0.0
        fulltext = fulltext_scores[fulltext_sorted_indices.index(idx)] if idx in fulltext_sorted_indices else 0.0
        rerank_score = (sparse + dense + fulltext) / 3
        results.append(CombinedSearchResponse(
            title = news["item.label"],
            text=item["text"],
            label=item["label"],
            score_sparse=sparse,
            score_dense=dense,
            umap_x=item["umap_x"],
            umap_y=item["umap_y"],
            cluster=item["cluster"]
        ))

    return results



@app.get("/search2", response_model=List[CombinedSearchResponse2])
def search2(
    query1: str = Query(..., min_length=3),
    query2: str = Query(..., min_length=3),
    top_k: int = 7900,
):
    results = []

    # --- Sparse: BM25 ---
    tokenized_query1 = query1.lower().split()
    sparse_scores1 = bm25.get_scores(tokenized_query1) / 100
    sparse_sorted_indices1 = np.argsort(sparse_scores1)[::-1][:top_k]

    tokenized_query2 = query2.lower().split()
    sparse_scores2 = bm25.get_scores(tokenized_query2) / 100
    sparse_sorted_indices2 = np.argsort(sparse_scores2)[::-1][:top_k]

    
    # --- Dense: Cosine Similarity ---
    query_embedding1 = model.encode([query1])
    dense_scores1 = cosine_similarity(query_embedding1, embeddings)[0]
    dense_sorted_indices1 = np.argsort(dense_scores1)[::-1][:top_k]

    query_embedding2 = model.encode([query2])
    dense_scores2 = cosine_similarity(query_embedding2, embeddings)[0]
    dense_sorted_indices2 = np.argsort(dense_scores2)[::-1][:top_k]

    # --- Full-text BM25 (Whoosh) ---
    fulltext_scores1 = {}
    fulltext_scores2 = {}
    with ix.searcher() as searcher:
        qp = QueryParser("content", schema=ix.schema)

        q1 = qp.parse(clean_query(query1))
        hits1 = searcher.search(q1, limit=top_k)
        for hit in hits1:
            fulltext_scores1[int(hit["id"])] = float(hit.score) / 100

        q2 = qp.parse(clean_query(query2))
        hits2 = searcher.search(q2, limit=top_k)
        for hit in hits2:
            fulltext_scores2[int(hit["id"])] = float(hit.score) / 100
    news = ['World','Sci/Tech','Business','Sports']  
    # --- Combine All Indices ---
    all_indices = set(sparse_sorted_indices1) | set(sparse_sorted_indices2) | \
                  set(dense_sorted_indices1) | set(dense_sorted_indices2) | \
                  set(fulltext_scores1.keys()) | set(fulltext_scores2.keys())

    # --- Build Response ---
    for idx in all_indices:
        item = metadata[idx]
        sparse1 = sparse_scores1[idx] if idx in sparse_sorted_indices1 else 0.0
        dense1 = dense_scores1[idx] if idx in dense_sorted_indices1 else 0.0
        fulltext1 = fulltext_scores1.get(idx, 0.0),
        sparse2 = sparse_scores2[idx] if idx in sparse_sorted_indices2 else 0.0
        dense2 = dense_scores2[idx] if idx in dense_sorted_indices2 else 0.0
        fulltext2 = fulltext_scores2.get(idx, 0.0),
        results.append(CombinedSearchResponse2(
            title = news["item.label"],
            text=item["text"],
            label=item["label"],
            score_sparse=sparse1,
            score_dense=dense1,
            score_sparse2=sparse2,
            score_dense2=dense2,
            umap_x=item["umap_x"],
            umap_y=item["umap_y"],
            cluster=item["cluster"]
        ))

    return results

@app.get("/tsvgsearch", response_model=List[CombinedSearchResponse])
def tsvgsearch(
    query: str = Query(..., min_length=3),
    top_k: int = 4947,
):
    results = []  # To store the combined results

    # --- Sparse: BM25 ---
    tokenized_query = query.lower().split()
    sparse_scores = bm25t2.get_scores(tokenized_query) / 100
    sparse_sorted_indices = np.argsort(sparse_scores)[::-1][:top_k]

    # --- Dense: Cosine Similarity ---
    query_embedding = model.encode([query])
    dense_scores = cosine_similarity(query_embedding, embeddings2)[0]
    dense_sorted_indices = np.argsort(dense_scores)[::-1][:top_k]

    # --- Full-text BM25 (Whoosh) ---
    fulltext_scores = []
    fulltext_sorted_indices = []
    # --- Combine Results from All Types ---
    all_sorted_indices = list(set(sparse_sorted_indices) | set(dense_sorted_indices))
    print(len(all_sorted_indices))
    for idx in all_sorted_indices:
        item = metadata2[idx]
        sparse = sparse_scores[idx] if idx in sparse_sorted_indices else 0.0
        dense = dense_scores[idx] if idx in dense_sorted_indices else 0.0
        results.append(CombinedSearchResponse(
            title = item['title'],
            text=item["text"],
            label=item["label"],
            score_sparse=sparse,
            score_dense=dense,
            umap_x=item["umap_x"],
            umap_y=item["umap_y"],
            cluster=item["cluster"]
        ))

    return results



@app.get("/tsvgsearch2", response_model=List[CombinedSearchResponse2])
def tsvgsearch2(
    query1: str = Query(..., min_length=3),
    query2: str = Query(..., min_length=3),
    top_k: int = 4947,
):
    results = []

    # --- Sparse: BM25 ---
    tokenized_query1 = query1.lower().split()
    sparse_scores1 = bm25t2.get_scores(tokenized_query1) / 100
    sparse_sorted_indices1 = np.argsort(sparse_scores1)[::-1][:top_k]

    tokenized_query2 = query2.lower().split()
    sparse_scores2 = bm25.get_scores(tokenized_query2) / 100
    sparse_sorted_indices2 = np.argsort(sparse_scores2)[::-1][:top_k]

    
    # --- Dense: Cosine Similarity ---
    query_embedding1 = model.encode([query1])
    dense_scores1 = cosine_similarity(query_embedding1, embeddings2)[0]
    dense_sorted_indices1 = np.argsort(dense_scores1)[::-1][:top_k]

    query_embedding2 = model.encode([query2])
    dense_scores2 = cosine_similarity(query_embedding2, embeddings2)[0]
    dense_sorted_indices2 = np.argsort(dense_scores2)[::-1][:top_k]

    # --- Full-text BM25 (Whoosh) ---
    max_docs = len(metadata2)
    top_k = min(top_k, max_docs)
    # --- Combine All Indices ---
    all_indices = set(sparse_sorted_indices1) | set(sparse_sorted_indices2) | \
                  set(dense_sorted_indices1) | set(dense_sorted_indices2) 
    valid_indices = [idx for idx in all_indices if idx < max_docs]

    print(f"Valid indices count: {len(valid_indices)}")
    # --- Build Response ---
    print(len(all_indices))
    for idx in valid_indices:
        item = metadata2[idx]
        sparse1 = sparse_scores1[idx] if idx in sparse_sorted_indices1 else 0.0
        dense1 = dense_scores1[idx] if idx in dense_sorted_indices1 else 0.0

        sparse2 = sparse_scores2[idx] if idx in sparse_sorted_indices2 else 0.0
        dense2 = dense_scores2[idx] if idx in dense_sorted_indices2 else 0.0

        results.append(CombinedSearchResponse2(
            title = item['title'],
            text=item["text"],
            label=item["label"],
            score_sparse=sparse1,
            score_dense=dense1,
            score_sparse2=sparse2,
            score_dense2=dense2,
            umap_x=item["umap_x"],
            umap_y=item["umap_y"],
            cluster=item["cluster"]
        ))

    return results



@app.get("/search3", response_model=list[SearchResponse2])
def search3(query: str = Query(..., min_length=3), top_k: int = 1):
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, embeddings)[0]

    # Sort results by similarity
    sorted_indices = np.argsort(scores)[::-1]
    results = []

    for idx in sorted_indices:
        item = metadata[idx]
        results.append(SearchResponse2(
            text=item["text"],
            label=item["label"],
            score2=float(scores[idx]),
            query = top_k,
            umap_x=item["umap_x"],
            umap_y=item["umap_y"],
            cluster=item["cluster"]
        ))

    return results

class SearchResponseFull(BaseModel):
    text: str
    label: int
    score: float
    score2: float
    query: int
    umap_x: float
    umap_y: float
    cluster: str
    
@app.get("/search-merged", response_model=list[SearchResponseFull])
def search_merged(
    query1: str = Query(..., min_length=3),
    query2: str = Query(..., min_length=3),
    type: Literal["sparse", "dense"] = Query("sparse"),
):
    # Compute both embeddings

    if type == "sparse":
        # Sparse: BM25
        tokenized_query1 = query1.lower().split()
        tokenized_query2 = query2.lower().split()
        scores1 = bm25.get_scores(tokenized_query1)
        scores2 = bm25.get_scores(tokenized_query2)
    else:
        # Dense: cosine similarity
        query_embedding1 = model.encode([query1])
        query_embedding2 = model.encode([query2])
        scores1 = cosine_similarity(query_embedding1, embeddings)[0]
        scores2 = cosine_similarity(query_embedding2, embeddings)[0]


    results = []
    for idx, item in enumerate(metadata):
        results.append(SearchResponseFull(
            text=item["text"],
            label=item["label"],
            score=float(scores1[idx]),
            score2=float(scores2[idx]),
            query=0,  # You can change this dynamically if needed
            umap_x=item["umap_x"],
            umap_y=item["umap_y"],
            cluster=item["cluster"]
        ))

    return results
class SummaryRequest(BaseModel):
    summary1: str
    summary2: str

# Function to compare summaries
def compare_summaries(summary1: str, summary2: str):
    """Compare two summaries and return two separate JSON objects."""

    prompt = f"""
    Compare the following two summaries and return two separate JSON objects.

    **Summary 1:** {summary1}
    **Summary 2:** {summary2}

    **JSON 1 Format:**
    {{
        "common_theme": "Common theme or idea",
        "common_fact": "Common fact from both summaries"
    }}

    **JSON 2 Format:**
    {{
        "summary_1_unique": "Unique fact from summary 1",
        "summary_2_unique": "Unique fact from summary 2"
    }}

    Provide only the JSON output.
    """

    try:
        chat = ChatOpenAI(
            openai_api_key=api_key,
            model='gpt-4o'
        )

        chat_messages = [
            SystemMessage(content="You are a helpful assistant."),  # Role definition
            HumanMessage(content=prompt),                 # Current query with context
        ]
        response = chat(chat_messages)

        # Extract JSON response from GPT output
        print(response,32)
        output_text = response.content.strip()
        # json_objects = output_text.split("\n\n")
        json_objects = re.findall(r'```json\n(.*?)\n```', output_text, re.DOTALL)
        print(output_text,281)
        # json_objects = output_text.strip().split("\n\n")  # Split into two separate JSONs

        json1 = json.loads(json_objects[0])
        json2 = json.loads(json_objects[1])

        return json1, json2

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# FastAPI route to compare summaries
@app.post("/compare-summaries/")
async def compare_summaries_api(request: SummaryRequest):
    json1, json2 = compare_summaries(request.summary1, request.summary2)
    return {"commonalities": json1, "differences": json2}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

class InputTexts(BaseModel):
    text_a: str
    text_b: str

class InputTexts(BaseModel):
    query_a: str
    query_b: str
    chunk_a: str
    chunk_b: str

# Load models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
models = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def convert_highlights(text):
    # Convert **word** into <span class="highlight">word</span>
    return re.sub(r'\*\*(.+?)\*\*', r"<span class='highlight'>\1</span>", text)
@app.post("/semantic-summary")
def get_semantic_summary(data: InputTexts):
    # Combine both chunks for summary
    combined = f"Chunk A: {data.chunk_a}\n\nChunk B: {data.chunk_b}"
    if len(combined.split()) > 1000:
        combined = " ".join(combined.split()[:1000])
    
    summary = summarizer(combined, max_length=180, min_length=60, do_sample=False)[0]["summary_text"]

    # Highlight relevant words from chunks
    highlighted_a = highlight_keywords(data.query_a, data.chunk_a)
    highlighted_b = highlight_keywords(data.query_b, data.chunk_b)

    # Compute embeddings
    emb_query_a = get_embedding(data.query_a)
    emb_query_b = get_embedding(data.query_b)
    emb_chunk_a = get_embedding(data.chunk_a)
    emb_chunk_b = get_embedding(data.chunk_b)

    # Compute similarities
    sim_a_to_a = cosine_sim(emb_query_a, emb_chunk_a)
    sim_a_to_b = cosine_sim(emb_query_a, emb_chunk_b)
    sim_b_to_a = cosine_sim(emb_query_b, emb_chunk_a)
    sim_b_to_b = cosine_sim(emb_query_b, emb_chunk_b)

    return {
        "summary": summary,
        "highlighted_chunk_a": (highlighted_a),
        "highlighted_chunk_b": (highlighted_b),
        "similarities": {
            "query_a_to_chunk_a": {
                "score": sim_a_to_a,
                "reason": explain_similarity(data.query_a, data.chunk_a, sim_a_to_a)
            },
            "query_a_to_chunk_b": {
                "score": sim_a_to_b,
                "reason": explain_similarity(data.query_a, data.chunk_b, sim_a_to_b)
            },
            "query_b_to_chunk_a": {
                "score": sim_b_to_a,
                "reason": explain_similarity(data.query_b, data.chunk_a, sim_b_to_a)
            },
            "query_b_to_chunk_b": {
                "score": sim_b_to_b,
                "reason": explain_similarity(data.query_b, data.chunk_b, sim_b_to_b)
            }
        }
    }


def highlight_keywords(query, chunk):
    # Tokenize query using spaCy
    query_tokens = [token.text.lower() for token in nlp(query) if not token.is_stop and not token.is_punct]
    
    # Extract meaningful keywords (non-stopwords, non-punctuation)
    highlighted_chunk = chunk
    for word in sorted(set(query_tokens), key=len, reverse=True):
        # Highlight the matching words in chunk
        highlighted_chunk = re.sub(
            rf'\b({re.escape(word)})\b',
            r"<span class='highlight'>\1</span>",
            highlighted_chunk,
            flags=re.IGNORECASE
        )
    return highlighted_chunk
def get_embedding(text):
    # Assume you're using a transformer model like sentence-transformers
    return model.encode(text, convert_to_tensor=True)

def cosine_sim(a, b):
    return float(cosine_similarity([a], [b])[0][0])

def explain_similarity(query, chunk, score):
    if score > 0.75:
        return f"The query is highly relevant to the chunk. Score: {score:.2f}"
    elif score > 0.5:
        return f"The query shares moderate relevance with the chunk. Score: {score:.2f}"
    else:
        return f"The query is weakly related to the chunk. Score: {score:.2f}"
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # generate_summary_embeddings()
