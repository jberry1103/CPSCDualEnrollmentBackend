'''
This file handles the general search bar (for both admin and student views)
'''


from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import re

def _normalize_rows(df, cols):
    for c in cols:
        df[c] = df[c].astype(str).fillna("").replace("nan", "", regex=False)
    return df

# Choose columns that cover "anything the user might type"
GENERAL_SEARCH_COLS = [
    "College Course Name",
    "College Course",                # catalog number like ENGL 101
    "College Course Description",
    "HS Course Name",
    "HS Course Description",
    "College Course CIP Code",
    "Career Cluster",
    "High School",
    "College",
]

# Lightweight heuristics for column weights based on query shape
CIP_RE   = re.compile(r"^\s*\d{2}\.\d{4}\s*$")     # e.g., 11.0901
COURSE_RE = re.compile(r"^\s*[A-Za-z]{2,4}\s*[- ]?\d{3}[A-Z]?\s*$")  # e.g., CSCI 141, ENGL101, MATH-151

def _column_base_weights():
    # Baseline weights — text-rich columns get a bit more weight
    return {
        "College Course Name": 1.2,
        "College Course": 1.1,
        "College Course Description": 1.3,
        "HS Course Name": 1.1,
        "HS Course Description": 1.2,
        "College Course CIP Code": 1.0,
        "HS Course CIP Code": 1.0,
        "Career Cluster": 0.9,
        "High School": 0.9,
        "College": 0.9,
    }

def _query_boosts(query):
    # This helper function improves accuracy for search terms that are unique, not real words (ENGL 101, 56.9038, etc)
    boosts = {}
    if CIP_RE.match(query or ""):
        boosts["College Course CIP Code"] = 1.4 # the 11.0201 numeric pattern will almost always be a college cip code
    if COURSE_RE.match(query or ""):
        boosts["College Course"] = 1.3
        boosts["College Course Name"] = 1.15
    return boosts

def build_general_indices(df, model, cols=GENERAL_SEARCH_COLS):
    df = _normalize_rows(df.copy(), cols)
    # Precompute embeddings and FAISS indices for each column
    indices = {}
    d = None
    for col in cols:
        texts = df[col].tolist()
        embs = model.encode(texts, convert_to_numpy=True).astype("float32")
        # Normalize to use cosine similarity via inner product
        norms = (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        embs = embs / norms
        if d is None:
            d = embs.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embs)
        indices[col] = {"index": index, "embs": embs}
    return indices

'''
This function runs the backend search algorithm,
    query: the input search term
    df: the courses dataset (this can be small when filters are applied)
    model: sentence transformer model, as defined in server.py (this should not change)
    indices: indices used for FAISS (ai algorithm)
    top_k_per_col: amount of rows used for sample (training) dataset. Too many rows can cause hallucination. Default 25.
    min_results: the user will always see this many courses. Default 10
    rel_threshold: courses with over a certain percent match will be presented to the user. Default 60%
    max_results: the user will not see more than this many courses. Default none
'''
def general_search(query, df, model, indices, top_k_per_col=25, min_results=10, rel_threshold=0.60, max_results=None):
    
    if not query or not query.strip():
        return df.head(0)  # empty result for empty query

    base_w = _column_base_weights()
    boosts = _query_boosts(query)

    # Combine weights
    col_weights = {c: base_w.get(c, 1.0) * boosts.get(c, 1.0) for c in indices.keys()}

    # Encode + normalize query
    q = model.encode([query], convert_to_numpy=True).astype("float32")
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)

    # Score accumulator per row
    row_scores = {}

    # Substring / exact-match lightweight boost (case-insensitive)
    qlow = query.strip().lower()
    substring_bonus_cols = ["College Course Name", "College Course", "HS Course Name", "College Course CIP Code"]
    substring_hits = set()
    for col in substring_bonus_cols:
        vals = df[col].astype(str).str.lower()
        exact_mask = vals == qlow
        contains_mask = vals.str.contains(re.escape(qlow), regex=True)
        # Exact matches get a solid bump; substrings a smaller one
        exact_rows = df.index[exact_mask].to_numpy().astype("int64")
        for r in exact_rows:
            k = int(r)
            row_scores[k] = row_scores.get(k, 0.0) + 2.0 * col_weights.get(col, 1.0)
            substring_hits.add(r)
        contains_rows = df.index[contains_mask].tolist()
        for r in contains_rows:
            k = int(r)
            row_scores[k] = row_scores.get(k, 0.0) + 0.5 * col_weights.get(col, 1.0)

    # FAISS search per column + weighted aggregation
    for col, pack in indices.items():
        index = pack["index"]
        D, I = index.search(q, top_k_per_col)  # inner-product == cosine sim (since normalized)
        sims = D[0]  # higher is better ([-1, 1])
        rows = I[0]
        w = col_weights.get(col, 1.0)
        for sim, row_id in zip(sims, rows):
            # Small extra bump if we also substring-hit the same row
            k = int(row_id)
            bonus = 0.25 if k in substring_hits else 0.0
            row_scores[k] = row_scores.get(k, 0.0) + w * (float(sim) + bonus)

    # Rank rows
    if not row_scores:
        return df.head(0)
    
    ranked = sorted(row_scores.items(), key=lambda x: x[1], reverse=True)
    scores = np.array([s for _, s in ranked], dtype=np.float32)
    max_s = float(scores.max())

    if max_s <= 0.0:
        # if all scores <= 0, just take the top min_results
        selected = [r for r, _ in ranked[:min_results]]
    else:
        cutoff = rel_threshold * max_s 
        selected = [r for r, s in ranked if s >= cutoff]
        if len(selected) < min_results:
            selected = [r for r, _ in ranked[:min_results]]

    if max_results is not None:
        selected = selected[:max_results]

    top_rows = selected
    print("Top Rows: ", top_rows)
    print("Top Rows datatype: ", type(top_rows))
    #print("Courses_df: ", courses_df)
    return df.iloc[top_rows]

