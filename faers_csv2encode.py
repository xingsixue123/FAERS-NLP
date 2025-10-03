import os
import sys
import pandas as pd
from tqdm import tqdm
import ast
import torch
import json
from sentence_transformers import util

# --- path setup ---
cwd_path = os.path.dirname(os.path.realpath(__file__))
FAERS_ABS = cwd_path
FAERS_CSV_ABS = os.path.join(FAERS_ABS, "out")
FAERS_EMB_ABS = os.path.join(FAERS_ABS, "embs")
print(FAERS_CSV_ABS)
print(FAERS_EMB_ABS)

if not os.path.exists(FAERS_EMB_ABS):
    os.makedirs(FAERS_EMB_ABS)

from utils.encode_model import miniLM_encoder
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import AutoTokenizer, AutoModel
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.to(device)


def ensure_list(x):
    """
    Convert x into a proper Python list.
    If x is already a list, return as is.
    If x is a string that looks like a list, parse it.
    If x is a string but not a list, wrap in a list.
    """
    if isinstance(x, list):
        return x
    elif isinstance(x, str):
        x = x.strip()
        if x.startswith("[") and x.endswith("]"):
            try:
                return ast.literal_eval(x)
            except:
                return [x]  # fallback if parsing fails
        else:
            return [x]
    else:
        return []


def construct_clean_text(faers_row):
    """
    row: dict or pandas Series with keys: 'active_substance', 'indication', 'country'
    Returns a concise, clean representation for DPR
    """
    country = faers_row["occurcountry"]
    drug_names = ", ".join(ensure_list(faers_row["activesubstancenames"]))
    disease_names = ", ".join(ensure_list(faers_row["drugindications"]))
    return f"{country}; {drug_names}; {disease_names}"

def test_encoder():
    
    mystr = "The capital of France is Paris."
    encode = miniLM_encoder(mystr, model, tokenizer, device)
    print(encode)
    

def encode():
    df = pd.read_csv(f"{FAERS_ABS}/out/1_ADR18Q2_format.csv")
    
    small_texts = []
    indices = []
   
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Cleaning small texts"):
        text = construct_clean_text(row)
        small_texts.append(text)
        indices.append(idx)  # save original row index

    # Encode all small texts with biobert_encoder
    embeddings = []
    for text in tqdm(small_texts, desc="Encoding small texts"):
        emb = miniLM_encoder(text, model, tokenizer, device)
        embeddings.append(emb)

    embeddings = torch.stack(embeddings)  # shape: [num_reports, hidden_dim]
    
    # Save embeddings + indices
    torch.save({
        "embeddings": embeddings,
        "indices": indices
    }, f"{FAERS_EMB_ABS}/small_emb_test.pt")
    
    # Save small_texts + indices as JSON for backup / accuracy check
    small_text_backup = [{"index": idx, "text": text} for idx, text in zip(indices, small_texts)]
    with open(f"{FAERS_EMB_ABS}/small_text_backup.json", "w", encoding="utf-8") as f:
        json.dump(small_text_backup, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(small_texts)} small texts to JSON backup.")
    

def retrieve_topk(query, top_k=5, subdf_idx=None):
    """
    query: str
    top_k: int
    subdf_idx: list of indices (subset of all_indices)
    """
    
    print(f"Query: {query}")
    
    data = torch.load(f"{FAERS_EMB_ABS}/small_emb_test.pt")
    all_embeddings = data["embeddings"]
    all_indices = data["indices"]
    
    # Encode query
    query_emb = miniLM_encoder(query, model, tokenizer, device).unsqueeze(0)  # [1, hidden_dim]
    
    # Select subset embeddings
    if subdf_idx is not None:
        # Map subdf_idx to positions in all_indices
        subset_positions = [all_indices.index(i) for i in subdf_idx]
        subset_embeddings = all_embeddings[subset_positions]
        subset_indices = subdf_idx
    else:
        subset_embeddings = all_embeddings
        subset_indices = all_indices
    
    # Compute cosine similarity
    scores = util.cos_sim(query_emb, subset_embeddings)[0]  # [subset_size]
    
    # Get top-k
    topk_scores, topk_pos = torch.topk(scores, k=min(top_k, len(scores)))
    
    # Map back to original row indices
    topk_indices = [subset_indices[i] for i in topk_pos]
    
    return topk_indices, topk_scores


if __name__ == "__main__":

    encode()  
    # query = "United States; fluticasone furoate; cough"
    # topk_indices, topk_scores = retrieve_topk(query, top_k=5)
    # print(topk_indices)
    # print(topk_scores)

        


    # df = pd.read_csv(f"{FAERS_ABS}/out/1_ADR18Q2_format.csv")
    # print(df.head())


