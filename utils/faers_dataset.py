# tools/faers_dataset.py
import os
import glob
import pandas as pd
from typing import List, Dict, Optional
from collections import defaultdict
from sentence_transformers import util
import sys
from transformers import AutoTokenizer, AutoModel
import torch
import json
from tqdm import tqdm
import ast
import re
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cwd_path = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, project_root)

FAERS_ABS = project_root    
FAERS_CSV_ABS = os.path.join(FAERS_ABS, "out")
FAERS_EMB_ABS = os.path.join(FAERS_ABS, "embs")


if not os.path.exists(FAERS_EMB_ABS):
    os.makedirs(FAERS_EMB_ABS)

from utils.similarity_search import fuzzy_search_difflib, fuzzy_search_rapidfuzz
from utils.encode_model import miniLM_encoder


model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.to(device)

class FaersDataset:
    def __init__(self, FAERS_CSV_ABS=FAERS_CSV_ABS, recursive=False, verbose=True):
        """
        Load all FAERS CSVs into memory at initialization.

        Args:
            FAERS_CSV_ABS: dir to search for CSVs.
            recursive: Whether to search recursively under data/faers/out.
            verbose: Print status info.
        """

        # Set FAERS paths
     
        self.FAERS_CSV_ABS = FAERS_CSV_ABS

        # Find all CSVs
        pattern = "**/*.csv" if recursive else "*.csv"
        self.csv_files = sorted(glob.glob(os.path.join(self.FAERS_CSV_ABS, pattern), recursive=recursive))

        if verbose:
            print(f"[FaersDataset] Found {len(self.csv_files)} CSV files in {self.FAERS_CSV_ABS}")

        # Load all CSVs
        self.df = self._load_all_csvs(verbose=verbose)
        print("All CSVs loaded.")
        
        # Construct or load encoded texts
        self.embeddings, self.indices, self.small_texts = self._load_small_texts(FAERS_EMB_ABS)  
        print("All embeddings loaded.")
        
    
    
    def _load_small_texts(self, path):
        """
        Try loading precomputed encodings. If missing, construct them.
        """
        pt_file = os.path.join(path, "faers_small_emb.pt")
        json_file = os.path.join(path, "faers_small_text.json")

        if os.path.exists(pt_file) and os.path.exists(json_file):
            print(f"[FaersDataset] Found existing encodings at {pt_file}, loading...")
            data = torch.load(pt_file)
            embeddings, indices = data["embeddings"], data["indices"]
            with open(json_file, "r", encoding="utf-8") as f:
                small_texts = [entry["text"] for entry in json.load(f)]
            return embeddings, indices, small_texts
        else:
            print("[FaersDataset] No encodings found, constructing from scratch...")
            return self.construct_small_texts(path)
        
    
    def construct_small_texts(self, path):
        """
        Construct clean text per row, encode with MiniLM, and save embeddings + JSON backup.
        """
        df = self.df
        small_texts, indices = [], []

        for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Constructing small texts"):
            # Build compact text
            text = construct_clean_text(row)
            small_texts.append(text)
            indices.append(idx)

        # Encode with MiniLM
        embeddings = []
        for text in tqdm(small_texts, desc="Encoding small texts"):
            emb = miniLM_encoder(text, model, tokenizer, device)
            embeddings.append(emb)

        embeddings = torch.stack(embeddings)  # [num_reports, hidden_dim]

        # Save torch file
        torch.save({
            "embeddings": embeddings,
            "indices": indices
        }, os.path.join(path, "faers_small_emb.pt"))

        # Save json backup
        backup = [{"index": idx, "text": text} for idx, text in zip(indices, small_texts)]
        with open(os.path.join(path, "faers_small_text.json"), "w", encoding="utf-8") as f:
            json.dump(backup, f, ensure_ascii=False, indent=2)

        print(f"[FaersDataset] Encoded {len(small_texts)} rows, saved to {path}")
        return embeddings, indices, small_texts


    def _read_csv_safe(self, path):
        """Read one CSV robustly with dtype=str."""
        try:
            return pd.read_csv(path, dtype=str, encoding="utf-8", on_bad_lines="warn", engine="pyarrow")
        except Exception as e:
            print(f"[ERROR] Could not read {path}: {e}")
            return None

    def _load_all_csvs(self, verbose=True):
        dfs = []
        
        
        # ctr = 0 ## for debugging
        
        for f in self.csv_files:
            
            # ctr += 1 ## for debugging
            # if ctr > 2: ## for debugging
            #     break
            
            
            df = self._read_csv_safe(f)
            if df is None:
                continue
            df["__source_file"] = os.path.basename(f)
            dfs.append(df)

            if verbose:
                print(f"  loaded {os.path.basename(f)}: {df.shape}")

        if not dfs:
            print("[FaersDataset] No data loaded.")
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True, sort=False)
        
        # Compute absolute age 
        ABS_AGE_COEFFICIENT = {
            "Decade": 10,
            "Year": 1,
            "Month": 1/12,
            "Week": 1/52,
            "Day": 1/365,
            "Hour": 1/8760,
        }
        coef_map = combined["patientonsetageunit"].map(ABS_AGE_COEFFICIENT).fillna(0)
        age_val = pd.to_numeric(combined["patientonsetage"], errors="coerce")
        combined["absage"] = coef_map * age_val
        
        if verbose:
            print(f"[FaersDataset] Combined shape: {combined.shape}")
        return combined
    

    def get_dataframe(self):
        """Return the combined DataFrame in memory."""
        return self.df

    def head(self, n=5):
        """Shortcut for self.df.head()."""
        return self.df.head(n)
    
    def columns(self):
        """Shortcut for self.df.columns."""
        return self.df.columns  
    
    def top_k_relevant_data_retrieve(self, query: Dict[str, str], k: int = 5, fetch_method: str = "word_similarity"):
        """
        Retrieve top-k most relevant FAERS rows based on query filters and fuzzy scores.

        Args:
            query: dict with possible keys:
                "serious": "Yes"/"No"/"All"
                "sex": "Male"/"Female"/"All"
                "time": {"min": start_time, "max": end_time} (int e.g., 20160101)
                "age": {"min": min_age, "max": max_age} (int in years)
                "active_substance": drug_name string
                "indication": disease_name string
                "occur_country": country_name string
            k: number of top rows to return
            weights: tuple of 3 floats (drug, disease, country)

        Returns:
            top_k_df: pd.DataFrame of top-k rows sorted by combined score
        """
        
        df = self.df
        if df.empty:
            print("[top_k_relevant_data_retrieve] DataFrame is empty")
            return pd.DataFrame()

        # Initial index: all rows
        idx = df.index

        # 1. Filter by time
        time_query = query.get("time", {})
        start_time = time_query.get("min")
        end_time = time_query.get("max")
        idx = filter_by_time_idx(df, start=start_time, end=end_time, subdf_idx=idx)

        # 2. Filter by seriousness
        serious = query.get("serious", "All")
        idx = filter_by_serious_idx(df, serious=serious, subdf_idx=idx)

        # 3. Filter by sex
        sex = query.get("sex", "All")
        idx = filter_by_sex_idx(df, sex=sex, subdf_idx=idx)

        # 4. Filter by age
        age_query = query.get("age", {})
        min_age = age_query.get("min")
        max_age = age_query.get("max")
        idx = filter_by_age(df, min_age=min_age, max_age=max_age, subdf_idx=idx)

        # 5. Compute fuzzy scores for active substance, indication, occur country
        
        #  Word similarity method  ---
        if fetch_method == "word_similarity":
            w_drug, w_disease, w_country = (1/3, 1/3, 1/3)

            score_dict_drug = {}
            score_dict_disease = {}
            score_dict_country = {}

            if "active_substance" in query:
                score_dict_drug = filter_by_active_substance(df, query["active_substance"], subdf_idx=idx)
            if "indication" in query:
                score_dict_disease = filter_by_indication(df, query["indication"], subdf_idx=idx)
            if "occur_country" in query:
                score_dict_country = filter_by_occurcountry(df, query["occur_country"], subdf_idx=idx)
                

            combined_scores = defaultdict(float)
            dicts_to_intersect = [d for d in [score_dict_drug, score_dict_disease, score_dict_country] if d]
            if dicts_to_intersect:
                common_idx = set(dicts_to_intersect[0].keys())
                for d in dicts_to_intersect[1:]:
                    common_idx &= set(d.keys())
            else:
                common_idx = set(idx)

            for i in common_idx:
                combined_scores[i] = (
                    score_dict_drug.get(i, 0) * w_drug +
                    score_dict_disease.get(i, 0) * w_disease +
                    score_dict_country.get(i, 0) * w_country
                )

            sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            top_k_indices = [i for i, _ in sorted_combined[:k]]
            
        elif fetch_method == "encode_similarity":
            # Build clean text query
            query_text = f"{query.get('occur_country','')}; {query.get('active_substance','')}; {query.get('indication','')}"

            all_embeddings = self.embeddings
            all_indices = self.indices

            # Encode query
            query_emb = miniLM_encoder(query_text, model, tokenizer, device).unsqueeze(0)

            # Restrict to filtered idx if provided
            # Build index → position dict once
            index2pos = {idx_val: pos for pos, idx_val in enumerate(all_indices)}

            if idx is not None and len(idx) > 0:
                subset_positions = [index2pos[i] for i in idx if i in index2pos]  # O(1) lookup
                subset_embeddings = all_embeddings[subset_positions]
                subset_indices = [all_indices[i] for i in subset_positions]
            else:
                subset_embeddings = all_embeddings
                subset_indices = all_indices

            # Cosine similarity
            scores = util.cos_sim(query_emb, subset_embeddings)[0]
            topk_scores, topk_pos = torch.topk(scores, k=min(k, len(scores)))    
            top_k_indices = [subset_indices[i] for i in topk_pos]
            combined_scores = {subset_indices[i]: float(topk_scores[j]) 
                       for j, i in enumerate(topk_pos)}
            
            
        else:
            raise ValueError(f"Invalid fetch_method: {fetch_method}")

        # print(combined_scores)
        #print(top_k_indices)
        
        # Return the corresponding rows
        top_k_df = df.loc[top_k_indices].copy()
        # Optional: add combined_score column
        top_k_df["combined_score"] = [combined_scores[i] for i in top_k_indices]
        
        return top_k_df
    



    def row_to_natural_language(self, df_idx):
        """
        Convert a FAERS dataframe row into a compact, scientific natural language summary
        suitable for RAG retrieval, using REACTIONOUTCOME_LABELS for outcomes.
        """
        
        row = self.df.iloc[df_idx]

        # Patient info
        age = int(row["absage"]) if not pd.isna(row["absage"]) else "Unknown age"
        sex = "Male" if row["patientsex"] == "Male" else "Female" if row["patientsex"] == "Female" else "Unknown sex"
        country = row["occurcountry"]
        
        # Report info
        report_type = row["reporttype"]
        serious_specific = row["serious_specific"] or "No"
        received_date = row["receivedate"]
        
        # Reactions
        reactions_raw = row["reactions"]  
        reaction_outcomes = []
        for r in reactions_raw.split(";"):
            if "(" in r:
                name, code = r.split("(")
                code = code.replace(")", "").strip()
                reaction_outcomes.append((name.strip().lower(), code))
                       
        # Group reactions by outcome
        grouped = {"resolved": [], "recovering/resolving": [], "ongoing/not resolved": [], "fatal": [], "unknown": []}
        for name, outcome in reaction_outcomes:
            if outcome == "Recovered / Resolved" or outcome == "Recovered / Resolved with Sequelae":
                grouped["resolved"].append(name)
            elif outcome == "Recovering / Resolving":
                grouped["recovering/resolving"].append(name)
            elif outcome == "Not Recovered / Not Resolved / Ongoing":
                grouped["ongoing/not resolved"].append(name)
            elif outcome == "Fatal":
                grouped["fatal"].append(name)
            else:
                grouped["unknown"].append(name)
        
        # Drugs
        drugs = row["drugs_specific"]
        single_drug_texts = drugs.split(";")
        drugs_info = {}
        
        for single_drug_text in single_drug_texts:
            # Regex to find 'Active:' and everything after as key-value pairs
            pattern = r'(\bActive\b|\bDosage\b|\bIndication\b|\bAction\b|\bAdditional\b):\s*([^,]+)'
            matches = re.findall(pattern, single_drug_text)

            # Convert matches to a dictionary
            drug_parts = {k: v.strip() for k, v in matches}

            # Drug name is the text **right before 'Active:'**
            drug_name_match = re.search(r'(.+?),\s*Active:', single_drug_text)
            drug_name = drug_name_match.group(1).strip() if drug_name_match else ""

            active_name = drug_parts.get("Active", "")
            dosage = drug_parts.get("Dosage", "")
            indication = drug_parts.get("Indication", "")
            action = drug_parts.get("Action", "")
            additional = drug_parts.get("Additional", "")
            
            # Add to drugs_info
            if drug_name not in drugs_info:
                drugs_info[drug_name] = {
                    "active": active_name,
                    "indication": [indication],
                    "action": [action],
                    "dosage": [dosage],
                    "additional": [additional]
                }
            else:
                if indication != "" and indication not in drugs_info[drug_name]["indication"]:
                    drugs_info[drug_name]["indication"].append(indication)

                if action != "" and action not in drugs_info[drug_name]["action"]:
                    drugs_info[drug_name]["action"].append(action)
                    
                if dosage != "" and dosage not in drugs_info[drug_name]["dosage"]:
                    drugs_info[drug_name]["dosage"].append(dosage)

                if additional != "" and additional not in drugs_info[drug_name]["additional"]:
                    drugs_info[drug_name]["additional"].append(additional)

        # clear redundancy
        for drug_name in drugs_info:
            for key in drugs_info[drug_name]:
                if key == "active":
                    continue
                if key == "dosage":
                    # remove "UNK"
                    drugs_info[drug_name][key] = [value for value in drugs_info[drug_name][key] if value != "UNK"]
                    
                drugs_info[drug_name][key] = [value for value in drugs_info[drug_name][key] if value != ""]
                # remove redundancy
                drugs_info[drug_name][key] = list(set(drugs_info[drug_name][key]))
        
        def construct_drug_sentence_natural(drugs_info):
            
            sentences = []
            ctr = 1
            for drug_name, info in drugs_info.items():
                sentence_parts = []
                
                # if not only one drug
                if drugs_info.keys().__len__() > 1:
                    sentence_parts.append(f"{ctr}.")

                # Drug and active ingredient
                if info.get("active"):
                    sentence_parts.append(f"{info['active']} ({drug_name})")
                else:
                    sentence_parts.append(f"{drug_name}")

                # Indication(s)
                if info.get("indication"):
                    indications = ", ".join(info["indication"])
                    sentence_parts.append(f"prescribed for {indications}")

                # Dosage(s)
                if info.get("dosage"):
                    dosages = ", ".join(info["dosage"])
                    sentence_parts.append(f"at a dosage of {dosages}")

                # Action(s)
                if info.get("action"):
                    actions = ", ".join(info["action"])
                    sentence_parts.append(f"with the action taken: {actions}")

                # Additional info
                if info.get("additional"):
                    # Include only if Yes or No
                    valid_additionals = [x for x in info["additional"] if x in ["Yes", "No"]]
                    if valid_additionals == ["Yes"]:
                        sentence_parts.append(", adverse event resolved/reliefed after withdrawal/reduction")
                    elif valid_additionals == ["No"]:
                        sentence_parts.append(", adverse event NOT resolved after withdrawal/reduction")


                # Combine into a natural sentence
                sentence = " ".join(sentence_parts) + "."
                sentences.append(sentence)
            
                ctr += 1

            # Combine all drug sentences for the patient
            return " ".join(sentences)

        drug_sentence = construct_drug_sentence_natural(drugs_info)

            
        # Build reaction summary
        reaction_summary = []
        if grouped["resolved"]:
            reaction_summary.append(f"resolved ({', '.join(grouped['resolved'])})")
        if grouped["recovering/resolving"]:
            reaction_summary.append(f"recovering/resolving ({', '.join(grouped['recovering/resolving'])})")
        if grouped["ongoing/not resolved"]:
            reaction_summary.append(f"ongoing/not resolved ({', '.join(grouped['ongoing/not resolved'])})")
        if grouped["fatal"]:
            reaction_summary.append(f"fatal ({', '.join(grouped['fatal'])})")
        if grouped["unknown"]:
            reaction_summary.append(f"unknown ({', '.join(grouped['unknown'])})")
        
        # Build final summary
        summary = (
            f"FAERS Report: {age}-year-old {sex} ({country}) patient treated with "
            f"{drug_sentence} "
            f"Experienced {', '.join([name for name, _ in reaction_outcomes])}. "
            f"Outcomes: {', '.join(reaction_summary)}. "
            f"Seriousness: {serious_specific}. "
            f"Treatment action: {action}. "
            f"Report type: {report_type}; received: {received_date}."
        )
        
        # Clean up multiple spaces and commas
        summary = summary.replace(" ,", ",").replace(" .", ".")
        
        return summary
        

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
        if not x:
            return []
        if x.startswith("[") and x.endswith("]"):
            try:
                return ast.literal_eval(x)
            except Exception:
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
    country = str(faers_row.get("occurcountry", "") or "")
    drug_names = ", ".join(ensure_list(faers_row.get("activesubstancenames", "")))
    disease_names = ", ".join(ensure_list(faers_row.get("drugindications", "")))
    return f"{country}; {drug_names}; {disease_names}"

# Two-step retrieve filters
def filter_by_id_idx(df, safetyreportid, subdf_idx=None):
    """
    Filter by safetyreportid
    Return: filtered index -> List[int]
    """
    base_idx = subdf_idx if subdf_idx is not None else df.index
    mask = df.loc[base_idx, "safetyreportid"] == safetyreportid
    return base_idx[mask]


def filter_by_serious_idx(df, serious="All", subdf_idx=None):
    """
    serious: "Yes" / "No" / "All"
    Return: filtered index -> List[int]
    """
    base_idx = subdf_idx if subdf_idx is not None else df.index

    if serious == "All":
        return base_idx
    elif serious in ("Yes", "No"):
        mask = df.loc[base_idx, "serious"] == serious
        return base_idx[mask]
    else:
        raise ValueError(f"Invalid serious: {serious}")


def filter_by_sex_idx(df, sex="All", subdf_idx=None):
    """
    sex: "Male" / "Female" / "All"
    Return: filtered index -> List[int]
    """
    base_idx = subdf_idx if subdf_idx is not None else df.index

    if sex == "All":
        return base_idx
    elif sex in ("Male", "Female"):
        mask = df.loc[base_idx, "patientsex"] == sex
        return base_idx[mask]
    else:
        raise ValueError(f"Invalid sex: {sex}")



def filter_by_time_idx(df, start=None, end=None, subdf_idx=None):
    """
    Return filtered index based on receive date.

    Parameters
    ----------
    df : DataFrame
        Original dataframe (not modified).
    start : int or None
        Start date, e.g. 20140326.
    end : int or None
        End date, e.g. 20181010.
    subdf_idx : Index or None
        Previously filtered index (for chaining).

    Returns
    -------
    List[int] : filtered indices
    """
    base_idx = subdf_idx if subdf_idx is not None else df.index
      
    # Convert receivedate to numeric, invalid -> NaN
    rec_dates = pd.to_numeric(df.loc[base_idx, "receivedate"], errors="coerce")
    
    # mask: drop NaN and apply start/end filters
    mask = rec_dates.notna()
    if start is not None:
        mask &= rec_dates >= start
    if end is not None:
        mask &= rec_dates <= end

    # Return filtered indices
    return base_idx[mask]



def filter_by_age(df, min_age=None, max_age=None, subdf_idx=None):
    """
    Filter by absolute age (in years), derived from patientonsetage and patientonsetageunit.
    min_age, max_age are in years.
    Return: filtered index -> List[int]
    """
    base_idx = subdf_idx if subdf_idx is not None else df.index
    abs_age = df.loc[base_idx, "absage"]

    mask = pd.Series(True, index=base_idx)
    if min_age is not None:
        mask &= abs_age >= min_age
    if max_age is not None:
        mask &= abs_age <= max_age

    return base_idx[mask]



def filter_by_active_substance(df, substance: str, subdf_idx=None):
    """
    Fuzzy filter by active substance.

    Steps:
    1. Collect all unique drug names from the selected indices (base_idx),
       making sure each row is treated as a list.
    2. Compute a fuzzy match score for each unique drug against the query substance.
    3. For each row, take the maximum score among its drugs as the row's score.
    
    Returns:
        idx_score_dict: Dict[int, float] mapping row index to highest fuzzy match score (0-1). -> Dict[int, float]
    """
    # Use provided subdf_idx or default to all indices
    base_idx = subdf_idx if subdf_idx is not None else df.index

    # 1. Collect all candidate drugs
    all_drugs = set()
    series = df.loc[base_idx, "activesubstancenames"].dropna()

    # convert to list
    series = series.apply(lambda x: [x] if isinstance(x, str) else x)

    # flatten
    all_drugs = set(series.explode().dropna().unique())

    # 2. Compute fuzzy match score for each unique drug
    drug_score_dict: Dict[str, float] = {}
    for drug in all_drugs:
        scores = fuzzy_search_rapidfuzz(substance, [drug])
        # fuzzy_search_rapidfuzz returns {score: [candidates]}, take the max score
        drug_score_dict[drug] = max(scores.keys()) if scores else 0.0

    # 3. Assign each row the maximum score among its drugs
    idx_score_dict: Dict[int, float] = {}
    for idx in base_idx:
        row_drugs = df.at[idx, "activesubstancenames"]

        # Ensure list
        if isinstance(row_drugs, str):
            row_drugs = [row_drugs]
        elif not row_drugs:
            continue

        scores = [drug_score_dict.get(d, 0) for d in row_drugs]
        if scores:
            idx_score_dict[idx] = max(scores)

    return idx_score_dict



def filter_by_indication(df, indication: str, subdf_idx=None):
    """
    Fuzzy filter by drug indication (disease).

    Steps:
    1. Collect all unique indications from the selected indices (base_idx),
       ensuring each row is treated as a list.
    2. Compute a fuzzy match score for each unique indication against the query.
    3. For each row, assign the maximum score among its indications.

    Returns:
        idx_score_dict: Dict[int, float] mapping row index to highest fuzzy match score (0-1). -> Dict[int, float]
    """
    base_idx = subdf_idx if subdf_idx is not None else df.index

    # 1. Collect all unique indications
    all_indications = set()
    series = df.loc[base_idx, "drugindications"].dropna()

    # makesure elements are list
    series = series.apply(lambda x: [x] if isinstance(x, str) else x)

    # flatten
    all_indications = set(series.explode().dropna().unique())


    # 2. Compute fuzzy match score for each unique indication
    indication_score_dict: Dict[str, float] = {}
    for ind in all_indications:
        scores = fuzzy_search_rapidfuzz(indication, [ind])
        indication_score_dict[ind] = max(scores.keys()) if scores else 0.0

    # 3. Assign each row the maximum score among its indications
    idx_score_dict: Dict[int, float] = {}
    for idx in base_idx:
        row_indications = df.at[idx, "drugindications"]

        if isinstance(row_indications, str):
            row_indications = [row_indications]
        elif not row_indications:
            continue

        scores = [indication_score_dict.get(ind, 0) for ind in row_indications]
        if scores:
            idx_score_dict[idx] = max(scores)

    return idx_score_dict



def filter_by_occurcountry(df, country: str, subdf_idx=None):
    """
    Fuzzy filter by occur country.
    
    Parameters:
        df : DataFrame
        country : str
            Query country name
        subdf_idx : optional index subset for chaining

    Returns:
        idx_score_dict : Dict[int, float] mapping row index to fuzzy match score (0-1) -> Dict[int, float]
    """
    base_idx = subdf_idx if subdf_idx is not None else df.index
    idx_score_dict: Dict[int, float] = {}

    for idx in base_idx:
        row_country = df.at[idx, "occurcountry"]
        if not row_country:
            continue

        # fuzzy match single string
        scores = fuzzy_search_rapidfuzz(country, [row_country])
        idx_score_dict[idx] = max(scores.keys()) if scores else 0.0

    return idx_score_dict








    
    


            






