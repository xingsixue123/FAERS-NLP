from typing import List, Dict, Optional
from rapidfuzz import fuzz, process
from difflib import SequenceMatcher



def fuzzy_search_rapidfuzz(query: str, candidates: List[str], k: Optional[int] = None) -> Dict[float, str]:
    """
    Fuzzy match using rapidfuzz (faster).
    Returns {score: [candidates]} (group by score).
    """
    if not candidates:
        return {}

    # extract top-k (if k=None, extract all)
    results = process.extract(query, candidates, scorer=fuzz.ratio, limit=k)

    out: Dict[float, List[str]] = {}
    for cand, score, _ in results:
        s = round(score / 100, 4)  # normalize 0–1 like difflib
        out.setdefault(s, []).append(cand)
    return out




def fuzzy_search_difflib(query: str, candidates: List[str], k: Optional[int] = None) -> Dict[float, str]:
    """
    Fuzzy match using difflib.
    Returns {score: [candidates]} (group by score).
    """
    q = query.strip().lower()
    scores = []

    for cand in candidates:
        score = SequenceMatcher(None, q, cand.lower()).ratio()
        if score > 0:
            scores.append((score, cand))

    if not scores:
        return {}

    # sort by score descending
    scores.sort(key=lambda x: x[0], reverse=True)

    # handle k logic
    if k is not None and len(scores) > k:
        scores = scores[:k]

    out: Dict[float, List[str]] = {}
    for s, cand in scores:
        s = round(s, 4)
        out.setdefault(s, []).append(cand)
    return out



