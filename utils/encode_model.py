from transformers import AutoTokenizer, AutoModel
import torch
torch.manual_seed(0)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model_name = "dmis-lab/biobert-base-cased-v1.2"
#model_name = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
# model_name = "sentence-transformers/all-MiniLM-L6-v2"


# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)
# model.to(device)


def biobert_encoder(sentence, model, tokenizer, device):
    
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (1, 768)

    result = cls_embedding.squeeze(0)
    result = result.detach().cpu()

    return result


def miniLM_encoder(sentence, model, tokenizer, device):
    """
    Encode a sentence into a dense embedding using MiniLM (384-dim)
    """
    # Tokenize
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=32)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean pooling over token embeddings
    # outputs.last_hidden_state shape: [1, seq_len, hidden_dim]
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    sentence_embedding = sum_embeddings / sum_mask  # shape: [1, 384]

    return sentence_embedding.squeeze(0).detach().cpu()