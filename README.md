# FAERS-NLP Preprocessing Code

**Author:** Sixue Xing  
**License:** MIT  
**Dataset:** [FAERS-NLP on Hugging Face](https://huggingface.co/datasets/sixuexing/FAERS-NLP)

---

## Overview

This repository contains code for processing the FDA Adverse Event Reporting System (FAERS) data into a clean, NLP-ready tabular dataset.

The code includes:

1. **Downloading raw FAERS data** from the FDA public site.  
2. **Parsing FAERS XMLs** into CSV files.  
3. **Encoding and preprocessing CSVs** for NLP tasks.  
4. **Two-Step Retrieval Search pipeline**, filtering by age, gender, time, etc; drug/disease/country vector search

The final processed dataset is available at: [FAERS-NLP on Hugging Face](https://huggingface.co/datasets/sixuexing/FAERS-NLP)

---


## Usage

Clone the repository and navigate into it:

```bash
git clone https://github.com/xingsixue123/FAERS-NLP.git
cd FAERS-NLP
````

Download raw FAERS data:

```bash
bash download.sh
```

Convert FAERS XML files to CSV:

```bash
python faers_xml2csv.py
```

Encode CSV data for retrieval:

> ⚠️ **Note:** `faers_csv2encode.py` may take **up to 5 hours** depending on your system and GPU availability.

```bash
python faers_csv2encode.py
```

Run a retrieval example:

```bash
python Retrieval_Example.py
```


---

## Example Query & Retrieval

**Query:**

- Serious: Yes  
- Sex: All  
- Time: 2012-01-01 to 2025-01-01  
- Age: 18 to 60  
- Active substances: octreotide acetate, tamoxifen citrate  
- Indication: Breast Cancer  
- Occur countries: United States, Canada  

**Top FAERS Report Retrieved:**

FAERS Report: 29-year-old Male (United States of America) patient treated with TAMOXIFEN CITRATE (TAMOXIFEN CITRATE. (Suspect)) prescribed for BREAST CANCER with the action taken: Drug Withdrawn, adverse event resolved/reliefed after withdrawal/reduction. Experienced seizure, gynaecomastia, vomiting, diarrhoea, neuralgia, neuropathy peripheral, hypertension, diverticulitis, mammogram abnormal, blood oestrogen abnormal, memory impairment, nausea, hot flush, colitis, emotional distress, galactorrhoea, malaise, migraine. Outcomes: resolved (gynaecomastia), ongoing/not resolved (seizure, vomiting, diarrhoea, neuralgia, neuropathy peripheral, hypertension, diverticulitis, mammogram abnormal, blood oestrogen abnormal, memory impairment, nausea, hot flush, colitis, emotional distress, galactorrhoea, malaise, migraine). Seriousness: Other (Yes). Treatment action: Drug Withdrawn. Report type: Spontaneous; received: 20170214.
