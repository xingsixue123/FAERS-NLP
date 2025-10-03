from utils.faers_dataset import FaersDataset


def print_row():
    
    faers = FaersDataset()
    df = faers.get_dataframe()
    print(df.columns)
    
    for i in range(200):
    
        print(df.iloc[i]["occurcountry"])
        print(df.iloc[i]["reporttype"])
        print(df.iloc[i]["serious"])
        print(df.iloc[i]["serious_specific"])
        print(df.iloc[i]["receivedate"])
        print(df.iloc[i]["absage"])
        print(df.iloc[i]["patientsex"])
        print(df.iloc[i]["reactions"])
        print()
        print(df.iloc[i]["drugs_specific"])
        print()
        
        sumary = faers.row_to_natural_language(i)
        
        print(sumary)
        print()
        breakpoint()

def retrieve_test():
    
    faers = FaersDataset()
    queries =[
    {
        "serious": "Yes",
        "sex": "All",
        "time": {"min": 20120101, "max": 20250101},
        "age": {"min": 18, "max": 60},
        "active_substance": 'mirabegron',
        "indication": "unknown",
        "occur_country": "United States"
    },
    {
        "serious": "Yes",
        "sex": "All",
        "time": {"min": 20120101, "max": 20250101},
        "age": {"min": 18, "max": 60},
        "active_substance": 'Premarin and Provera',
        "indication": "Systemic Lupus Erythematosus",
        "occur_country": "United States"
    },
    {
        "serious": "Yes",
        "sex": "All",
        "time": {"min": 20120101, "max": 20250101},
        "age": {"min": 18, "max": 60},
        "active_substance": 'octreotide acetate, tamoxifen citrate',
        "indication": "Breast Cancer",
        "occur_country": "United States, Canada"
    }
    ]

    for query in queries:
        
        ## uncomment to run option = "word_similarity"
        #top_df = faers.top_k_relevant_data_retrieve(query, k=3, fetch_method="word_similarity")
        
        ## uncomment to run option = "encode_similarity"
        top_df = faers.top_k_relevant_data_retrieve(query, k=3, fetch_method="encode_similarity")


        print(f"Find 3 for query: {query}")
        for idx, row in top_df.iterrows():
            
            sumary = faers.row_to_natural_language(idx)
            print(sumary)
            print()
            print(f"serious: {row['serious']}")
            print(f"absage: {row['absage']}")
            print(f"patientsex: {row['patientsex']}")
            print(f"occurcountry: {row['occurcountry']}")
            print(f"activesubstancenames: {row['activesubstancenames']}")
            print(f"drugindications: {row['drugindications']}")
            print()
            
if __name__ == "__main__":
    retrieve_test()