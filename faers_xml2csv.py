import os
import sys
import xml.etree.ElementTree as ET
import glob
import pandas as pd
from tqdm import tqdm

# Make absolute FAERS path
cwd_path = os.path.dirname(os.path.realpath(__file__))
FAERS_ABS = cwd_path


from utils.faers_label_dictionary import COUNTRY_LABELS, REPORTTYPE_LABELS, SERIOUS_LABELS, SENDERTYPE_LABELS, \
    PATIENTONSETAGEUNIT_LABELS, PATIENTSEX_LABELS, REACTIONOUTCOME_LABELS, \
    DRUGCHARACTERIZATION_LABELS, ACTIONDRUG_LABELS, DRUGADDITIONAL_LABELS


def find_file_paths():
    xml_dirs = ["raw/xml", "raw/XML/"]
    file_paths = []
    for dir_path in xml_dirs:
        full_dir = os.path.join(FAERS_ABS, dir_path)
        xml_files = glob.glob(os.path.join(full_dir, "*.xml"))
        for file_path in xml_files:
            file_paths.append(file_path)
    return file_paths


def xmls_to_csv(
    sex_option="all",            # "all", "maleonly", "femaleonly"
    keep_unknown_outcome=True,   # True, False
    drugchar_option="all",       # "all", "suspect_only", "remove_unknown"
    max_text_length=None         # None or int
):
    
    # ---- Parameter validation ----
    valid_sex_options = {"all", "maleonly", "femaleonly"}
    valid_drugchar_options = {"all", "suspect_only", "remove_unknown"}

    if sex_option not in valid_sex_options:
        raise ValueError(f"Invalid sex_option: {sex_option}. Must be one of {valid_sex_options}")

    if not isinstance(keep_unknown_outcome, bool):
        raise ValueError(f"Invalid keep_unknown_outcome: {keep_unknown_outcome}. Must be True or False")

    if drugchar_option not in valid_drugchar_options:
        raise ValueError(f"Invalid drugchar_option: {drugchar_option}. Must be one of {valid_drugchar_options}")
    
    if max_text_length is not None:
        if not isinstance(max_text_length, int) or max_text_length <= 0:
            raise ValueError(f"max_text_length must be None or a positive integer, got {max_text_length}")
    
    print("Function called with parameters:")
    print(f"  sex_option: {sex_option}")
    print(f"  keep_unknown_outcome: {keep_unknown_outcome}")
    print(f"  drugchar_option: {drugchar_option}")
    print(f"  max_text_length: {max_text_length}")

    out_dir = os.path.join(FAERS_ABS, "out")
    os.makedirs(out_dir, exist_ok=True)

    file_paths = find_file_paths()
    print(f"File paths found: {len(file_paths)}")

    for file_path in tqdm(file_paths, desc="Processing XML files"):
        
        rows = []
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            for sr in root.findall("safetyreport"):
                row = {}

                # ---- General info ----
                row["safetyreportid"] = sr.findtext("safetyreportid") or None
                row["safetyreportversion"] = sr.findtext("safetyreportversion") or None

                occurcountry_code = sr.findtext("occurcountry") or None
                row["occurcountry"] = COUNTRY_LABELS.get(occurcountry_code, None)

                reporttype_code = sr.findtext("reporttype") or None
                row["reporttype"] = REPORTTYPE_LABELS.get(reporttype_code, None)

                # Seriousness breakdown
                serious_code = sr.findtext("serious") or None
                row["serious"] = SERIOUS_LABELS.get(serious_code, None)

                serious_parts = []
                mapping = {
                    "seriousnesscongenitalanomali": "Congenital anomaly",
                    "seriousnessdeath": "Death",
                    "seriousnessdisabling": "Disabling",
                    "seriousnesshospitalization": "Hospitalization",
                    "seriousnesslifethreatening": "Life-threatening",
                    "seriousnessother": "Other",
                }

                serious_specific = {
                    "seriousnesscongenitalanomali": SERIOUS_LABELS.get(sr.findtext("seriousnesscongenitalanomali") or None, None),
                    "seriousnessdeath": SERIOUS_LABELS.get(sr.findtext("seriousnessdeath") or None, None),
                    "seriousnessdisabling": SERIOUS_LABELS.get(sr.findtext("seriousnessdisabling") or None, None),
                    "seriousnesshospitalization": SERIOUS_LABELS.get(sr.findtext("seriousnesshospitalization") or None, None),
                    "seriousnesslifethreatening": SERIOUS_LABELS.get(sr.findtext("seriousnesslifethreatening") or None, None),
                    "seriousnessother": SERIOUS_LABELS.get(sr.findtext("seriousnessother") or None, None),
                }

                for key, value in serious_specific.items():
                    if value:  # only include if not None
                        serious_parts.append(f"{mapping[key]} ({value})")

                
                serious_text = "; ".join(serious_parts) if serious_parts else None
                if max_text_length and serious_text:
                    serious_text = serious_text[:max_text_length]
                row["serious_specific"] = serious_text

                # Dates
                row["receivedate"] = sr.findtext("receivedate") or None
                row["transmissiondate"] = sr.findtext("transmissiondate") or None

                # Company
                row["companynumb"] = sr.findtext("companynumb") or None

                # Sender
                sender = sr.find("sender")
                if sender is not None:
                    sendertype_code = sender.findtext("sendertype") or None
                    row["sendertype"] = SENDERTYPE_LABELS.get(sendertype_code, None)
                    row["senderorganization"] = sender.findtext("senderorganization") or None
                else:
                    row["sendertype"] = None
                    row["senderorganization"] = None

                # ---- Patient info ----
                patient = sr.find("patient")
                if patient is not None:
                    row["patientonsetage"] = patient.findtext("patientonsetage") or None

                    onset_age_unit_code = patient.findtext("patientonsetageunit") or None
                    row["patientonsetageunit"] = PATIENTONSETAGEUNIT_LABELS.get(onset_age_unit_code, None)

                    sex_code = patient.findtext("patientsex") or None
                    sex_label = PATIENTSEX_LABELS.get(sex_code, None)
                    row["patientsex"] = sex_label

                    # ---- Filter by sex_option ----
                    if sex_option.lower() == "maleonly" and sex_label != "Male":
                        continue
                    if sex_option.lower() == "femaleonly" and sex_label != "Female":
                        continue

                    # Reactions (multi)
                    reactions = []
                    for reaction in patient.findall("reaction") or []:
                        outcome_code = reaction.findtext("reactionoutcome") or None
                        outcome_label = REACTIONOUTCOME_LABELS.get(outcome_code, "Missing/Unknown Outcome")
                        term = reaction.findtext("reactionmeddrapt") or "Unspecified reaction"

                        # keep_unknown_outcome=False, skip unknown reaction
                        if not keep_unknown_outcome and "Unknown" in outcome_label:
                            continue

                        reactions.append(f"{term} ({outcome_label})")

                    # if keep_unknown_outcome=False and reactions Unknown，skip report
                    if not keep_unknown_outcome and not reactions:
                        continue  # skip whole report

                    # save as flat text
                   
                    reactions_text = "; ".join(reactions) if reactions else None
                    if reactions_text and max_text_length:
                        reactions_text = reactions_text[:max_text_length]
                    row["reactions"] = reactions_text


                        
                        
                else:
                    # No patient info
                    continue


                # ---- Drugs info ----
                row["drugs_specific"] = []
                row["drugnames"] = []               # flattened list of all medicinalproduct names
                row["activesubstancenames"] = []    # flattened list of all active substances
                row["drugindications"] = []         # flattened list of all indication names

                if patient is not None:
                    for drug in patient.findall("drug") or []:
                        char_code = drug.findtext("drugcharacterization") or None
                        char_label = DRUGCHARACTERIZATION_LABELS.get(char_code, "Unknown")

                        # ---- Filter by drugchar_option ----
                        if drugchar_option == "suspect_only" and char_label != "Suspect":
                            continue
                        if drugchar_option == "remove_unknown" and char_label in ["Unknown", None]:
                            continue

                        drug_name = drug.findtext("medicinalproduct") or "Unknown drug"
                        if drug_name:
                            row["drugnames"].append(drug_name.lower())

                        # Active substances
                        active_list = []
                        for active in drug.findall("activesubstance") or []:
                            active_name = active.findtext("activesubstancename") or None
                            if active_name:
                                active_list.append(active_name)
                                row["activesubstancenames"].append(active_name.lower())

                        # Drug additional / action
                        action_label = ACTIONDRUG_LABELS.get(drug.findtext("actiondrug") or None, "Unknown")
                        additional_label = DRUGADDITIONAL_LABELS.get(drug.findtext("drugadditional") or None, "Unknown")
                        dosage_text = drug.findtext("drugdosagetext") or ""
                        indication_text = drug.findtext("drugindication") or ""
                        
                        if indication_text != "":
                            row["drugindications"].append(indication_text.lower())

                        # Flatten to human-readable string
                        flat_drug_text = f"{drug_name} ({char_label})"
                        if active_list:
                            flat_drug_text += f", Active: {', '.join(active_list)}"
                        if dosage_text:
                            flat_drug_text += f", Dosage: {dosage_text}"
                        if indication_text:
                            flat_drug_text += f", Indication: {indication_text}"
                        flat_drug_text += f", Action: {action_label}"
                        if additional_label != "Unknown":
                            flat_drug_text += f", Additional: {additional_label}"

                        row["drugs_specific"].append(flat_drug_text)
                        
                    
                    row["drugs_specific"] = "; ".join(set(row["drugs_specific"]))
                        


                # Remove duplicates for NLP-friendly table
                row["drugnames"] = list(set(row["drugnames"]))
                row["activesubstancenames"] = list(set(row["activesubstancenames"]))
                row["drugindications"] = list(set(row["drugindications"]))


                rows.append(row)
                

            # Save CSV
            if rows:
                filename = os.path.splitext(os.path.basename(file_path))[0]
                output_path = os.path.join(out_dir, f"{filename}.csv")
                df = pd.DataFrame(rows)
                df.to_csv(output_path, index=False)
                print(f"Saved {len(rows)} rows → {output_path}")
            else:
                print(f"No rows extracted from {file_path}")

        except Exception as e:
            print(f"Error parsing {file_path}: {e}")

    

    
    
if __name__ == "__main__":

    
    xmls_to_csv(sex_option="all", keep_unknown_outcome=False, drugchar_option="suspect_only", max_text_length=None)

    "read the csv"
    # out_dir = os.path.join(FAERS_ABS, "out")
    # # List all CSVs
    # csv_files = [f for f in os.listdir(out_dir) if f.endswith(".csv")]
    # print("Found CSVs:", csv_files)
    # # Read one CSV
    # if csv_files:
    #     df = pd.read_csv(os.path.join(out_dir, csv_files[0]))
    #     print("Shape:", df.shape)
    #     print(df.head())



