#!/bin/bash
set -euo pipefail

# Base dir of the script itself (so it's independent of where you run it)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
FAERS_PATH=$SCRIPT_DIR

# Create dirs
RAW_DIR="${FAERS_PATH}/raw"
mkdir -p "$RAW_DIR"

base_url="https://fis.fda.gov/content/Exports"

# Download FAERS data from 2010 to 2024, quarters 1 to 4
for year in $(seq 2010 2024); do
  for q in {1..4}; do
    file="faers_xml_${year}Q${q}.zip"
    url="${base_url}/${file}"
    echo "Checking $url ..."
    
    if wget --spider -q "$url"; then
      echo "Downloading $file ..."
      wget -q -O "${RAW_DIR}/${file}" "$url"
    else
      echo "Not found: $file"
    fi
  done
done

# Extract all downloaded zip files
for zipfile in "$RAW_DIR"/*.zip; do
  echo "Extracting $zipfile ..."
  unzip -o "$zipfile" -d "$RAW_DIR"
done

echo "All available files have been downloaded and extracted into $RAW_DIR"
