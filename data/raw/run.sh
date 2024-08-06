#!/bin/bash
mkdir -p Compounds
cd Compounds
wget -r -l1 -nd -np -A "*.sdf.gz,*.md5" -e robots=off https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/

verify_and_extract() {
    local file=$1
    local md5_file=$file.md5

    if [ -f "$md5_file" ]; then
        expected_md5=$(cat "$md5_file" | awk '{ print $1 }')
        actual_md5=$(md5sum "$file" | awk '{ print $1 }')

        if [ "$expected_md5" == "$actual_md5" ]; then
            gunzip "$file"
            echo "Verified and extracted: $file"
        else
            echo "Checksum mismatch for $file. Skipping extraction."
        fi
    else
        echo "MD5 file not found for $file. Skipping extraction."
    fi
}

# Loop over all downloaded .sdf.gz files
for sdf_file in *.sdf.gz; do
    verify_and_extract "$sdf_file"
done

cd ..
python3 sdf_to_smiles.py --sdf_dir Compounds --output_file smiles.txt

# Get SMILES from ChEMBL and append it to smiles.txt
wget https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_34_chemreps.txt.gz
gunzip chembl_34_chemreps.txt.gz
tail -n +2 chembl_34_chemreps.txt | cut -f2 >> smiles.txt

cd ..

mv raw/smiles.txt processing/