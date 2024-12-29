#!/bin/bash

# Directory to process
DIR="/path/to/your/directory"
TMP_DIR="/tmp/dedup"
mkdir -p "$TMP_DIR"

# Function to compute file hash
compute_hash() {
    local file="$1"
    local ext="${file##*.}"
    case "$ext" in
        txt)
            cat "$file" ;;
        html|htm)
            lynx -dump "$file" ;;
        pdf)
            pdftotext "$file" - ;;
        *)
            echo "Unsupported file type: $file" >&2
            return ;;
    esac | md5sum | awk '{print $1}'
}

# Remove duplicates
declare -A hash_map
find "$DIR" -type f | while read -r file; do
    hash=$(compute_hash "$file")
    if [[ -n "$hash" && "${hash_map[$hash]}" ]]; then
        echo "Duplicate found: $file (original: ${hash_map[$hash]})"
        rm "$file"
    else
        hash_map[$hash]="$file"
    fi
done
