#!/usr/bin/env bash
set -e
FILE_ID="1T14_iOJwgcMrbyDK1J_tqBaClUJ3QrlQ"
OUT="data/mass_roads.zip"
mkdir -p data
python - <<PY
import gdown, os
url = f'https://drive.google.com/uc?id={FILE_ID}'
if not os.path.exists('data/mass_roads.zip'):
    gdown.download(url, 'data/mass_roads.zip', quiet=False)
PY
unzip -q data/mass_roads.zip -d data/mass_roads
echo "Done. Dataset extracted to data/mass_roads/tiff/"
