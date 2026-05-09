conda run -n mineru python3 -u /data/lsb/deep_past/script/extract_akt8_mineru_ocr.py \
  --source-start-page 49 --source-end-page 517 \
  --ocr-lang latin --mineru-device cuda \
  --output-dir /data/lsb/deep_past/output/akt8_mineru_ocr_extract