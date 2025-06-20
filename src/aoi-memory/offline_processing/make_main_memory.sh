#! /bin/bash
LOG_TO_FILE=data/log_0620.txt
export OPENAI_API_KEY=$(cat ../../configs/openai_api_key)
python offline_processing/make_main_memory.py \
    --input_db_type sqlite \
    --input_db_path data/memory_0620.db \
    --output_db_type sqlite \
    --output_db_path data/main_memory_0620.db \
    --llm_model gpt-4.1-nano \
    --timeslice_size 1day 2>&1 | tee $LOG_TO_FILE