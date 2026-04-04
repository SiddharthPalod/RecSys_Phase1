2D Layout Synthesis (using local Ollama):
python run_layoutgpt_2d.py --llm_type ollama --icl_type fixed-random --setting counting --n_iter 5 --test --verbose

python run_layoutgpt_2d.py --llm_type ollama --icl_type fixed-random --setting room --n_iter 5 --test --verbose

3D Layout Synthesis (using Gemini via google.generativeai; set GEMINI_API_KEY). Requires preprocessed ATISS data under ./ATISS/data_output (see README Data Preparation); override with --dataset_dir if needed:
python run_layoutgpt_3d.py --llm_type gpt4 --test --verbose

3D with local Ollama (same script):
python run_layoutgpt_3d.py --llm_type ollama --ollama_model llama3.2:1b --test --verbose
