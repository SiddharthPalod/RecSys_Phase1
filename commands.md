<!-- Phase 1 Terminal Generation Command -->
python Phase1/layout_terminal_interface.py --dataset "Phase1/detailed_fixed_strict.json"

python Phase1/layout_terminal_interface.py --dataset "Phase1/detailed_fixed_strict.json" --show_results --filtered_items_out "Phase1/filter.json"

Without Launching LayoutGPT
python Phase1/layout_terminal_interface.py --dataset "Phase1/detailed_fixed_strict.json" --dry_run

Status of BG Jobs
python "Phase1/layout_terminal_interface.py" --job_status

Non-interactive rerun to update filter.json file
python "Phase1/layout_terminal_interface.py" --dataset "Phase1/room_dataset_cleaned.json" --show_results --filtered_items_out "Phase1/filter.json" --dry_run

<!-- Layout GPT JSON Generation Commands -->
2D Layout Synthesis (using local Ollama):
python run_layoutgpt_2d.py --llm_type ollama --icl_type fixed-random --setting counting --n_iter 5 --test --verbose

python run_layoutgpt_2d.py --llm_type ollama --icl_type fixed-random --setting room --n_iter 5 --test --verbose

python run_layoutgpt_2d.py --llm_type ollama --ollama_model llama3.2:1b --ollama_temperature 0.95 --ollama_top_p 0.95 --ollama_num_predict 512 --icl_type fixed-random --setting counting --val_json prompt_lists/room_prompts_v1.json --n_iter 5 --output_file llm_output/counting/ollama.counting.room_dataset.json --resume --incremental      

High-variation dataset generation with Ollama (many prompts, incl. empty/dense rooms):
python make_room_prompt_list.py --out prompt_lists/room_prompts_v1.json --n 2000 --seed 123
python run_layoutgpt_2d.py --llm_type ollama --ollama_model llama3.2:1b --ollama_temperature 0.95 --ollama_top_p 0.95 --ollama_num_predict 512 --icl_type fixed-random --setting counting --val_json prompt_lists/room_prompts_v1.json --n_iter 5 --verbose --output_file llm_output/counting/ollama.counting.room_dataset.json

3D Layout Synthesis (using Gemini via google.generativeai; set GEMINI_API_KEY). Requires preprocessed ATISS data under ./ATISS/data_output (see README Data Preparation); override with --dataset_dir if needed:
python run_layoutgpt_3d.py --llm_type gpt4 --test --verbose

3D with local Ollama (same script):
python run_layoutgpt_3d.py --llm_type ollama --ollama_model llama3.2:1b --test --verbose