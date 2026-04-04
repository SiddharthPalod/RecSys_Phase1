2D Layout Synthesis (using local Ollama):
python run_layoutgpt_2d.py --llm_type ollama --icl_type fixed-random --setting counting --n_iter 5 --test --verbose

python run_layoutgpt_2d.py --llm_type ollama --icl_type fixed-random --setting room --n_iter 5 --test --verbose

3D Layout Synthesis (using Gemini):
python run_layoutgpt_3d.py --llm_type gpt4 --test --verbose
