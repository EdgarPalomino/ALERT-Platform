# LLama 3.2 3B

python src/run_pipeline.py \
  --dataset.data_path data/advbench/train-00000-of-00001.parquet \
  --dataset.limit 50 \
  --attack.use_identity true \
  --attack.use_roleplay true \
  --attack.use_obfuscation true \
  --attack.use_prefix_injection true \
  --attack.prefix_mode no_safety_or_refusal \
  --target.backend ollama \
  --target.model_name llama3.2-3b \
  --classifier.use_compliance true \
  --classifier.compliance_model llama3.2:latest \
  --classifier.use_llama_guard true \
  --classifier.llama_guard_model llama-guard3:1b \
  --output.run_name llama3_2_3b_experiment

# DeepSeek-R1 1.5B

python src/run_pipeline.py \
  --dataset.data_path data/advbench/train-00000-of-00001.parquet \
  --dataset.limit 50 \
  --attack.use_identity true \
  --attack.use_roleplay true \
  --attack.use_obfuscation true \
  --attack.use_prefix_injection true \
  --attack.prefix_mode no_safety_or_refusal \
  --target.backend ollama \
  --target.model_name deepseek-r1-1.5b \
  --classifier.use_compliance true \
  --classifier.compliance_model llama3.2:latest \
  --classifier.use_llama_guard true \
  --classifier.llama_guard_model llama-guard3:1b \
  --output.run_name deepseek_r1_1_5b_experiment

# GPT-4o Mini

python src/run_pipeline.py \
  --dataset.data_path data/advbench/train-00000-of-00001.parquet \
  --dataset.limit 50 \
  --attack.use_identity true \
  --attack.use_roleplay true \
  --attack.use_obfuscation true \
  --attack.use_prefix_injection true \
  --attack.prefix_mode no_safety_or_refusal \
  --target.backend openai_api \
  --target.model_name gpt-4o-mini \
  --target.base_url https://api.openai.com/v1 \
  --classifier.use_compliance true \
  --classifier.compliance_model llama3.2:latest \
  --classifier.use_llama_guard true \
  --classifier.llama_guard_model llama-guard3:1b \
  --output.run_name gpt_4o_mini_experiment

# Llama 3.1 8B Instruct

python src/run_pipeline.py \
  --dataset.data_path data/advbench/train-00000-of-00001.parquet \
  --dataset.limit 50 \
  --attack.use_identity true \
  --attack.use_roleplay true \
  --attack.use_obfuscation true \
  --attack.use_prefix_injection true \
  --attack.prefix_mode no_safety_or_refusal \
  --target.backend local_hf \
  --target.model_name llama3.1-8b-instruct \
  --target.hf_model_id meta-llama/Llama-3.1-8B-Instruct \
  --target.device_map auto \
  --target.torch_dtype float16 \
  --target.load_in_4bit true \
  --output.run_name llama_3_1_8b_instruct_experiment
