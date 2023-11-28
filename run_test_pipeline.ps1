"======================================================"
"[PREP 1/3] DOWNLOADING CORPORA"
"======================================================"
python prepare/prepare_corpora.py download


"======================================================"
"[PREP 2/3] PREPARING CORPORA"
"======================================================"
python prepare/prepare_corpora.py prepare --context_length 10


"======================================================"
"[PREP 3/3] CREATING TESTING SUB-SET"
"======================================================"
python prepare/mini_samples.py


"======================================================"
"[1/6] GENERATING RESPONSES AND ATTRIBUTING"
"======================================================"
python generate/generate_and_attribute.py full_attribution `
--corpus switchboard `
--model_id gpt2 `
--input_file data/samples_mini.tsv `
--output_file data/samples_mini_gpt2.tsv

python generate/generate_and_attribute.py full_attribution `
--corpus switchboard `
--model_id gpt2 `
--style comprehend `
--input_file data/samples_mini_gpt2.tsv `
--output_file data/samples_mini_gpt2.tsv


"======================================================"
"[2/6] COMPUTING GENERATION QUALITY METRICS"
"======================================================"
python analysis/compute_properties/generation_quality.py run `
--input_file data/samples_mini_gpt2.tsv `
--output_file data/samples_mini_gpt2_genq.tsv `
--corpus switchboard `
--test_mode


"======================================================"
"[3/6] EXTRACTING CONSTRUCTIONS"
"======================================================"
python analysis/compute_properties/constructions.py `
--input_file data/samples_mini_gpt2_genq.tsv `
--output_file data/samples_mini_gpt2_genq_constr.tsv `
--working_dir data/_tmp_dialign/ `
--delete_working_dir False


"======================================================"
"[4/6] COMPUTING SURPRISAL"
"======================================================"
python analysis/compute_properties/surprisal.py compute_surprisal `
--input_file data/samples_mini_gpt2_genq_constr.tsv `
--output_file data/samples_mini_gpt2_genq_constr_ppl.tsv `
--test_mode


"======================================================"
"[5/6] COMPUTING OVERLAP SCORES"
"======================================================"
python analysis/compute_properties/overlaps.py `
--input_file data/samples_mini_gpt2_genq_constr_ppl.tsv `
--output_file data/samples_mini_gpt2_genq_constr_ppl_ol.tsv `
--dialogues_dir data/_tmp_dialign/dialogues/switchboard/ `
--lexica_dir data/_tmp_dialign/lexica/switchboard `
--corpus switchboard


"======================================================"
"[6/6] COMPUTING PMI"
"======================================================"
python analysis/compute_properties/pmi.py `
--input_file data/samples_mini_gpt2_genq_constr_ppl_ol.tsv `
--output_file data/samples_mini_gpt2_genq_constr_ppl_ol_pmi.tsv `
--dialign_output_dir data/_tmp_dialign/lexica/switchboard/ `
--dialign_input_dir data/_tmp_dialign/dialogues/switchboard/ `
--clean_working_dir False


"======================================================"
"CLEANUP"
"======================================================"
python analysis/compute_properties/cleanup.py


"======================================================"
"DONE."
"======================================================"
