"""
Determine the quality of model-generated responses to dialogue exceprts.
Compute the following metrics:
- BERTScore
- BLEU
- MAUVE (https://github.com/krishnap25/mauve)
 """
import os
import sys
import shortuuid
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from evaluate import load
from mauve import compute_mauve


# make sure script can be run from anywhere
SCRIPT_PATH = Path(__file__).parent.resolve()
DATA_FOLDER = SCRIPT_PATH.parent.parent / 'data'


def compute_generation_quality(
        input_file: Path = None,
        output_file: Path = None,
        corpus: str = None,
        n_model_responses: int = 5,
        test_mode: bool = False,
        working_dir: Path = None,
        save_intermediate: bool = False,
):
    """
    Compute BERTScore, BLEU and MAUVE for model-generated responses to dialogue excerpts.

    Note 1: BLEU and MAUVE are corpus-level metrics, so added columns in the
            will have the same values across all rows
    Note 2: BERTScore is sample-level, so added columns will have different
            values for each row
    """
    
    # check if input file exists
    if not os.path.exists(input_file):
        print(f'Input file {input_file} does not exist.')
        sys.exit(1)
    
    # set up working directory
    print(f'Input file: {input_file}')
    if working_dir is None:
        working_dir = DATA_FOLDER / Path('_tmp_gen_quality_' + shortuuid.uuid()[:6])
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    print(f'Working directory: {working_dir}')

    # add new columns to samples dataframe
    samples = pd.read_csv(input_file, sep='\t', header=0, index_col='sample_index')
    samples['bleu_score'] = np.nan
    samples['bleu_brevity_penalty'] = np.nan
    samples['bleu_length_ratio'] = np.nan
    samples['mauve_score'] = np.nan
    # there will be a bertf1 for each model response
    for i in range(n_model_responses):
        samples[f'bert_f1_r{i}'] = np.nan

    if corpus is not None:
        print(f'Filtering for corpus: {corpus}')
        samples_filtered = samples[samples['corpus'] == corpus]
    if len(samples_filtered) == 0:
        print('No samples found.')
        sys.exit(1)

    # construct pairs dataframe
    print('Constructing pairs dataframe...')
    pairs = pd.DataFrame(columns=['model_response', 'human_response', 'sample_index'])
    for sample_index, sample_row in tqdm(samples_filtered.iterrows(), total=len(samples_filtered)):
        human_response = sample_row['human_response']
        model_responses = [sample_row[f"model_response_r{i}"] for i in range(n_model_responses)]
        for model_response_idx,model_response in enumerate(model_responses):
            pairs = pd.concat([pairs, pd.DataFrame({
                'model_response': [model_response],
                'human_response': [human_response],
                'sample_index': [sample_index],
                'model_response_index': [model_response_idx],
            })])
    pairs = pairs.reset_index(drop=True)
    print(f'Number of pairs: {len(pairs)}')

    # load bertscore and bleu
    bleu = load("bleu")
    bert = load("bertscore")

    # compute bleu
    print('Computing BLEU score...')
    bleu_results = bleu.compute(
        predictions=pairs['model_response'].values, 
        references=pairs['human_response'].values)
    print("Done.")
    bleu_score = bleu_results['bleu']
    bleu_brevity_penalty = bleu_results['brevity_penalty']
    bleu_length_ratio = bleu_results['length_ratio']
    print(f"BLEU score: {bleu_score}")
    print(f"Brevity penalty: {bleu_brevity_penalty}")
    print(f"Length ratio: {bleu_length_ratio}")
    # add to pairs dataframe
    pairs['bleu_score'] = [bleu_score] * len(pairs)
    pairs['bleu_brevity_penalty'] = [bleu_brevity_penalty] * len(pairs)
    pairs['bleu_length_ratio'] = [bleu_length_ratio] * len(pairs)

    # compute bertscore
    print('Computing BERT score...')
    bert_results = bert.compute(
        predictions=pairs['model_response'].values,
        references=pairs['human_response'].values,
        model_type="distilbert-base-uncased", 
        use_fast_tokenizer=True,
        verbose=True)
    mean_bert_f1 = np.mean(bert_results['f1'])
    n_bert_f1s = len(bert_results['f1'])
    bert_f1s = bert_results['f1']
    print(f"Mean BERT F1 score: {mean_bert_f1}")
    print(f"Number of BERT F1 scores: {n_bert_f1s}")
    print("Done.")
    # add to pairs dataframe
    pairs['bert_f1'] = bert_f1s

    # compute mauve
    print('Computing MAUVE score...')
    mauve_model_name = 'gpt2-large' if not test_mode else 'gpt2'
    num_buckets = 500 if not test_mode else 25
    print(f'Using model {mauve_model_name} for featurization.')

    mauve_results = compute_mauve(
        p_text=pairs['model_response'].values,
        q_text=pairs['human_response'].values,
        device_id=0,  # use GPU 0 for featurization
        max_text_length=256,  # truncate text to x token
        verbose=False,  # print progress
        # num_buckets=num_buckets,  # quantize to x buckets
        num_buckets=num_buckets,  # quantize to x buckets
        featurize_model_name=mauve_model_name,  # use this model for featurization
    )
    mauve_score = mauve_results.mauve
    print(f'MAUVE score: {mauve_score}')
    print('Done.')
    # add to pairs dataframe
    pairs['mauve_score'] = [mauve_score] * len(pairs)

    # save intermediate
    if save_intermediate:
        print('Saving results to working directory...')
        pairs.to_csv(os.path.join(working_dir, 'pairs.tsv'), sep='\t', index=False, header=True)
        np.save(os.path.join(working_dir, 'bert_results.npy'), bert_results)
        np.save(os.path.join(working_dir, 'bleu_results.npy'), bleu_results)
        np.save(os.path.join(working_dir, 'mauve_results.npy'), mauve_results)
        print('Done.')

    # add scores to samples dataframe
    print('Adding scores to samples dataframe...')
    for _, pair in tqdm(pairs.iterrows(), total=len(pairs)):
        sample_idx = pair['sample_index']
        model_response_idx = str(int(pair['model_response_index']))
        samples.at[sample_idx, 'bleu_score'] = pair['bleu_score']
        samples.at[sample_idx, 'bleu_brevity_penalty'] = pair['bleu_brevity_penalty']
        samples.at[sample_idx, 'bleu_length_ratio'] = pair['bleu_length_ratio']
        samples.at[sample_idx, 'mauve_score'] = pair['mauve_score']
        samples.at[sample_idx, f'bert_f1_r{model_response_idx}'] = pair['bert_f1']
    print('Done.')

    # save samples dataframe
    if output_file is None:
        output_file = input_file
    print(f'Saving to {output_file}...')
    samples.to_csv(output_file, sep='\t', index=True, header=True, index_label='sample_index')
    print('Done.')

    # clean up
    if not save_intermediate:
        os.system(f'rm -rf {working_dir}')
    else:
        print(f'Data saved to working directory {working_dir}.')
    print('Finished.')


def test_computation():
    compute_generation_quality(
        input_file=os.path.join(DATA_FOLDER, 'samples_mini_gpt2.tsv'),
        output_file=os.path.join(DATA_FOLDER, 'samples_mini_gpt2_genq.tsv'),
        corpus='switchboard',
        n_model_responses=5,
        test_mode=True,
        save_intermediate=True,
        working_dir=None,
    )


if __name__ == '__main__':
    import fire
    fire.Fire({
        'run': compute_generation_quality,
        'test': test_computation,
    })
    test_computation()
