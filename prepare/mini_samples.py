"""
Create a `samples_mini.tsv` under `data/` that has the same 
columns as `data/samples.tsv` but less rows.
This file is used for testing the evaluation pipeline.

Usage:

    python prepare/mini_samples.py

    python prepare/mini_samples.py \
        --input_file data/samples.tsv \
        --output_file data/samples_mini.tsv \
        --n_samples_per_corpus 5
"""
import fire
import pandas as pd
from pathlib import Path


def create_mini_samples(
        input_file: str = None,
        output_file: str = None,
        n_samples_per_corpus: int = 5,
):
    # determine input and output files
    if input_file is None:
        input_file = Path(__file__).parent.parent / 'data' / 'samples.tsv'
    if output_file is None:
        output_file = Path(__file__).parent.parent / 'data' / 'samples_mini.tsv'

    # check if output file already exists
    if Path(output_file).exists():
        print(f'Output file {output_file} already exists.')
        print('Overwrite? [y/n]')
        if input() != 'y':
            print('Aborting.')
            return
        
    # check if input file exists
    if not Path(input_file).exists():
        print(f'Input file {input_file} does not exist.')
        print('Aborting.')
        return

    # load samples
    samples = pd.read_csv(
        input_file, 
        sep='\t', 
        header=0, 
        index_col='sample_index'
    )
    print(f'Loaded {len(samples)} samples from {input_file}')

    # filter for n_samples_per_corpus
    samples_mini = pd.DataFrame(columns=samples.columns)
    for corpus in samples['corpus'].unique():
        samples_corpus = samples[samples['corpus'] == corpus]
        samples_corpus_mini = samples_corpus.sample(n=min(n_samples_per_corpus, len(samples_corpus)))
        samples_mini = pd.concat([samples_mini, samples_corpus_mini])

    # save samples
    samples_mini.to_csv(
        output_file, 
        sep='\t', 
        header=True, 
        index='sample_index', 
        index_label='sample_index'
    )
    print(f'Saved {len(samples_mini)} samples to {output_file}')


if __name__ == '__main__':
    fire.Fire(create_mini_samples)
