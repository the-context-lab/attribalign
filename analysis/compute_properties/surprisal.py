""" 
Compute perplexities of model-generated and human responses.
We are interested in how perplexed certain high quality, held-out
models are by model-generated and human responses.

Usage:

    python surprisal.py compute_surprisal \
        --input_file path/to/sample_level.tsv \
        --output_file path/to/sample_level_perplexed.tsv
    
    python surprisal.py test
"""
import re
import torch
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
tqdm.pandas()


DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"

MODEL_NAMES = {
    'gpt2': 'gpt2',
    'EleutherAI/pythia-1.4b': 'pythia'
}


def safe_model_name(model_id):
    return re.sub(r'\W+', '-', model_id)


def surprisal(
        text: str, 
        model: 'transformers.AutoModelForCausalLM',
        tokenizer: 'transformers.AutoTokenizer'
) -> float:
    """Compute perplexity of a single utterance."""
    if not isinstance(text, str):
        return None
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(DEVICE)
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss.item()
    return loss


def surprisal_context(
        text: str, 
        context: str, 
        model: 'transformers.AutoModelForCausalLM',
        tokenizer: 'transformers.AutoTokenizer'
) -> float:
    """
    Compute perplexity of a single utterance with the 
    model conditioned on some context.
    """
    if not isinstance(text, str):
        return None
    n_tokens_target = len(tokenizer.tokenize(text))
    model_input = context.replace('\\n', '\n') + text
    encodings = tokenizer(model_input, return_tensors="pt")
    input_ids = encodings.input_ids.to(DEVICE)
    masked_labels = input_ids.clone()
    masked_labels[:, :n_tokens_target] = -100  # mask context input ids to -100
    outputs = model(input_ids, labels=masked_labels)
    loss = outputs.loss.item()
    return loss


def compute_surprisal(
        input_file: str,
        output_file: str = None,
        model_ids: list = list(MODEL_NAMES.keys()),
        n_model_responses: int = 5,
        test_mode: bool = False,
):
    """
    Compute perplexities of model-generated and human responses.
    Input is a sample-level dataframe. This function will add new
    columns to the input dataframe.

    Args
    ----
    input_file : str
        path to sample-level input dataframe
    output_file : str
        path to save output dataframe
    model_ids : list
        list of model ids to use
    n_model_responses : int
        number of model responses to the same dialogue excerpt
    test_mode : bool
        in test mode, we're only using `gpt2` to compute perplexities
    """
    model_ids = {
        model_id: safe_model_name(model_id) \
            if not model_id in MODEL_NAMES.keys() \
                else MODEL_NAMES[model_id]
        for model_id in model_ids
    }
    if test_mode:
        model_ids = {'gpt2': 'gpt2'}


    print('Using model(s) ' + ', '.join([f"'{model_id}'" for model_id in model_ids.keys()])
          + f" on '{DEVICE}' to compute perplexities.")
    print('Loading models and tokenizers...')

    # load models and tokenizers
    models = [
        AutoModelForCausalLM.from_pretrained(model_id).to(DEVICE)
        for model_id in model_ids.keys()
    ]
    tokenizers = [
        AutoTokenizer.from_pretrained(model_id)
        for model_id in model_ids.keys()
    ]
    print('Models loaded.')

    # read df and compute perplexities
    df = pd.read_csv(input_file, sep='\t', header=0)
    columns_now = df.columns
    for model_idx, model_name in enumerate(model_ids.values()):
        print('-'*60)
        print(f"Model '{model_name.upper()}' ({model_idx+1}/{len(model_ids)})")

        # human responses, no context
        print(f'[1/4] Human responses, no context...')
        df[f"ppl_human_response_{model_name}"] = df['human_response'].progress_apply(
            lambda x: surprisal(
                text=x,
                model=models[model_idx],
                tokenizer=tokenizers[model_idx]
            ),
        )

        # human responses, with context
        print(f'[2/4] Human responses, with context...')
        df[f"ppl_human_response_context_{model_name}"] = df.progress_apply(
            lambda x: surprisal_context(
                text=x['human_response'],
                context=x['prompt'],
                model=models[model_idx],
                tokenizer=tokenizers[model_idx]),
            axis=1,
        )

        # check if model responses are present
        if 'model_response_r0' not in df.columns:
            print('WARNING: No model responses found in input dataframe.')
            continue

        # model responses, no context
        for response_idx in range(n_model_responses):
            print(f'[3/4] Model responses, no context, response {response_idx}'
                  f'... ({response_idx+1}/{n_model_responses})')
            df[f"ppl_model_response_r{response_idx}_{model_name}"] = \
                df[f"model_response_r{response_idx}"].progress_apply(
                lambda x: surprisal(
                    text=x,
                    model=models[model_idx],
                    tokenizer=tokenizers[model_idx]
                ),
            )

        # model responses, with context
        for response_idx in range(n_model_responses):
            print(f'[4/4] Model responses, with context, response {response_idx}'
                  f'... ({response_idx+1}/{n_model_responses})')
            df[f"ppl_model_response_r{response_idx}_context_{model_name}"] = df.progress_apply(
                lambda x: surprisal_context(
                    text=x[f"model_response_r{response_idx}"],
                    context=x['prompt'],
                    model=models[model_idx],
                    tokenizer=tokenizers[model_idx]
                ),
                axis=1,
            )

    # done, save results and report added columns
    columns_added = df.columns.difference(columns_now)
    print(f"Added columns ({len(columns_added)}):\n" \
        + '\n'.join([f'  - {col}' for col in columns_added]))
    save_to = input_file
    if output_file:
        save_to = output_file
    df.to_csv(save_to, sep='\t', header=True, index=False)
    print(f"Saved to '{save_to}'.")


def test_compute_surprisal():
    input_file = '../data/output/sample_level.tsv'
    output_file = '../data/output/sample_level_perplexed.tsv'
    compute_surprisal(input_file, output_file)


if __name__ == '__main__':
    import fire
    fire.Fire({
        'compute_surprisal': compute_surprisal,
        'test': test_compute_surprisal,
    })
