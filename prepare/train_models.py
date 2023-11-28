"""Train and evaluate models on the prepared corpora."""
import os
import re
import torch
from itertools import chain
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')

# Switchboard and Map Task train and validation files
FILES = {
    'train': {
        'switchboard': DATA_DIR+'/corpora/model_train/switchboard/train.txt',
        'maptask': DATA_DIR+'/corpora/model_train/maptask/train.txt',
    },
    'val': {
        'switchboard': DATA_DIR+'/corpora/model_train/switchboard/val.txt',
        'maptask': DATA_DIR+'/corpora/model_train/maptask/val.txt',
    }
}

# models trained by default
ALL_MODELS = [
    'gpt2',
    'facebook/opt-125m',
    'microsfot/DialoGPT-small',
    # 'EleutherAI/python-1.4B',
]


def safe_model_name(model_id):
    return re.sub(r'\W+', '-', model_id)


def train_model(
        model_id: str,
        block_size: int,
        n_epochs: int,
        train_file: str,
        eval_file: str,
        output_dir: str = None,
        device: str = None,
) -> None:
    """Train a model on a given corpus."""

    print(f"Training model '{model_id}'")
    print(f"Training file: {train_file}")
    print(f"Eval file: {eval_file}")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    # set up output directory
    if output_dir is None:
        output_dir = f"{DATA_DIR}/models/{safe_model_name(model_id)}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Output dir: {output_dir}")

    # load and tokenize datasets
    print('Processing data...')
    def tokenize_function(examples):
        return tokenizer(examples["text"])
    raw_datasets = load_dataset(
        'text',
        data_files={
            'train': train_file,
            'validation': eval_file,
        },
        # cache_dir='data/.cache',
        keep_linebreaks=True,
    )
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        remove_columns=['text'],
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    # prepare datasets for training
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=None,
        load_from_cache_file=False,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    # create data collator
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # set up training arguments
    training_args = TrainingArguments(
        # ----------------------
        # training hyperparameters
        per_device_train_batch_size=7,
        num_train_epochs=n_epochs,
        # ----------------------
        # saving best model
        do_eval=True,
        save_total_limit=1,
        evaluation_strategy="steps",
        eval_steps=0.05,
        save_strategy="steps",
        save_steps=0.05,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # ----------------------
        # misc
        output_dir=output_dir,
        overwrite_output_dir=True,
        logging_dir=output_dir,
        logging_steps=0.05,
        report_to="none",
    )

    # train model
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
    )
    print('Training model...')
    trainer.train()
    trainer.save_model()
    trainer.save_state()


def train_all() -> None:
    for model_id in ALL_MODELS:
        for corpus in ['switchboard', 'maptask']:
            print('*'*110)
            print(f"Training {model_id} on {corpus} corpus")
            print('*'*110)
            
            train_model(
                model_id=model_id,
                block_size=128,
                n_epochs=10,
                train_file=FILES['train'][corpus],
                eval_file=FILES['val'][corpus],
            )


def train_one(
        model_id: str,
        train_file: str,
        val_file: str,
        block_size: int = 128,
        n_epoch: int = 10
) -> None:
    
    if not os.path.exists(train_file):
        raise Exception("Provided train file does not exist.")
    if not os.path.exists(val_file):
        raise Exception("Provided validation file does not exist.")
    
    train_model(
        model_id=model_id,
        block_size=block_size,
        n_epochs=n_epoch,
        train_file=train_file,
        eval_file=val_file
    )
    

if __name__ == '__main__':
    import fire
    fire.Fire({
        'all': train_all,
        'one': train_one
    })