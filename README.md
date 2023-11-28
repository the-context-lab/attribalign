<h1><p align="center">
Attribution & Alignment
</p></h1>

<h2><p align="center">
Effects of Local Context Repetition on<br>Utterance Production and Comprehension in Dialogue
</p></h2>

<strong><p align="center">
Aron Molnar*, Jaap Jumelet^, Mario Giulianelli^, Arabella Sinclair*
</p></strong>

<p align="center"><strong>*</strong> Department of Computing Science, University of Aberdeen <br/> <strong>^</strong> Institute for Logic, Language and Computation, University of Amsterdam</p>


---

🥳 _**The paper will be presented at [CoNLL 2023](https://www.conll.org/2023), co-located with [EMNLP](https://2023.emnlp.org/) in Singapore!**_ 🥳

**[📝 <ins>Paper PDF on arXiv</ins>](https://arxiv.org/abs/2311.13061)**

---

Table of Contents:

- [Abstract](#abstract)
- [Citing](#citing)
- [Contact](#contact)
- [Usage](#usage)
  - [Experiment Pipeline](#experiment-pipeline)
  - [Test Run](#test-run)
  - [Adapting to Other Corpora](#adapting-to-other-corpora)
  - [Evaluating Other Metrics](#evaluating-other-metrics)
- [Repository Structure](#repository-structure)
- [License](#license)


---


# Abstract

Language models are often used as the backbone of modern dialogue systems. These models are pre-trained on large amounts of written _fluent_ language.
Repetition is typically penalised when evaluating language model generations. However, it is a key component of dialogue.
Humans use _local_ and _partner specific_ repetitions; these are preferred by human users and lead to more successful communication in dialogue.
In this study, we evaluate (a) **whether language models produce human-like levels of repetition in dialogue**, and (b) **what are the processing mechanisms related to lexical re-use they use during comprehension**.
We believe that such joint analysis of model production and comprehension behaviour can inform the development of cognitively inspired dialogue generation systems.


# Citing

Please use the following format to cite this work.

```latex
Coming Soon
```


# Contact

* [Aron Molnar](https://arotte.github.io/)
* [Jaap Jumelet](https://jumelet.ai/)
* [Mario Giulianelli](https://glnmario.github.io/)
* [Arabella Sinclair](https://j-anie.github.io/index.html) - [University of Aberdeen](https://www.abdn.ac.uk/people/arabella.sinclair)

🔬 **Find out about other work going on at [The Context Lab](https://the-context-lab.github.io/).**


# Usage


## Experiment Pipeline

* **Data** is split for train vs. text purposes and pre-processed to our format of a 10 utterance sample (see [`prepare_corpora.py`](prepare/prepare_corpora.py), [`model_train/`](data/model_train/), [`samples.tsv`](data/samples.tsv)).

* **Trained models** that we use in our experiments are available [here](https://huggingface.co/Arotte). If training your own, the training script is [`train_models.py`](prepare/train_models.py).

* **Generation:** Models are then used to generate utterances directly in the generation scripts [here](generate/generate_and_attribute.py) using the contexts from the test samples.

* **Computing sample properties:** All analysis `.py` files augment or label the samples in [`samples.tsv`](data/samples.tsv) file, which can then be used to analyse the human or model produced samples.

* **Analysis:**
  Once all properties have been extracted, the analysis is conducted at the turn-level so that local factors can be explored and compared between human- vs. model-produced utterances.
  * [`combine_properties_for_turn_level_analysis.ipynb`](analysis/scripts/combine_properties_for_turn_level_analysis.ipynb) contains a script to create turn-level dataframes containing all the repetition and attribution properties computed.
  * [`imspect_samples.ipynb`](analysis/scripts/imspect_samples.ipynb) allows samples to be easily inspected.


## Test Run

Follow the steps described below to run the whole experiment pipeline on a very small sub-set of our data ([`samples_mini.tsv`](data/samples_mini.tsv)).

> _Note:_ Running the pipeline with all of the data might take hours or days depending on the hardware configuration of your system.

> 🚀 If you are on Windows, the [**`run_test_pipeline.ps1`**](run_test_pipeline.ps1) PowerShell script will execute all the steps described below. A similar script for UNIX-like systems is coming soon.


1. **Prep (a):** Download and prepare Switchboard and Map Task.

    ```sh
    # download data from github
    python prepare/prepare_corpora.py download

    # create samples.tsv
    python prepare/prepare_corpora.py prepare --context_length 10
    ```

2. **Prep (b):** Create a sub-set for testing.

    ```sh
    python prepare/mini_samples.py
    ```

3. **Generate:** Generate model responses and extract attribution scores (with GPT-2 on Switchboard for testing).

    ```sh
    python generate/generate_and_attribute.py full_attribution \
      --corpus switchboard \
      --model_id gpt2 \
      --input_file data/samples_mini.tsv \
      --output_file data/samples_mini_gpt2.tsv
    ```
    <!-- python generate/generate_and_attribute.py full_attribution --corpus switchboard --model_id gpt2 --input_file data/samples_mini.tsv --output_file data/samples_mini_gpt2.tsv -->

    Then extract attribution scores for the human responses (human response comprehension).

    ```sh
    python generate/generate_and_attribute.py full_attribution \
      --corpus switchboard \
      --model_id gpt2 \
      --style comprehend \
      --input_file data/samples_mini_gpt2.tsv \
      --output_file data/samples_mini_gpt2.tsv
    ```
    <!-- python generate/generate_and_attribute.py full_attribution --corpus switchboard --model_id gpt2 --style comprehend --input_file data/samples_mini_gpt2.tsv --output_file data/samples_mini_gpt2.tsv -->

4. **Quality:** Compute generation quality metrics.

    ```sh
    python analysis/compute_properties/generation_quality.py run \
      --input_file data/samples_mini_gpt2.tsv \
      --output_file data/samples_mini_gpt2_genq.tsv \
      --corpus switchboard \
      --test_mode
    ```
    <!-- python analysis/compute_properties/generation_quality.py run --input_file data/samples_mini_gpt2.tsv --output_file data/samples_mini_gpt2_genq.tsv --corpus switchboard --test_mode -->

5. **Constructions:** Extract constructions from responses.

    ```sh
    python analysis/compute_properties/constructions.py \
      --input_file data/samples_mini_gpt2_genq.tsv \
      --output_file data/samples_mini_gpt2_genq_constr.tsv \
      --working_dir data/_tmp_dialign/ \
      --delete_working_dir False
    ```
    <!-- python analysis/compute_properties/constructions.py --input_file data/samples_mini_gpt2_genq.tsv --output_file data/samples_mini_gpt2_genq_constr.tsv --working_dir data/_tmp_dialign/ --delete_working_dir False -->

6. **Surprisal:** Compute surprisal of model- and human-produced responses.

    ```sh
    python analysis/compute_properties/surprisal.py compute_surprisal \
      --input_file data/samples_mini_gpt2_genq_constr.tsv \
      --output_file data/samples_mini_gpt2_genq_constr_ppl.tsv \
      --test_mode
    ```
    <!-- python analysis/compute_properties/surprisal.py compute_surprisal --input_file data/samples_mini_gpt2_genq_constr.tsv --output_file data/samples_mini_gpt2_genq_constr_ppl.tsv --test_mode -->

7. **Overlaps:** Compute lexical, structural and construction overlap scores.

    ```sh
    python analysis/compute_properties/overlaps.py \
      --input_file data/samples_mini_gpt2_genq_constr_ppl.tsv \
      --output_file data/samples_mini_gpt2_genq_constr_ppl_ol.tsv \
      --dialogues_dir data/_tmp_dialign/dialogues/switchboard/ \
      --lexica_dir data/_tmp_dialign/lexica/switchboard \
      --corpus switchboard
    ```
    <!-- python analysis/compute_properties/overlaps.py --input_file data/samples_mini_gpt2_genq_constr_ppl.tsv --output_file data/samples_mini_gpt2_genq_constr_ppl_ol.tsv --dialogues_dir data/_tmp_dialign/dialogues/switchboard/ --lexica_dir data/_tmp_dialign/lexica/switchboard --corpus switchboard -->

8. **PMI:** Compute pointwise mutual information (PMI) of extracted constructions.

    ```sh
    python analysis/compute_properties/pmi.py \
      --input_file data/samples_mini_gpt2_genq_constr_ppl_ol.tsv \
      --output_file data/samples_mini_gpt2_genq_constr_ppl_ol_pmi.tsv \
      --dialign_output_dir data/_tmp_dialign/lexica/switchboard/ \
      --dialign_input_dir data/_tmp_dialign/dialogues/switchboard/ \
    ```
    <!-- python analysis/compute_properties/pmi.py --input_file data/samples_mini_gpt2_genq_constr_ppl_ol.tsv --output_file data/samples_mini_gpt2_genq_constr_ppl_ol_pmi.tsv --dialign_output_dir data/_tmp_dialign/lexica/switchboard/ --dialign_input_dir data/_tmp_dialign/dialogues/switchboard/ --clean_working_dir False -->

9. **Cleanup:** Clean up the temporary working folders created during the augmentation process.

    ```sh
    python analysis/compute_properties/cleanup.py
    ```



## Adapting to Other Corpora

 * Other corpora can be processed an evaluated using this pipeline, following the procedure within the [`prepare`](prepare/) folder.
 * It is possible to vary parameters for the processing and attribution script (e.g. if more or less than 10 utterances in a sample or similar).


## Evaluating Other Metrics

* Other evaluation metrics and properties can be added to the final turn-level `tsv` to investigate other factors which may contribute to repetition.


# Repository Structure

This repository is structured as follows.

<ins>**Data**</ins>

The **[`data/`](data/)** folder contains all of our prepared and generated data used and created during evaluations.

* The **[`model_train/`](data/model_train/)** folder contains Switchboard and Map Task data prepared for LLM training. Contents of this folder can be re-generated with [`prepare_corpora.py`](prepare/prepare_corpora.py).
* The **[`samples.tsv`](data/samples.tsv)** file contains prepared dialogue excerpts (samples). This file is used for all of our evaluations. This file can be re-generated with [`prepare_corpora.py`](prepare/prepare_corpora.py).

<ins>**Prepare**</ins>

The **[`prepare/`](prepare/)** folder contains scripts that prepare data and models for analysis.

* **[`prepare_corpora.py`](prepare/prepare_corpora.py)** prepares Switchboard and Map Task for model training and analysis.
* **[`train_models.py`](prepare/train_models.py)** trains our selected suite of LLMs with [`transformers.Trainer`](https://huggingface.co/docs/transformers/main_classes/trainer).
* **[`mini_samples.py`](prepare/mini_samples.py)** assembles a sub-set of the larger [`samples.tsv`](data/samples.tsv) for testing purposes.

<ins>**Generate**</ins>

The **[`generate/`](generate/)** folder contains scripts that generate the data we use in our evaluations.

* **[`generate_and_attribute.py`](generate/generate_and_attribute.py)** generates responses to dialogue excerpts (samples) and extracts raw attribution scores. The script can also extract attribution scores while comprehending human-produced responses to dialogue excerpts.
* **[`attribution_aggregation.py`](generate/attribution_aggregation.py)** implements our attribution aggregation algorithm. It takes in raw attribution matrices extracted with [`Inseq`](https://github.com/inseq-team/inseq/) during generation and transforms them to utterance-level attribution scores.

<ins>**Analysis**</ins>

The **[`analysis/`](analysis/)** folder contains scripts and notebooks that analyse, evaluate and enrich (e.g. with construction annotations) the prepared and generated data.

* **[`compute_properties/`](analysis/compute_properties/)** contains scripts to extract the properties we examine and use in our analysis and evaluation

  * **[`helpers.py`](analysis/compute_properties/helpers.py)** contains helper functions for string operations.
  * **[`surprisal.py`](analysis/compute_properties/surprisal.py)** computes perplexities of LLM-generated and human responses.
  * **[`generation_quality.py`](analysis/compute_properties/generation_quality.py)** computes metrics ([BERTScore](https://arxiv.org/abs/1904.09675), [BLEU](https://aclanthology.org/P02-1040.pdf) and [MAUVE](https://krishnap25.github.io/mauve/)) with which we aim to gauge the relative quality of the LLM-generated responses.
  * **[`constructions.py`](analysis/compute_properties/constructions.py)** extracts constructions (i.e. shared word sequences) from LLM-generated and human responses with [`Dialign`](https://github.com/GuillaumeDD/dialign).
  * **[`overlaps.py`](analysis/compute_properties/overlaps.py)** computes lexical, structural and construction overlap scores between responses and each utterance of the context (dialogue excerpt).
  * **[`pmi.py`](analysis/compute_properties/pmi.py)** computes the pointwise mutual information (PMI) of extracted constructions Implementation adopted from [here](https://github.com/dmg-illc/uid-dialogue/tree/main).

* **[`scripts/`](analysis/scripts/)** contains notebooks that combine the data and extracted properties for analysis.


# License

Creative Commons.
