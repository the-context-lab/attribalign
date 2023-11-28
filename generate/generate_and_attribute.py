"""
Generate responses to dialogue excerpts (sample prompts) and
extract attributions while doing so.

Example usage:

[test if generation works]
    python generate/generate_and_attribute.py test_generation
    python generate/generate_and_attribute.py test_generation --model_id facebook/opt-125m

[test if attribution works]
    python generate/generate_and_attribute.py test_attribution --model_id facebook/opt-125m

[full generation+attribution on switchboard with opt]
    python generate/generate_and_attribute.py full_attribution --corpus switchboard \
        --model_id facebook/opt-125m

[full generation+attribution on maptask with dgpt]
    python generate/generate_and_attribute.py full_attribution --corpus maptask \
        --model_id microsoft/DialoGPT-small
"""
import os
import re
import time
import csv
import torch
import warnings
import shortuuid
import inseq
import fire
import pandas as pd
from typing import Union, List, Dict, Any

from inseq import FeatureAttributionOutput
from tqdm import tqdm
from pathlib import Path
from transformers import (
    StoppingCriteria,
    StoppingCriteriaList,
)
from attribution_aggregation import get_utterance_attribs

warnings.filterwarnings("ignore")

# make sure script can be run from anywhere
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')


# =====================================================================================


class StoppingCriteriaSub(StoppingCriteria):
    """Stop generation when one of the stop tokens is reached."""

    def __init__(self, stops: list = []):
        super().__init__()
        self.stops = stops

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ):
        c_id = input_ids[0][-1:].item()
        for stop in self.stops:
            if c_id == stop:
                return True
        return False


def safe_model_name(model_id):
    """Remove non-safe characters from a model id."""
    return re.sub(r'\W+', '-', model_id)


def pipe(in_list: list):
    """Convert a list of a pipe-separated string."""
    return '|'.join([str(i) for i in in_list])


# =====================================================================================


class AttributionExtractor:
    """
    Generate responses to sample prompts and extracts
    attribution matrices while doing so.
    """

    def __init__(
            self,
            style: str,
            algorithm: str,
            model_id: str,
            corpus: str,
            from_idx: int = 0,
            to_idx: int = -1,
            samples_path: str = Path(DATA_DIR) / 'corpora' / 'samples.tsv',
            out_samples_path: str = None,
            save_attribution_matrices: bool = False,
            save_every: int = 10,
            save_matrices_to: str = None,
            generation_max_new_tokens: int = 64,
            generation_do_sample: bool = True,
            generation_num_beams: int = 1,
            generation_num_return_sequences: int = 5,
            context_length: int = 10,
            logfile: str = None,
            eos_token: str = '<|endoftext|>'
    ):
        self.eos_token = eos_token
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f"Using device: {self.device}")

        # check arguments
        supported_algorithms = ['input_x_gradient', 'deeplift']
        supported_corpora = ['switchboard', 'maptask']
        supported_styles = ['generate', 'comprehend']
        if algorithm not in supported_algorithms:
            raise Exception(f"Unsupported algorithm: {algorithm}\nSupported algorithms: {supported_algorithms}")
        if corpus not in supported_corpora:
            raise Exception(f"Unsupported corpus: {corpus}\nSupported corpora: {supported_corpora}")
        if style not in supported_styles:
            raise Exception(f"Unsupported style: {style}\nSupported styles: {supported_styles}")
        try:
            from transformers import AutoTokenizer
            _ = AutoTokenizer.from_pretrained(model_id)
        except Exception as e:
            raise Exception(f"Could not load model: {e}")
        if not os.path.exists(samples_path):
            raise Exception(f"Samples file {samples_path} does not exist.")

        # create safe model name
        self.model_name_safe = ''
        if '/' in model_id or '\\' in model_id:
            last_part = Path(model_id).parts[-1]
            self.model_name_safe = safe_model_name(last_part)
        else:
            self.model_name_safe = safe_model_name(model_id)

        # load samples
        if out_samples_path is None:
            out_samples_path = str(samples_path).replace('.tsv', '') + '_' + self.model_name_safe + '.tsv'
        _samples = pd.read_csv(
            samples_path, 
            sep='\t', 
            header=0, 
            index_col='sample_index'
            # quoting=csv.QUOTE_NONE, 
        )

        # add new columns to samples
        new_cols = ['comprehend_attribs', 'comprehend_attribs_tag']
        for gen_idx in range(generation_num_return_sequences):
            new_cols += [
                f'model_response_r{gen_idx}',
                f'attribs_r{gen_idx}',
                f'attribs_tag_r{gen_idx}',
            ]
        print(f"\nAdding the following new columns to the dataframe:")
        print('\n'.join([f"  - {c}" for c in new_cols]))
        
        added_cols = []
        for new_col in new_cols:
            if new_col not in _samples.columns.to_list():
                _samples[new_col] = ''
                added_cols.append(new_col)
        _samples.to_csv(out_samples_path, sep='\t', header=True, index=True, index_label='sample_index')
        self.samples = pd.read_csv(
            out_samples_path, 
            sep='\t', 
            header=0, 
            index_col='sample_index',
            # quoting=csv.QUOTE_NONE,
        )
        self.samples = self.samples[self.samples['corpus'] == corpus]
        print(f"Added {len(added_cols)} new columns to the dataframe.")
        print('These are:')
        print('\n'.join([f"  - {c}" for c in added_cols]))

        # check indices
        if from_idx < 0:
            from_idx = 0
        if to_idx > len(self.samples) or to_idx == -1:
            to_idx = len(self.samples)
        if from_idx >= to_idx:
            raise Exception(f"from_idx ({from_idx}) >= to_idx ({to_idx}).")

        # create output attribution matrix directory
        if save_attribution_matrices == True and save_matrices_to is None:
            save_matrices_to = os.path.join(DATA_DIR, 'attributions', self.run_uuid)

        # recovery
        if logfile is not None:
            logfile_filename = os.path.basename(logfile)
            self.run_uuid = logfile_filename.replace('log_', '').replace('.txt', '')
            log = self.read_log_file(logfile)
            self.recovery_file = logfile
        else:
            # this is a new run
            self.run_uuid = shortuuid.uuid()
            log_contents = {  # all the paramters of this run
                'run_uuid': self.run_uuid,
                'style': style,
                'algorithm': algorithm,
                'model_id': model_id,
                'corpus': corpus,
                'from_idx': from_idx,
                'to_idx': to_idx,
                'samples_path': samples_path,
                'out_samples_path': out_samples_path,
                'save_attribution_matrices': save_attribution_matrices,
                'save_every': save_every,
                'save_matrices_to': save_matrices_to,
                'generation_max_new_tokens': generation_max_new_tokens,
                'generation_do_sample': generation_do_sample,
                'generation_num_beams': generation_num_beams,
                'generation_num_return_sequences': generation_num_return_sequences,
                'context_length': context_length,
                'progress_idx': from_idx,
            }
            logfile = os.path.join(DATA_DIR, 'attributions', f"log_{self.run_uuid}.txt")
            self.logfile = logfile
            if not os.path.exists(os.path.dirname(logfile)):
                os.makedirs(os.path.dirname(logfile))
            self.write_log_file(logfile, log_contents)
            self.recovery_file = logfile
        self.n_to_gen = len(self.samples)

        # init vars
        self.generation_config = {
            'max_new_tokens': generation_max_new_tokens,
            'do_sample': generation_do_sample,
            'num_beams': generation_num_beams,
            'num_return_sequences': generation_num_return_sequences,
        }
        self.context_length = context_length
        self.style = style
        self.algorithm = algorithm
        self.model_id = model_id
        self.corpus = corpus
        self.from_idx = from_idx
        self.to_idx = to_idx
        self.samples_path = samples_path
        self.out_samples_path = out_samples_path
        self.save_attribution_matrices = save_attribution_matrices
        self.save_every = save_every
        self.save_matrices_to = save_matrices_to
        self.result_buffer = []
        self.comprehension = style == 'comprehend'

        # init model
        self.is_dialogpt = "dialogpt" in model_id.lower() or "dgpt" in model_id.lower()
        self.model_inseq = inseq.load_model(
            model_id,
            algorithm,
            # kwargs={'device': self.device,}
        )

        # create stopping criteria on newlines
        stop_words = ["\n"]
        stop_words_ids = [
            self.model_inseq.tokenizer(stop_word, add_special_tokens=False)['input_ids']
            for stop_word in stop_words
        ]
        stop_words_ids = [item for sublist in stop_words_ids for item in sublist]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        # print info
        print()
        print(self.get_class_params_as_string())

    def get_class_params_as_string(self):
        info_str = f"AttributionExtractor\n"
        info_str += f"  UUID: {self.run_uuid}\n"
        info_str += f"  style: {self.style}\n"
        info_str += f"  algorithm: {self.algorithm}\n"
        info_str += f"  model_id: {self.model_id}\n"
        info_str += f"  corpus: {self.corpus}\n"
        info_str += f"  from_idx: {self.from_idx}\n"
        info_str += f"  to_idx: {self.to_idx}\n"
        info_str += f"  samples_path: {self.samples_path}\n"
        info_str += f"  out_samples_path: {self.out_samples_path}\n"
        info_str += f"  save_attribution_matrices: {self.save_attribution_matrices}\n"
        info_str += f"  save_every: {self.save_every}\n"
        info_str += f"  save_matrices_to: {self.save_matrices_to}\n"
        info_str += f"  generation:\n"
        info_str += f"    max_new_tokens: {self.generation_config['max_new_tokens']}\n"
        info_str += f"    do_sample: {self.generation_config['do_sample']}\n"
        info_str += f"    num_beams: {self.generation_config['num_beams']}\n"
        info_str += f"    num_return_sequences: {self.generation_config['num_return_sequences']}\n"
        info_str += f"  context_length: {self.context_length}\n"
        return info_str

    def model_generate(self, input_txt: str) -> List[str]:
        with torch.no_grad():
            input_ids = self.model_inseq.tokenizer.encode(input_txt, return_tensors='pt')
            input_ids = input_ids.to(self.device)
            gen_txt = []
            for _ in range(self.generation_config['num_return_sequences']):
                generated_response = self.model_inseq.model.generate(
                    input_ids,
                    pad_token_id=self.model_inseq.tokenizer.eos_token_id,
                    max_new_tokens=self.generation_config['max_new_tokens'],
                    do_sample=self.generation_config['do_sample'],
                    num_beams=self.generation_config['num_beams'],
                    stopping_criteria=self.stopping_criteria,
                )
                decoded_response = self.model_inseq.tokenizer.decode(
                    generated_response[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                gen_txt.append(decoded_response)
            return gen_txt

    def generate_with_attributions(
            self,
            prompt: str,
            human_response: str
    ) -> Dict[str, Union[FeatureAttributionOutput, None, List[Any], List[str]]]:
        """Main attribution function."""

        # add last speaker label to prompt
        prompt = prompt.strip()
        human_response = human_response.strip()
        if human_response.startswith('A:') or human_response.startswith('B:'):
            human_speaker_label = human_response[:2]
            prompt += f'\n{human_speaker_label}'
            human_response = human_response[2:]
        
        # add eos token to prompt
        if self.is_dialogpt:
            prompt = self.reformat_prompt_for_dialogpt(prompt, self.eos_token)
            human_response += self.eos_token

        # try to generate and attribute
        try:
            responses = []

            # GENERATION
            if self.style == 'generate':
                
                # generate responses from prompt
                responses = self.model_generate(prompt)
                
                # check for empty responses
                for i, generated_response in enumerate(responses):
                    generated_text = generated_response.strip().replace(prompt.strip(), '')
                    if len(generated_text.strip()) == 0:
                        print(f"WARNING: Empty response generated for prompt")
                
                # reformat raw responses for dialogpt attribution
                if self.is_dialogpt:
                    genr = []
                    for g in responses:
                        genr.append(self.reformat_prompt_for_dialogpt(g, self.eos_token))
                    responses = genr
                
                # attribute
                attribution = self.model_inseq.attribute(
                    input_texts=[prompt for _ in range(self.generation_config['num_return_sequences'])],
                    generated_texts=responses,
                    show_progress=False,
                    pretty_progress=False,
                    # device=self.device,
                )
                
                # squash along attention head dimension
                for i, _ in enumerate(attribution.sequence_attributions):
                    __att = attribution.sequence_attributions[i].target_attributions
                    __att = __att.nansum(axis=2)
                    attribution.sequence_attributions[i].target_attributions = __att

            # COMPREHENSION
            elif self.style == 'comprehend':
                gen_text = prompt + human_response
                attribution = self.model_inseq.attribute(
                    input_texts=[prompt],
                    generated_texts=[gen_text],
                    show_progress=False,
                    # device=self.device,
                )
                __att = attribution.sequence_attributions[0].target_attributions
                __att = __att.nansum(axis=2)
                attribution.sequence_attributions[0].target_attributions = __att

            # get utterance-level attributions
            utterance_attributions = []
            if not self.comprehension:  # if generation
                for generation_idx in range(self.generation_config['num_return_sequences']):
                    # get utterance-level attributions
                    utterance_att = get_utterance_attribs(
                        attribution=attribution,
                        tokenizer=self.model_inseq.tokenizer,
                        sample_idx=generation_idx,
                        print_info=False,
                        count_newline_scores=False,
                        n_utts=self.context_length,
                        comprehension=False,
                        return_tag_scores=True,  # the function will return speaker label scores
                    )
                    utterance_attributions.append(utterance_att)
            else:  # if comprehension
                utterance_att = get_utterance_attribs(
                    attribution,
                    self.model_inseq.tokenizer,
                    0,
                    print_info=False,
                    count_newline_scores=False,
                    n_utts=self.context_length,
                    comprehension=True,
                    return_tag_scores=True,  # the function will return speaker label scores
                )
                utterance_attributions.append(utterance_att)

            # done
            return {
                'attribution': attribution if self.save_attribution_matrices else None,
                'utterance_attributions': [ua[0] for ua in utterance_attributions],
                'tag_attributions': [ua[1] for ua in utterance_attributions],
                'model_responses': [r.replace(prompt, '') for r in responses] if self.style == 'generate' else None,
            }
        
        except Exception as e:
            print(f'Error while attributing: {e}')
            return None

    def save_attributions(self):
        """Save attribution matrices to disk."""
        for idx, result in self.result_buffer:
            attrib = result['attribution']
            comp = '_comp' if self.comprehension else ''
            out_fname = os.path.join(
                self.save_matrices_to,
                f'attr_{str(idx)}_{self.model_name_safe}{comp}.json'
            )
            try:
                attrib.save(
                    out_fname,
                    compress=True,  # reduce space
                    ndarray_compact=True,  # reduce space
                    overwrite=True  # overwrite if exists
                )
            except Exception as e:
                print(f'ERROR saving: {e}', '\n> out_fname:', out_fname)
                pass

    def update_samples_dataframe(self):
        """Update output dataframe with new results."""

        # load input dataframe
        samples_og = pd.read_csv(
            self.samples_path,
            sep='\t',
            header=0,
            index_col='sample_index',
            # quoting=csv.QUOTE_NONE,
        )

        # clean model response
        clean_model_response = lambda s: s.replace('\n', '\\n').replace('\t', '\\t').strip()

        # update in-memory and original dataframe
        for df in [self.samples, samples_og]:
            for df_idx, res in self.result_buffer:
                utt_attribs = res['utterance_attributions']
                label_attribs = res['tag_attributions']
                model_responses = res['model_responses']
                if not self.comprehension:
                    for gen_idx in range(self.generation_config['num_return_sequences']):
                        df.at[df_idx, f"model_response_r{gen_idx}"] = \
                            clean_model_response(model_responses[gen_idx])
                        df.at[df_idx, f"attribs_r{gen_idx}"] = \
                            pipe(utt_attribs[gen_idx])
                        df.at[df_idx, f"attribs_tag_r{gen_idx}"] = \
                            pipe(label_attribs[gen_idx])
                else:
                    df.at[df_idx, 'comprehend_attribs'] = pipe(utt_attribs[0])
                    df.at[df_idx, 'comprehend_attribs_tag'] = pipe(label_attribs[0])
            
        # save to disk
        samples_og.to_csv(
            self.out_samples_path, 
            sep='\t', 
            header=True, 
            index=True, 
            index_label='sample_index'
        )

    def attribute(self):
        print('*' * 80)
        print(f"STARTING ATTRIBUTION IN [{self.style.upper()}] MODE")
        print('*' * 80)

        # determine index subset 
        index_subset = self.samples.index.to_list()[self.from_idx:self.to_idx]
        print(f"Attributing {len(index_subset)} samples.")
        print(f"Index subset: {index_subset[:5]}...{index_subset[-5:]}")

        n_generations = 0
        successfully_generated = 0
        start_time = time.time()
        progress_bar = tqdm(
            index_subset,
            desc=f"{self.from_idx}_{str(self.to_idx).replace('-', '_')} {self.model_name_safe}"
        )

        for global_idx in progress_bar:
            row = self.samples.loc[global_idx]
            # get prompt and human response
            prompt = row['prompt']
            human_response = row['human_response']

            # generate response and attribute
            g = self.generate_with_attributions(prompt.replace('\\n', '\n'), human_response)
            
            # bookkeeping
            self.result_buffer.append((global_idx, g))
            n_generations += 1
            if g is not None:
                successfully_generated += 1

            # save attribution matrices and update samples dataframe
            if (n_generations % self.save_every == 0) and successfully_generated > 0:
                time_now = time.time()
                time_elapsed = time_now - start_time
                time_per_generation = time_elapsed / n_generations
                time_remaining = time_per_generation * (self.n_to_gen - n_generations)

                print(f'[{n_generations}/{self.n_to_gen}]')
                print(f'  time per generation: {time_per_generation:.2f} s')
                print(f'  time time_remaining_h: {time_remaining:.2f} h')

                if self.save_attribution_matrices:
                    try:
                        self.save_attributions()
                    except Exception as e:
                        print(f'ERROR SAVING ATTRIBUTIONS: {e}')
                        pass
                try:
                    self.update_samples_dataframe()
                except Exception as e:
                    print(f'ERROR UPDATING SAMPLES DATAFRAME: {e}')
                    raise e

                # clear buffer
                self.result_buffer.clear()

                # write global index to recovery log
                log_contents = self.read_log_file(self.recovery_file)
                log_contents['progress_idx'] = global_idx
                self.write_log_file(self.recovery_file, log_contents)

        if self.save_attribution_matrices:
            self.save_attributions()  # save the last batch of attrubitons
        gen_or_comp = 'generated' if self.style == 'generate' else 'comprehended'
        print(f"\n\nSuccessfully {gen_or_comp.upper()} {successfully_generated} attributions out of {n_generations}.")
        print(f"Attribution took {time.time() - start_time:.2f} seconds.")
        print(f"Results saved to: {self.out_samples_path}")
        print(f"Log file: {self.logfile}")
        print("DONE.")

    @classmethod
    def reformat_prompt_for_dialogpt(cls, prompt: str, eos_token: str) -> str:
        """ Reformat dialogue for DialoGPT by adding the
            end of sequence token to each utterance.
        """
        p = prompt.strip().split('\n')
        if len(p) != 10:
            p = p[:10]
        pf = [u + eos_token for u in p[:-1]] + [p[-1]]
        return '\n'.join(pf)

    @classmethod
    def read_log_file(cls, fp: str):
        content = {}
        with open(fp, 'r') as f:
            # read content into dict
            content = f.readlines()
            content = [line.strip() for line in content]
            content = [line.split('=') for line in content]
            content = {line[0]: line[1] for line in content}
        return content

    @classmethod
    def write_log_file(cls, fp: str, content: dict):
        with open(fp, 'w') as f:
            for k, v in content.items():
                f.write(f'{k}={v}\n')

    @classmethod
    def start_from_log_file(cls, log_file: str):
        """ Start attribution from a log file.
            The log file contains the parameters of the
            previous run.
        """
        print(f"Starting attribution from log file: {log_file}")
        log = cls.read_log_file(log_file)
        # create attribution extractor
        ae = cls(
            style=log['style'],
            algorithm=log['algorithm'],
            model_id=log['model_id'],
            corpus=log['corpus'],
            from_idx=int(log['progress_idx']),
            to_idx=int(log['to_idx']),
            samples_path=log['samples_path'],
            out_samples_path=log['out_samples_path'],
            save_attribution_matrices=log['save_attribution_matrices'],
            save_every=int(log['save_every']),
            save_matrices_to=log['save_matrices_to'],
            generation_max_new_tokens=int(log['generation_max_new_tokens']),
            generation_do_sample=log['generation_do_sample'] == 'True',
            generation_num_beams=int(log['generation_num_beams']),
            generation_num_return_sequences=int(log['generation_num_return_sequences']),
            context_length=int(log['context_length']),
            logfile=log_file,
        )
        # start attribution
        ae.attribute()


# =====================================================================================


def test_model_generation(
    model_id: str = 'gpt2',
    samples_path: str = Path(DATA_DIR) / 'corpora' / 'samples.tsv',
):
    ae = AttributionExtractor(
        style='generate',
        algorithm='deeplift',
        model_id=model_id,
        corpus='maptask',
        from_idx=0,
        to_idx=10,
    )
    samples = pd.read_csv(
        samples_path,
        sep='\t',
        header=0,
        quoting=csv.QUOTE_NONE,
        index_col='sample_index'
    )
    samples = samples.sample(5)
    for idx, row in samples.iterrows():
        rep = '\n'
        if 'dialogpt' in model_id.lower():
            rep = '<|endoftext|>\n'
        if row['human_response'].strip().startswith('A:') or row['human_response'].strip().startswith('B:'):
            samples.at[idx, 'prompt'] += rep + row['human_response'][:2]
        samples.at[idx, 'prompt'] = samples.at[idx, 'prompt'].replace('\\n', rep)
    print('TESTING MODEL GENERATION')
    prompts = samples['prompt'].to_list()
    for prompt in prompts:
        print(f'PROMPT:\n------\n{prompt}\n------')
        gens = ae.model_generate(prompt)
        for i, gen in enumerate(gens):
            gen = gen.replace(prompt.replace('<|endoftext|>', ''), '').replace('\n', ' ')
            print(f"GEN{i}: {gen}")
        print('-' * 80)


def test_attribution(
    model_id: str = 'gpt2'
):
    ae = AttributionExtractor(
        style='generate',
        algorithm='deeplift',
        model_id=model_id,
        corpus='switchboard',
        from_idx=0,
        to_idx=10,
        samples_path=str(Path(DATA_DIR) / 'corpora' / 'samples.tsv'),
        out_samples_path=str(Path(DATA_DIR) / 'corpora' / 'samples_test.tsv'),
    )
    ae.attribute()


def full_attribution(
    corpus: str,
    model_id: str,
    style: str = 'generate',
    input_file: str = None,
    output_file: str = None,
    save_every: int = None,
):
    print(f"FULL ATTRIBUTION: {corpus} {model_id}")
    assert style in ['generate', 'comprehend'], f"Unsupported style: {style}"

    # check if input file exists
    if input_file is None or not os.path.exists(input_file):
        input_file = str(Path(DATA_DIR) / 'corpora' / 'samples.tsv')

    # determine saving frequency
    if save_every is None:
        samples_df = pd.read_csv(input_file, sep='\t', header=0, index_col='sample_index')
        samples_df = samples_df[samples_df['corpus'] == corpus]
        save_every = min(10, len(samples_df))

    ae = AttributionExtractor(
        style=style,
        algorithm='deeplift',
        model_id=model_id,
        corpus=corpus,
        from_idx=0,
        to_idx=-1,
        # output files
        samples_path=input_file,
        out_samples_path=output_file,
        save_attribution_matrices=False,
        save_every=save_every,
        save_matrices_to=None,
        # ancestral sampling
        generation_max_new_tokens=64,
        generation_do_sample=True,
        generation_num_beams=1,
        generation_num_return_sequences=5,
        # misc
        context_length=10,
        logfile=None,
    )
    ae.attribute()


if __name__ == '__main__':
    fire.Fire({
        'full_attribution': full_attribution,
        'test_generation': test_model_generation,
        'test_attribution': test_attribution,
    })
