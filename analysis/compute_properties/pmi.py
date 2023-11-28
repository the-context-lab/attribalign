"""
Compute the pointwise mutual information (PMI) of constructions.

Example usage:

    python analysis/compute_properties/pmi.py \
        --input_file data/samples.tsv \
        --output_file data/samples_pmi.tsv \
        --dialign_output_dir data/_tmp_dialign_qh42rk93/lexica/maptask/ \
        --dialign_input_dir data/_tmp_dialign_qh42rk93/dialogues/maptask/ \
        --clean_working_dir False

Requires the following files:
- input_file: a sample-level dataframe
- dialign_output_dir: the output directory of Dialign
- dialign_input_dir: the input directory of Dialign
"""
import os
import fire
import json
import csv
import shortuuid
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from tqdm.auto import tqdm
from helpers import *


# make sure script can be run from anywhere
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'data')


# ===========================================================================================
# PMI Class
# ===========================================================================================


class DialogueSpecificity:    
    def __init__(self, corpus_counts: dict):
        """
        Dialogue specificity statistics for expressions in a dialogue corpus.
        
        Adopted from: 
            Mario Giulianelli, Arabella Sinclair, Raquel Fernández. 2022. 
            Construction Repetition Reduces Information Rate in Dialogue.
            AACL-IJCNLP 2022.
        URL: 
            https://github.com/dmg-illc/uid-dialogue/blob/main/aacl2022/code/gather_construction_metadata.ipynb
        
        Args
        ----
        corpus_counts : dict
            A dictionary of dictionaries:
            corpus_counts[dialogue][expression] = expression frequency in dialogue 
        """
        self.corpus_counts = corpus_counts
        self.dialogues = list(self.corpus_counts.keys())
        
        # Probability distribution over dialogues is uniform -- P(dial)
        self.p_dial = 1 / len(corpus_counts.keys()) 
        
        # The number of expressions (tokens) in a dialogue -- N_dial
        self.n_tokens_in_dial = {}  # N_dial
        for dial in corpus_counts:
            self.n_tokens_in_dial[dial] = sum(corpus_counts[dial].values())
        
        # The (token) frequency of an expression in the corpus -- N_exp
        self.exp_freq_in_corpus = defaultdict(int)
        
        # The total number of expressions (tokens) in the corpus -- N_corpus
        self.n_exp_tokens = 0  # N_corpus
        
        # The probability of an expression in the corpus -- P(exp)
        self.p_exp = defaultdict(int)
        
        # The probability of an expression in a dialogue -- P(exp|dial)
        self.p_exp_given_dial = defaultdict(int)
        
        for dial in corpus_counts:
            for exp in corpus_counts[dial]:
                self.exp_freq_in_corpus[exp] += corpus_counts[dial][exp]
                self.n_exp_tokens += corpus_counts[dial][exp]
                
                # P(exp|dial) = fr(exp, dial) / N_exp
                # try:
                self.p_exp_given_dial[(exp, dial)] = self.corpus_counts[dial][exp] / self.n_tokens_in_dial[dial]
                # except ZeroDivisionError:
                #     self.p_exp_given_dial[(exp, dial)] = 0
                
                # P(exp) = sum(dial' in corpus) [ P(exp|dial') * P(dial') ]
                self.p_exp[exp] += self.p_exp_given_dial[(exp, dial)] * self.p_dial
        
        # The total number of expression types in the corpus -- E_corpus
        self.n_exp_types = len(self.exp_freq_in_corpus)
                  
        # P(dial|exp) for all expressions in all dialogues
        self.dialogue_posteriors = {}
        for dial in self.corpus_counts:
            self.dialogue_posteriors[dial] = self.get_dialogue_posteriors(dial)
            
        # Specificity [P(exp|dial) - P(exp)] for all expressions in all dialogues
        self.dialogue_specificity = {}
        for dial in self.corpus_counts:
            self.dialogue_specificity[dial] = self.get_dialogue_specificity(dial)  
            
        # PMI(exp, dial) for all expressions in all dialogues
        self.pmi = {}
        for dial in self.corpus_counts:
            self.pmi[dial] = self.get_dialogue_pmi(dial)
            
        # MD(exp, dial) for all expressions in all dialogues
        self.mutual_dependency = {}
        for dial in self.corpus_counts:
            self.mutual_dependency[dial] = self.get_mutual_dependency(dial)
            
        # LFMD(exp, dial) for all expressions in all dialogues
        self.lf_mutual_dependency = {}
        for dial in self.corpus_counts:
            self.lf_mutual_dependency[dial] = self.get_lfmd(dial)
            
    def posterior(self, expression, dialogue):
        """
        Compute P(dialogue|expression) for a given expression in a given dialogue.
        P(dialogue|expression) = P(expression|dialogue) * P(dialogue) / P(expression)
        """ 
        return self.p_exp_given_dial[(expression, dialogue)] * self.p_dial / self.p_exp[expression]
    
    def get_dialogue_posteriors(self, dialogue):
        """
        Compute P(dialogue|expression) for all expressions in a dialogue.
        """
        posteriors = Counter()
        for exp in self.corpus_counts[dialogue]:
            posteriors[exp] = self.posterior(exp, dialogue)
        return posteriors
    
    def get_dialogue_specificity(self, dialogue):
        """
        Compute P(expression|dialogue) - P(expression) for all expressions in a dialogue.
        """
        specificity = Counter()
        for exp in self.corpus_counts[dialogue]:
            specificity[exp] = self.p_exp_given_dial[(exp, dialogue)] - self.p_exp[exp]
        return specificity
    
    def get_dialogue_pmi(self, dialogue):
        """
        Compute pointwise mutual information for all expressions in a dialogue.
        PMI(expression, dialogue) = log[ P(expression|dialogue) / P(expression) ]
        """
        pmi = Counter()
        for exp in self.corpus_counts[dialogue]:
            pmi[exp] = np.log2(self.p_exp_given_dial[(exp, dialogue)] / self.p_exp[exp])
        return pmi
    
    def get_mutual_dependency(self, dialogue):
        """
        Compute mutual dependence MD(expression, dialogue) for all expressions in a dialogue (Thanopoulos et al, 2002).
        MD(expression, dialogue) = log[ P(expression|dialogue)**2 * P(dial) / P(expression) ]
        """
        md = Counter()
        for exp in self.corpus_counts[dialogue]:
            md[exp] = np.log2(((self.p_exp_given_dial[(exp, dialogue)] ** 2) * self.p_dial) / self.p_exp[exp])
        return md
    
    def get_lfmd(self, dialogue):
        """
        Compute log-frequency biased mutual dependence LFMD(expression, dialogue) for all expressions in a dialogue (Thanopoulos et al, 2002).
        LFMD(expression, dialogue) = MD(expression, dialogue + log P(expression, dialogue) 
        """
        lfmd = Counter()
        for exp in self.corpus_counts[dialogue]:
            lfmd[exp] = self.mutual_dependency[dialogue][exp] + np.log2(self.p_exp_given_dial[(exp, dialogue)] * self.p_dial)
        return lfmd
    

# ===========================================================================================
# Compute PMI
# ===========================================================================================


def compute_pmi(
        input_file: str,
        output_file: str,
        dialign_output_dir: str,
        dialign_input_dir: str,
        clean_working_dir: bool = False,
        working_dir: str = None,
        n_model_responses: int = 5,
) -> None:
    """Compute PMI of constructions."""

    # init working directory
    if working_dir is None:
        working_dir = DATA_DIR+'/_tmp_pmi_' + shortuuid.uuid()[:6] + '/'
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)
    print(f'Working directory: {working_dir}')

    # init intermediate files
    fp_pmi = working_dir + "pmi.json"
    fp_contexts = working_dir + "contexts.json"
    fp_cc = working_dir + "counts.json"
    fp_meta = working_dir + "meta.json"

    # create contexts
    n_empty_response = 0
    contexts = defaultdict(dict)
    for f in tqdm(os.listdir(dialign_input_dir), desc='constructing contexts'):
        if f.endswith('.tsv') and not f.startswith('.'):
            filepath = os.path.join(dialign_input_dir, f)
            dial_id = f.replace('.tsv', '')
            contexts[dial_id] = {}
            with open(filepath, 'r') as f:
                for lin_idx, line in enumerate(f):
                    utt = line.strip().split('\t')[1]
                    uttn = normalize(utt).strip()
                    contexts[dial_id][lin_idx] = uttn
            # assert that dialouge has 10 turns
            assert len(contexts[dial_id]) == 10, f'{dial_id} has {len(contexts[dial_id])} turns, not 10'
            # detect if some turns are empty
            if '' in contexts[dial_id].values():
                # if last turn is empty, remove from dialogue from dict
                if contexts[dial_id][9] == '':
                    contexts.pop(dial_id)
                    n_empty_response += 1
    with open(fp_contexts, 'w') as f:
        json.dump(contexts, f)
    print(f'{n_empty_response} dialogues removed due to empty responses')

    # compute counts
    corpus_counts = defaultdict(lambda: {})  # [dialogue][expression]
    constr_meta = defaultdict(lambda: {})  # [dialogue][expression][metadata (e.g., which turns)]
    cnt = []
    for lex_fn in tqdm(os.listdir(dialign_output_dir), desc='computing counts'):
        if not lex_fn.endswith('_tsv-lexicon.tsv') or lex_fn.startswith('.'):
            continue
        d_id = lex_fn.replace('_tsv-lexicon.tsv', '')
        try:
            dialogue = contexts[d_id]
        except KeyError:
            continue
        try:
            lexicon_df = pd.read_csv(
                dialign_output_dir + lex_fn, 
                sep='\t', 
                header=0, 
                quoting=csv.QUOTE_NONE
            )
        except:
            continue
        lexicon_df = lexicon_df[lexicon_df['Turns'].str.contains('9')]
        for _, row in lexicon_df.iterrows():
            constr = row['Surface Form']
            if not isinstance(constr, str):
                continue
            if len(normalize(constr.strip()).strip().split()) < 2:
                continue
            constr_norm = normalize(constr.strip()).strip()
            turns = [int(t.strip()) for t in row['Turns'].split(',')]
            try:
                _ = dialogue[turns[0]]
            except KeyError:
                turns = [str(t) for t in turns]
            _freq = 0
            for turn in turns:
                try:
                    text = dialogue[turn]
                except Exception as e:
                    raise e
                ranges = find_subsequence(constr_norm, text)
                _freq += len(ranges)
                if len(ranges) == 0:
                    raise Exception
            if _freq < row['Freq.']:
                raise Exception
            corpus_counts[d_id][constr_norm] = _freq
            constr_meta[d_id][constr_norm] = turns
    with open(fp_cc, 'w') as f:
        json.dump(corpus_counts, f)
    with open(fp_meta, 'w') as f:
        json.dump(constr_meta, f)

    # compute pmi
    ds = DialogueSpecificity(corpus_counts)
    with open(fp_pmi, 'w') as f:
        json.dump(ds.pmi, f)

    # load input samples and add new columns
    samples = pd.read_csv(input_file, sep='\t', header=0, index_col='sample_index')
    cols_before = samples.columns.tolist()
    samples['pmi_human'] = ''
    samples['pmi_human_constrs'] = ''
    for i in range(n_model_responses):
        samples[f'pmi_r{i}'] = ''
        samples[f'pmi_r{i}_constrs'] = ''
    cols_after = samples.columns.tolist()
    cols_added = [c for c in cols_after if c not in cols_before]

    # integrate pmi into samples dataframe
    for key, val in tqdm(ds.pmi.items(), desc='integrating pmi'):
        sample_idx = int(key.split('_')[0])
        gen_idx = int(key.split('_')[1])
        if gen_idx == 0:
            samples.at[sample_idx, 'pmi_human'] = pipe(list(val.values()))
            samples.at[sample_idx, 'pmi_human_constrs'] = pipe(list(val.keys()))
        else:
            samples.at[sample_idx, f'pmi_r{gen_idx-1}'] = pipe(list(val.values()))
            samples.at[sample_idx, f'pmi_r{gen_idx-1}_constrs'] = pipe(list(val.keys()))

    # save
    samples.to_csv(output_file, sep='\t', index=True, header=True)

    # remove working directory
    if clean_working_dir:
        os.system(f'rm -rf {working_dir}')
    else:
        print(f'Working dir remains: {working_dir}')

    print('Done.')
    print('Saved to:', output_file)
    print(f'Columns added ({len(cols_added)}):')
    print('\n'.join([f"  - {c}" for c in cols_added]))


if __name__ == '__main__':
    fire.Fire(compute_pmi)
