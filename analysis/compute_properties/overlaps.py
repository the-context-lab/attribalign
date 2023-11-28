"""
Compute surface-level metrics for the generated responses.
    - lexical overlap with context (VO - vocabulary overlap)
    - structural overlap with context (SO - structural overlap)
    - construction overlap with context (CO - construction overlap)

Usage:

    python analysis/compute_properties/overlaps.py \
        --output_file data/samples_ol.tsv \
        --dialogues_dir data/_tmp_dialign_fo7sbua2 \
        --corpus switchboard

"""
import os
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from helpers import (
    add_space_before_non_alphanum,
    rm_non_alphanum_from_str,
    pipe,
)



def compute_construct_overlap(
        lexicon: pd.DataFrame,
        conversation: pd.DataFrame,
        n_turns: int = 10,
) -> List[float]:
    """Compute construction overlap (CO) with context."""
    target = conversation.iloc[-1]['utterance']  # target is last utterance
    target = rm_non_alphanum_from_str(target).strip()
    overlap = [0.0 for _ in range(n_turns-1)]
    for idx in range(n_turns-1):
        _constructs = lexicon[lexicon['Turns'].str.contains(str(idx))]
        # if no CO with this utterance, skip
        if len(_constructs.index) == 0:
            continue
        constructs = ' '.join(_constructs['Surface Form'].tolist())
        constructs = rm_non_alphanum_from_str(constructs).strip()
        overlap_score = len(constructs.split()) / len(target.split())
        overlap[idx] = overlap_score
    return overlap


def calculate_overlap_metrics(
        input_file: str,
        output_file: str,
        dialogues_dir: str,
        lexica_dir: str,
        n_turns: int = 10,
        n_model_responses: int = 5,
        corpus: str = None,
        verbose: bool = False,
        leader_follower_sep: bool = False,
) -> None:
    """
    Calculate lexical, structural and constrution overlap metrics (VO, SO, SO).

    :param input_file : str
        path to input file (sample-level dataframe tsv)
    :param output_file : str
        path to output file (samples-level dataframe tsv)
    :param dialogues_dir : str
        path to directory with conversations (dialign input files)
    :param lexica_dir : str
        path to directory with lexica (dialign output files)
    :param n_turns : int 
        number of turns in each conversation
    :param corpus : str
        corpus name (e.g. switchboard or maptask)
    :param verbose : bool
        print results to stdout
    :param leader_follower_sep : bool
        if True, compute overlap metrics separately for leader and follower
        default: False
    """
    # determine corpus name
    if corpus is None:  # try to infer from conversations dir
        if 'switchboard' in dialogues_dir.lower():
            corpus = 'switchboard'
        elif 'maptask' in dialogues_dir.lower():
            corpus = 'maptask'
        else:
            raise ValueError('Could not infer corpus from conversations dir.')

    # get conversations and lexica
    conversations = [fn for fn in os.listdir(dialogues_dir) if fn.endswith('.tsv')]
    lexica = [fn for fn in os.listdir(lexica_dir) if fn.endswith('_tsv-lexicon.tsv')]

    # init overlap arrays
    overlap_all = [np.array([]) for _ in range(n_turns-1)]
    overlap_all_jaccard = [np.array([]) for _ in range(n_turns-1)]
    overlap_all_new = [np.array([]) for _ in range(n_turns-1)]
    overlap_structural_all = [np.array([]) for _ in range(n_turns-1)]
    overlap_construction_all = [np.array([]) for _ in range(n_turns-1)]

    if leader_follower_sep and corpus == 'maptask':
        # init overlap arrays for partner
        # The final utterance (response) is either performed by
        # speaker A or B. In case of Map Task, we are interested
        # in breaking the overlap analysis into the two speakers,
        # as they have different roles (leader & follower).
        # Speakers share the same role in SwitchBoard, so we can
        # just use the same arrays as above.
        overlap_all_partner = [np.array([]) for _ in range(n_turns-1)]
        overlap_all_jaccard_partner = [np.array([]) for _ in range(n_turns-1)]
        overlap_all_new_partner = [np.array([]) for _ in range(n_turns-1)]
        overlap_structural_all_partner = [np.array([]) for _ in range(n_turns-1)]
        overlap_construction_all_partner = [np.array([]) for _ in range(n_turns-1)]

    # read conversation dataframe
    read_dialogue = lambda x: pd.read_csv(
        os.path.join(dialogues_dir, x),
        sep='\t',
        header=None,
    )
    def read_lexion(filename: str) -> pd.DataFrame:
        df_lexicon = pd.read_csv(
            os.path.join(lexica_dir, filename),
            sep='\t',
            header=0,
        )
        df_lexicon = df_lexicon[df_lexicon['Turns'].str.contains(str(n_turns-1))]
        return df_lexicon

    # iterate over conversations
    sample_idxs = []
    for conv_fname in tqdm(conversations, desc='Computing overlaps...'):

        # load conversation
        sample_id, generation_id = conv_fname.split('_')[0], conv_fname.split('_')[1].replace('.tsv', '')
        sample_idxs.append(int(sample_id))
        try:
            df = read_dialogue(conv_fname)
        except UnicodeDecodeError as e:
            # fix encoding issues
            df = pd.read_csv(
                os.path.join(dialogues_dir, conv_fname),
                sep='\t',
                header=None,
                encoding='latin1',
            )

        df.columns = ['label', 'utterance']  # structure: speaker label and utterance text
        assert len(df) == n_turns

        # response is last row, context is rows (utterances) before response
        response_og = df.iloc[-1]['utterance']
        context_og = df.iloc[:-1]['utterance'].tolist()

        # clean response and context
        try:
            response = add_space_before_non_alphanum(response_og)
            context = [add_space_before_non_alphanum(c) for c in context_og]
        except Exception as e:
            continue
        response = response.split()
        response = [w.strip() for w in response]
        context = [c.split() for c in context]
        context = [[w.strip() for w in c] for c in context]

        # lexical overlap
        response_set = set(response)
        context_set = [set(c) for c in context]
        len_response_set = len(response_set)
        if len_response_set == 0:
            continue
        lexical_overlap = [len(response_set.intersection(c)) / len_response_set for c in context_set]
        lexical_overlap_jacc = [
            len(response_set.intersection(c)) / len(response_set.union(c))
            for c in context_set
        ]

        # new overlap metric
        lexical_overlap_metric = []
        for utt in context:
            common_words = response_set.intersection(utt)
            if len(common_words) == 0:
                lexical_overlap_metric.append(0.0)
                continue
            common_words = list(common_words)
            # count how many times each word appears in the response
            cnt = sum([response.count(w) for w in common_words])
            lexical_overlap_metric.append(cnt / len_response_set)

        # structural overlap
        # pos tag response
        pos_response = nltk.pos_tag(nltk.word_tokenize(response_og))
        pos_response = [p[1] for p in pos_response]
        pos_context = [nltk.pos_tag(nltk.word_tokenize(c)) for c in context_og]
        pos_context = [[p[1] for p in p] for p in pos_context]
        pos_response_set = set(pos_response)
        pos_context_set = [set(c) for c in pos_context]
        structural_overlap = [
            len(pos_response_set.intersection(c)) / len(pos_response_set)
            for c in pos_context_set
        ]

        # construction overlap
        # load lexicon
        lexicon = read_lexion(conv_fname.replace('.tsv', '_tsv-lexicon.tsv'))
        # compute construction overlap
        construction_overlap = compute_construct_overlap(
            lexicon=lexicon,
            conversation=df,
            n_turns=n_turns,
        )

        def append_scores_to_results():
            for i, o in enumerate(lexical_overlap):
                overlap_all[i] = np.append(overlap_all[i], \
                    {'sample_id': sample_id, 'generation_id': generation_id, 'value': o})
            for i, o in enumerate(lexical_overlap_jacc):
                overlap_all_jaccard[i] = np.append(overlap_all_jaccard[i], \
                    {'sample_id': sample_id, 'generation_id': generation_id, 'value': o})
            for i, o in enumerate(lexical_overlap_metric):
                overlap_all_new[i] = np.append(overlap_all_new[i], \
                    {'sample_id': sample_id, 'generation_id': generation_id, 'value': o})
            for i, o in enumerate(structural_overlap):
                overlap_structural_all[i] = np.append(overlap_structural_all[i], \
                    {'sample_id': sample_id, 'generation_id': generation_id, 'value': o})
            for i, o in enumerate(construction_overlap):
                overlap_construction_all[i] = np.append(overlap_construction_all[i], \
                    {'sample_id': sample_id, 'generation_id': generation_id, 'value': o})
                
        def append_scores_to_partner_results():
            for i, o in enumerate(lexical_overlap):
                overlap_all_partner[i] = np.append(overlap_all_partner[i], \
                    {'sample_id': sample_id, 'generation_id': generation_id, 'value': o})
            for i, o in enumerate(lexical_overlap_jacc):
                overlap_all_jaccard_partner[i] = np.append(overlap_all_jaccard_partner[i], \
                    {'sample_id': sample_id, 'generation_id': generation_id, 'value': o})
            for i, o in enumerate(lexical_overlap_metric):
                overlap_all_new_partner[i] = np.append(overlap_all_new_partner[i], \
                    {'sample_id': sample_id, 'generation_id': generation_id, 'value': o})
            for i, o in enumerate(structural_overlap):
                overlap_structural_all_partner[i] = np.append(overlap_structural_all_partner[i], \
                    {'sample_id': sample_id, 'generation_id': generation_id, 'value': o})
            for i, o in enumerate(construction_overlap):
                overlap_construction_all_partner[i] = np.append(overlap_construction_all_partner[i], \
                    {'sample_id': sample_id, 'generation_id': generation_id, 'value': o})

        if leader_follower_sep:
            # append the computed scores to final result lists
            if corpus == 'switchboard' or (corpus == 'maptask' and sample_id % 2 == 0):
                # if the corpus is SwitchBoard, or the corpus is Map Task but the conversation's
                # sample ID is even, we are appending the computed scores to the base results arrays
                append_scores_to_results()
            elif corpus == 'maptask' and (sample_id % 2 == 1 or sample_id == 1):
                # corpus is Map Task and conversation's sample ID is odd
                append_scores_to_partner_results()
        else:
            append_scores_to_results()

    if verbose:
        if corpus != 'maptask':
            print('lexical overlap with context')
            for i, o in enumerate(overlap_all[::-1]):
                m = np.mean(o)
                print(f'dr{i}: {m:.3f} +- {np.std(o):.3f} {"-" * round((m * 100) / 3) + "o"}')
            print('structural overlap with context')
            for i, o in enumerate(overlap_structural_all[::-1]):
                m = np.mean(o)
                print(f'dr{i}: {m:.3f} +- {np.std(o):.3f} {"-" * round((m * 100) / 3) + "o"}')

        elif corpus == 'maptask':
            print('(FOLLOWER) lexical overlap with context')
            for i, o in enumerate(overlap_all[::-1]):
                m = np.mean(o)
                print(f'dr{i}: {m:.3f} +- {np.std(o):.3f} {"-" * round((m * 100) / 3) + "o"}')
            print('(LEADER) lexical overlap with context')
            for i, o in enumerate(overlap_all_partner[::-1]):
                m = np.mean(o)
                print(f'dr{i}: {m:.3f} +- {np.std(o):.3f} {"-" * round((m * 100) / 3) + "o"}')
            print('(FOLLOWER) structural overlap with context')
            for i, o in enumerate(overlap_structural_all[::-1]):
                m = np.mean(o)
                print(f'dr{i}: {m:.3f} +- {np.std(o):.3f} {"-" * round((m * 100) / 3) + "o"}')
            print('(LEADER) structural overlap with context')
            for i, o in enumerate(overlap_structural_all_partner[::-1]):
                m = np.mean(o)
                print(f'dr{i}: {m:.3f} +- {np.std(o):.3f} {"-" * round((m * 100) / 3) + "o"}')

    # collect results
    results = dict()
    results['overlap_all'] = overlap_all[::-1]
    results['overlap_all_jaccard'] = overlap_all_jaccard[::-1]
    results['overlap_all_new'] = overlap_all_new[::-1]
    results['overlap_structural_all'] = overlap_structural_all[::-1]
    results['overlap_construction_all'] = overlap_construction_all[::-1]
    if leader_follower_sep and corpus == 'maptask':
        results['overlap_all_leader'] = overlap_all_partner[::-1]
        results['overlap_all_jaccard_leader'] = overlap_all_jaccard_partner[::-1]
        results['overlap_all_new_leader'] = overlap_all_new_partner[::-1]
        results['overlap_structural_all_leader'] = overlap_structural_all_partner[::-1]
        results['overlap_construction_all_leader'] = overlap_construction_all_partner[::-1]

    # integrate results into sample-level dataframe
    samples = pd.read_csv(input_file, sep='\t', header=0, index_col='sample_index')
    cols_before = samples.columns.tolist()
    samples[f"vo_human"] = ''
    samples[f"co_human"] = ''
    if leader_follower_sep and corpus == 'maptask':
        samples[f"vo_leader_human"] = ''
        samples[f"co_leader_human"] = ''
    for response_index in range(n_model_responses):
        samples[f"vo_r{response_index}"] = ''
        samples[f"co_r{response_index}"] = ''
        if leader_follower_sep and corpus == 'maptask':
            samples[f"vo_leader_r{response_index}"] = ''
            samples[f"co_leader_r{response_index}"] = ''
    cols_after = samples.columns.tolist()
    cols_added = [c for c in cols_after if c not in cols_before]

    # save results
    vo_version = 'overlap_all'
    vo = results[vo_version]
    co = results['overlap_construction_all']
    if leader_follower_sep and corpus == 'maptask':
        vo_leader = results[vo_version]
        co_leader = results['overlap_construction_all_leader']
    for sample_idx in sample_idxs:
        ot_strs = ['vo', 'co', 'vo_leader', 'co_leader'] \
            if leader_follower_sep and corpus == 'maptask' else ['vo', 'co']
        ots = [vo, co, vo_leader, co_leader] \
            if leader_follower_sep and corpus == 'maptask' else [vo, co]
        for ot_str, ot in zip(ot_strs, ots):
            if ot is None:
                continue
            o_scores = [
                [item for item in ot[utterance_idx] if str(item['sample_id']) == str(sample_idx)]
                for utterance_idx in range(n_turns-1)
            ]
            for response_index in range(n_model_responses+1):
                scores = [
                    item['value']
                    for utterance_idx in range(n_turns-1)
                    for item in o_scores[utterance_idx]
                    if str(item['generation_id']) == str(response_index)
                ]
                # idx 0 is for human response
                if response_index == 0:
                    samples.loc[sample_idx, f"{ot_str}_human"] = pipe(scores)
                else:
                    samples.loc[sample_idx, f"{ot_str}_r{response_index-1}"] = pipe(scores)

    # save to file
    samples.to_csv(
        output_file, 
        sep='\t', 
        header=True, 
        index=True, 
        index_label='sample_index'
    )

    # done    
    print('DONE!\nSaved to', output_file)
    print(f"Columns added ({len(cols_added)}):")
    print('\n'.join([f" - {c}" for c in cols_added]))


if __name__ == '__main__':
    import fire
    fire.Fire(calculate_overlap_metrics)
