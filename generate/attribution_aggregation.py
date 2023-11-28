"""
Compute utterance-level attribution scores.
We perform a set of tensor transformations on the input attribution matrix
to get a single score for each utterance in the conversation.
"""
import numpy as np
import torch
from typing import List


def tkns_to_str(tokens: List['inseq.TokenWithId']) -> str:
    """Convert a list of `TokenWithId`s to a string."""
    _str = ''
    for t in tokens:
        _str += t.token.replace('Ġ', ' ').replace('Ċ', '\n')
    return _str


def get_utterance_attribs(
        attribution: 'inseq.FeatureAttributionOutput',
        tokenizer: 'transformers.AutoTokenizer',
        sample_idx: int = 0,
        print_info: bool = False,
        count_newline_scores: bool = False,
        n_utts: int = 10,
        comprehension: bool = False,
        return_tag_scores: bool = False,
) -> list:
    """
    Extract attribution scores for each utterance in a dialogue sample.
    Extraction is performed on an inseq `FeatureAttributionOutput` object that is 
    generated during the attribution process.

    Parameters
    ----------
    attribution : inseq.FeatureAttributionOutput
        inseq attribution object
    tokenizer : HF Tokenizer
        HF tokenizer that was used during attribution
    sample_idx : int, optional
        index of sample in attribution object. Defaults to 0 (max: 5)
    print_info : bool, optional
        print info. Defaults to False.
    count_newline_scores : bool, optional
        count attribution scores corresponding to newlines. Defaults to False.
    n_utts : int, optional
        number of utterances in the dialogue. Defaults to 10.
    comprehension : bool, optional
        whether the attribution object was generated during human text comprehension.
        Defaults to False.
    return_tag_scores : bool, optional
        return attribution scores to speaker tags. Defaults to False.
        If True, returns a tuple: (utterance_scores, tag_scores)
    
    Returns
    -------
    list (of floats) [if return_tag_scores=False]
        - list of attribution scores for each utterance in the dialogue
        - can contain negative values
        - size of list = number of utterances in the dialogue (max: 10)
        - list of np.nans in case of error
        - ORDER: [response, utterance 9, utterance 8, ..., utterance 1]
    tuple (of lists) [if return_tag_scores=True]
        - first item of tuple: list of attribution scores for each utterance in the dialogue
        - second item of tuple: list of attribution scores for each speaker tag in the dialogue
    """
    opt_prefix_tkn = '</s>'
    gpt_eos_token = '<|endoftext|>'

    nan_scores = [np.nan for _ in range(n_utts)]  # nan scores to return in case of error
    nl_id = tokenizer.encode('\n', add_special_tokens=False)[0]  # newline token id

    # --------------------------------------
    # load and prepare attribution matrix

    # load and check attribution matrix (am)
    am = attribution.sequence_attributions[sample_idx].target_attributions
    if am is None:
        print('ERROR: attrib matrix is None') if print_info else None
        return nan_scores if not return_tag_scores else (nan_scores, nan_scores)
    if len(am.shape) == 3:
        am = torch.nansum(am, dim=2) # sum along last dimension
    print('attribution matrix loaded:', am.shape) if print_info else None

    # determine generated response
    gen = attribution.info['generated_texts'][sample_idx] \
        .replace(attribution.info['input_texts'][sample_idx], '')
    if comprehension:
        gen = gen.replace(gpt_eos_token, '')
    gen_ids = tokenizer.encode(gen, add_special_tokens=False)

    # check if generated response is empty
    if (
        len(gen.strip()) == 0 or 
        (len(gen_ids) >= 1 and gen_ids[0] == nl_id)
    ):
        print('ERROR: generated response is empty') if print_info else None
        return nan_scores if not return_tag_scores else (nan_scores, nan_scores)

    # only continue with last len(generated_response_token_ids) columns
    am = am[:, -len(gen_ids):]
    n_tokens_in_prompt = am.shape[0] - am.shape[1]

    if print_info:
        print('attrib matrix shape after cutting dim 1:', am.shape)
        print('generated response:', f'`{gen}`')
        print('generated response token ids:', gen_ids)
        print('generated response tokens:', tokenizer.convert_ids_to_tokens(gen_ids))

    # handle newline in generated response
    # remove attributions after a newline token in the generated response
    newline_token_idx = [idx for idx, x in enumerate(gen_ids) if x == nl_id]
    if len(newline_token_idx) > 0: # remove attributions after newline token
        am = am[:n_tokens_in_prompt+newline_token_idx[0], :newline_token_idx[0]]
        gen_ids = gen_ids[:newline_token_idx[0]]
        if print_info:
            print('newlines indexes in response:', newline_token_idx)
            print('newline token id:', nl_id)
            print('attrib matrix shape after removing attribs after newline:', am.shape)
    print('n tokens in prompt:', n_tokens_in_prompt) if print_info else None

    # --------------------------------------
    # transpose & squeeze

    # transpose:
    # - each row contains scores for each generated token
    # - each column contains scores for each input token
    # - a cell tells how much the input token in col x contributed to the generated token in row y
    am_T = torch.transpose(am, 0, 1)

    # squeeze attribution matrix: sum columns into one scalar
    # the resulting vector signifies how much each input token contributed to the generated response
    am_T_squeezed = torch.nansum(am_T, dim=0)

    print('attrib before transpose:', am.shape) if print_info else None
    print('attrib after transpose:', am_T.shape) if print_info else None
    print('attrib after squeeze:', am_T_squeezed.shape) if print_info else None

    # --------------------------------------
    # split into utterance-chunks

    # get tokens of prompt
    prompt_tokens = attribution.sequence_attributions[sample_idx].target[:n_tokens_in_prompt]
    print(attribution.sequence_attributions[sample_idx].target) if print_info else None

    # determine newline indexes in prompt
    newline_idxs_in_prompt = [
        idx + 1
        for idx, x in enumerate(prompt_tokens)
        if x.id == nl_id
    ]
    # prepend 0
    newline_idxs_in_prompt = [0] + newline_idxs_in_prompt
    # append length of prompt + length of generated response
    newline_idxs_in_prompt.append(
        len(prompt_tokens) + len(gen_ids))

    # compute chunk lengths
    chunk_lengths = [
        newline_idxs_in_prompt[idx] - newline_idxs_in_prompt[idx-1]
        for idx in range(1, len(newline_idxs_in_prompt))
    ]

    # perform splitting
    if sum(chunk_lengths) != len(am_T_squeezed):
        # happens when the generated response starts with a punctuation
        # in this case, the tokenizer merges the speaker tag punctuation (:)
        # with the generated punctuation (eg .,)
        chunk_lengths[-1] -= 1
    attrib_chunks = torch.split(am_T_squeezed, chunk_lengths)

    if print_info:
        print('prompt tokens:', prompt_tokens)
        print('newlines indexes in prompt:', newline_idxs_in_prompt)
        print('chunk lengths:', chunk_lengths, '\nprompt chunks:')
        prompt_chunks = [
            prompt_tokens[sum(chunk_lengths[:idx]):sum(chunk_lengths[:idx])+chunk_len]
            for idx, chunk_len in enumerate(chunk_lengths)
        ]
        for idx, chunk in enumerate(prompt_chunks):
            print('chunk', idx, ', length:', chunk_lengths[idx],
                  ', tokens:', tkns_to_str(chunk).replace('\n', '<NL>'))
        print('attrib chunks shape:', [c.shape for c in attrib_chunks])

    # --------------------------------------
    # handle special tokens and newlines

    # remove scores for newline tokens
    if not count_newline_scores:
        # remove last item from each chunk except the last one
        attrib_chunks = [c[:-1] for c in attrib_chunks[:-1]] + [attrib_chunks[-1]]
        print('attrib chunks shape (-NL):', [c.shape for c in attrib_chunks]) \
            if print_info else None
    else:
        # convert tuple of tensors to list of tensors
        attrib_chunks = [c for c in attrib_chunks]

    # remove opt prefix token
    if prompt_tokens[0].token == opt_prefix_tkn:
        attrib_chunks[0] = attrib_chunks[0][1:]
        print('detected special starting token </s> - removing it') if print_info else None
        print('attrib chunks shape after removing </s>:', [c.shape for c in attrib_chunks]) \
            if print_info else None

    # --------------------------------------
    # handle speaker tags - further split attrib chunks into sub-chunks

    # chunk 1: speaker tag (first two tokens)
    # chunk 2: utterance (rest)
    attrib_chunks = [torch.split(c, [2, c.shape[0]-2]) for c in attrib_chunks]
    attrib_chunks = [item for sublist in attrib_chunks for item in sublist]  # flatten list of lists
    attrib_chunks = [c for c in attrib_chunks if c.shape[0] > 0]  # remove empty chunks
    print('chunks after tag-split:', [c.shape for c in attrib_chunks]) \
        if print_info else None

    # --------------------------------------
    # calculate utterance-level attributions

    # take mean of each chunk
    attrib_utt = [torch.nanmean(c) for c in attrib_chunks]

    # normalise by maximum absolute value, squash to [-1, 1]
    mav = max([abs(x) for x in attrib_utt])
    attrib_utt = [x / mav for x in attrib_utt]

    utterance_scores = attrib_utt[1::2]  # odd indexes
    utterance_scores = utterance_scores[::-1]  # reverse order
    if return_tag_scores:
        tag_scores = attrib_utt[::2]  # even indexes
        tag_scores = tag_scores[::-1]  # reverse order

    # --------------------------------------
    # done.

    if print_info:
        print('\n> utt scores:', utterance_scores)
        print('> all scores:', attrib_utt)
        print('> final scores:')
        for idx, utt_score in enumerate(utterance_scores):
            print('dr' if idx != 0 else 're', idx - 1, ', utt:', round(utt_score.item(), 3))
        print('sum of scores:', round(sum(utterance_scores).item(), 3))

    utt_scores_return = [x.item() for x in utterance_scores]
    if return_tag_scores:
        tag_scores_return = [x.item() for x in tag_scores]
        return utt_scores_return, tag_scores_return
    else:
        return utt_scores_return
