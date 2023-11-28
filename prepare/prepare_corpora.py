"""
Prepare Switchboard and Map Task for analysis and model training.
    
Usage:

[to prepare raw corpora for model training and inference]
    python prepare_corpora.py prepare
    python prepare_corpora.py prepare --context_length 10 --eos_token '<|endoftext|>'

[to download raw corpora]
    python prepare_corpora.py download
    python prepare_corpora.py download --should_zip
    python prepare_corpora.py download --should_zip --zip_file ../data/corpora/raw.tar.gz
"""
import os
import sys
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# make sure script can be run from anywhere
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_FOLDER = os.path.join(SCRIPT_PATH, '..', 'data')


def get_file_from_url(
        url: str,
        output_dir: str,
        return_file_contents: bool = False,
        verboase: bool = False,
        progress_bar: bool = False,
):
    """Download file from URL if it doesn't exist already."""

    file_name = url.split('/')[-1]
    output_path = Path(output_dir) / file_name
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    if not output_path.exists():
        print(f"Downloading {file_name} from {url}...") if verboase else None
        response = requests.get(url, stream=True)
        if progress_bar:
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            with open(output_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print("ERROR, something went wrong")
                print(f"Downloaded {file_name} to {output_path}.") if verboase else None
                raise Exception("ERROR, something went wrong")
        else:
            with open(output_path, 'wb') as file:
                file.write(response.content)
        print(f"Downloaded {file_name} to {output_path}.") if verboase else None
    else:
        print(f"{file_name} already exists in {output_path}.") if verboase else None
    if return_file_contents:
        with open(output_path, 'r') as file:
            return file.read(), str(output_path)
    return str(output_path)


def download_corpora_from_github(corpus: str, output_dir: str) -> str:
    """
    Download Switchboard and Map Task from the following repositories:
    - Switchboard: https://github.com/NathanDuran/Switchboard-Corpus
    - Map Task: https://github.com/NathanDuran/Maptask-Corpus
    """

    base_urls = {
        'switchboard': "https://raw.githubusercontent.com/NathanDuran/Switchboard-Corpus/master/swda_data/",
        'maptask': "https://raw.githubusercontent.com/NathanDuran/Maptask-Corpus/master/maptask_data/"
    }
    index_links = {
        'train': 'metadata/train_split.txt',
        'val': 'metadata/val_split.txt',
        'test': 'metadata/test_split.txt',
    }

    base_url = base_urls[corpus]
    corpus_name = 'Switchboard' if corpus == 'switchboard' else 'Map Task'
    print(f"Downloading {corpus_name} corpus from GitHub repo to {output_dir}.")
    files_base_url = lambda split: base_url + split + "/"

    failed_files = []
    for split, index_link in index_links.items():
        temp_dir_path = Path(output_dir) / "__temp__"
        file_names, file_names_path = get_file_from_url(base_url + index_link, temp_dir_path, return_file_contents=True)
        file_names = file_names.split('\n')
        file_names = [file_name.strip() + '.txt' for file_name in file_names if file_name.strip() != '']
        out_dir_path = Path(output_dir) / split
        for file_name in tqdm(file_names, desc=f"Downloading {split} split"):
            file_url = files_base_url(split) + file_name
            try:
                get_file_from_url(file_url, out_dir_path)
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    sys.exit(0)
                failed_files.append(file_url)
        Path(file_names_path).unlink()  # delete downloaded index files
        # remove temp directory
        temp_dir_path.rmdir()
    if len(failed_files) > 0:
        print(f"Failed to download {len(failed_files)} files:")
        for file_url in failed_files:
            print(file_url)
    print("Done.")


def download(
        output_dir: str = None,
        should_zip: bool = False,
        zip_file: str = None,
):
    """Download Switchboard and Map Task"""

    print('*' * 80);
    print('Downloading corpora')
    if output_dir is None:
        out_dir = Path(DATA_FOLDER) / 'corpora' / 'raw'
    out_sw = Path(out_dir) / 'switchboard'
    out_mt = Path(out_dir) / 'maptask'
    print(f"Output dir: {out_dir}")

    # check if corpora already exist: if output directory exists and is not empty
    already_downloaded = False
    if (out_sw.exists() and len(list(out_sw.glob('*'))) > 0) or (out_mt.exists() and len(list(out_mt.glob('*'))) > 0):
        already_downloaded = True
        print("Corpora already downloaded.")

    # if already_downloaded and input("Do you want to re-download the corpora? (y/n) ").lower() != 'y':
    #     return

    if not already_downloaded:
        download_corpora_from_github('switchboard', out_sw)
        download_corpora_from_github('maptask', out_mt)

    if should_zip:  # zip corpora
        if zip_file is None:
            zip_file = Path(DATA_FOLDER) / 'corpora' / 'raw.tar.gz'
        print(f"Zipping corpora to {zip_file}")
        import tarfile
        with tarfile.open(zip_file, "w:gz") as tar:
            tar.add(out_dir, arcname=os.path.basename(out_dir))
        print(f"Done. Zipped size: {os.path.getsize(zip_file) / 1e6:.3f} MB")


def prepare(
        context_length: int = 10,
        eos_token: str = '<|endoftext|>',
):
    """Prepare raw Switchboard and Map Task corpora"""
    path_raw = Path(DATA_FOLDER) / 'corpora' / 'raw'

    fp_out = os.path.join(DATA_FOLDER, f'samples.tsv')
    # check if already exists and not empty
    if os.path.exists(fp_out) and os.path.getsize(fp_out) > 0:
        print(f"Samples already prepared in {fp_out}.")
        print("Re-create? (y/n)")
        if input().lower() != 'y':
            return

    # process to model training format
    print('*' * 80);
    print('Processing corpora to model training format')
    for corpus in ['switchboard', 'maptask']:
        for split in ['train', 'val', 'test']:  # no test split as that is used for generation
            filenames = list(Path(f'{path_raw}/{corpus}/{split}').glob('*.txt'))
            filename_out = os.path.join(DATA_FOLDER, 'model_train', corpus, f'{split}.txt')
            if not os.path.exists(os.path.dirname(filename_out)):
                os.makedirs(os.path.dirname(filename_out))
            with open(filename_out, 'w') as f_out:
                for filename in tqdm(filenames, desc=f"Processing {corpus} {split}"):
                    lines = open(filename, 'r').readlines()
                    lines = [line.strip() for line in lines]
                    lines = [line for line in lines if line != '']
                    # merge consecutive utterances from the same speaker
                    utterances = []
                    for utterance in lines:
                        try:
                            speaker, utterance, _ = utterance.split('|')
                        except:
                            continue
                        if len(utterances) > 0 and speaker == utterances[-1][0]:
                            utterances[-1][1] += '. ' + utterance
                        else:
                            utterances.append([speaker, utterance])
                    # write to file
                    gf_map = {'g': 'A', 'f': 'B'}
                    for speaker, utterance in utterances:
                        speaker_label = gf_map[speaker] if corpus == 'maptask' else speaker
                        f_out.write(speaker_label + ': ' + utterance + '\n')
                    f_out.write(eos_token + '\n')

    # process to model inference format
    print('*' * 80);
    print('Processing corpora to model inference format')
    cl = context_length - 1
    split = 'test'

    # define output dataframe
    if not os.path.exists(os.path.dirname(fp_out)):
        os.makedirs(os.path.dirname(fp_out))
    with open(fp_out, 'w') as f_out:
        f_out.write('corpus\tturns_in_diag\tfirst_utt_idx_in_diag\thuman_response\tprompt\n')
    df = pd.read_csv(fp_out, sep='\t', header=0, index_col=0)

    # add samples to output dataframe
    for corpus in ['switchboard', 'maptask']:
        model_train = open(os.path.join(DATA_FOLDER, 'model_train', corpus, f'{split}.txt'), 'r').read()

        # get dialogues
        diags = model_train.split(eos_token)
        diags = [diag.strip() for diag in diags]
        diags = [diag for diag in diags if diag != '']

        # get utterances in all dialogues
        diags = [[utt.strip() for utt in diag.split('\n') if utt.strip() != ''] for diag in diags]

        # retrieve samples from each dialogue
        samples = []
        for diag in tqdm(diags, desc="{corpus:15}-{split:<6}".format(corpus=corpus, split=split)):
            turns_in_diag = len(diag)
            for i in range(len(diag) - cl):
                prompt = '\\n'.join(diag[i:i + cl])
                human_response = diag[i + cl]
                prompt = prompt.replace('A: ', 'A:').replace('B: ', 'B:')
                human_response = human_response.replace('A: ', 'A:').replace('B: ', 'B:')
                samples.append([corpus, turns_in_diag, i, human_response, prompt])

        # add samples to output dataframe
        df = pd.concat([
            df,
            pd.DataFrame(samples, columns=[
                'corpus',
                'turns_in_diag',
                'first_utt_idx_in_diag',
                'human_response',
                'prompt'])
            ], axis=0
        )

    # save output dataframe
    df['sample_index'] = range(len(df))
    df.to_csv(fp_out, sep='\t', header=True, index=False)
    print(f"Saved {len(df)} samples to {fp_out}")
    print("Sample output:")
    print(df.head())

    # remove model_train/test.txt
    for corpus in ['switchboard', 'maptask']:
        fp_to_remove = os.path.join(DATA_FOLDER, 'model_train', corpus, f'test.txt')
        os.remove(fp_to_remove)
    print("Done.")


if __name__ == '__main__':
    import fire

    fire.Fire({
        'download': download,
        'prepare': prepare,
    })
