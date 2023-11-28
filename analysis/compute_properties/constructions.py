"""Extract constructions from dialogue excerpts (samples)."""
import os
import tempfile
import pandas as pd
import time
import subprocess as sp
from tqdm import tqdm
from pathlib import Path
from helpers import normalize, pipe


# make sure script can be run from anywhere
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'data')

# download Dialign JAR from this GitHub link
DIALIGN_GITHUB_URL = lambda version: \
    f"https://github.com/GuillaumeDD/dialign/releases/download/v{version}/dialign-{version}.zip"


class ConstructionExtractor:
    """
    Extract constructions from dialogue samples.

    Takes in a samples dataframe, and adds the following columns to it:
    - constrs_human_turn[0..CL-1]
        constructions in human response in turn x
    - constr_freqs_human_turn[0..CL-1]
        frequencies of constructions in human response in turn x
    - constrs_model_r[0-G-1]_turn[0..CL-1]
        constructions in model response r in turn x
    - constr_freqs_model_r[0..G-1]_turn[0..CL-1]
        frequencies of constructions in model response r in turn x
    
    Where
    - **CL**: context length
    - **G**: number of model responses

    With **CL**=10 and **G**=5, the total number of new columns added is:
        2*10 + 2*5*10 = 120
    """

    def __init__(
            self,
            samples_fp: str,
            output_file: str = None,
            n_model_responses: int = 5,
            context_length: int = 10,
            working_dir: str = None
    ) -> None:
        self.n_model_responses = n_model_responses
        self.context_length = context_length
        self.output_file = output_file

        # create temporary working folder
        if working_dir is not None:
            self.temp_dir = working_dir
        else:
            self.temp_dir = tempfile.mkdtemp(prefix='_tmp_dialign_', dir=DATA_DIR)
        if not Path(self.temp_dir).exists():
            Path(self.temp_dir).mkdir()
        print('WORKING FOLDER:', self.temp_dir)

        # download dialign jar
        self.dialign_path = self.download_dialign_jar_from_github()

        # load samples dataframe
        self.samples = pd.read_csv(samples_fp, sep='\t', header=0, index_col='sample_index')
        df_cols = self.samples.columns.tolist()

        # check if samples df has been populated with model responses
        self.df_has_model_responses = True
        try:
            self.df_has_model_responses = all(
                [f'model_response_r{i}' in self.samples.columns for i in range(self.n_model_responses)])
        except:
            self.df_has_model_responses = False

        # add new columns
        new_cols = []
        for turn_idx in range(self.context_length - 1):
            new_cols.append(f'constrs_human_turn{turn_idx}')
            new_cols.append(f'constr_freqs_human_turn{turn_idx}')
            if self.df_has_model_responses:
                for i in range(self.n_model_responses):
                    new_cols.append(f'constrs_model_r{i}_turn{turn_idx}')
                    new_cols.append(f'constr_freqs_model_r{i}_turn{turn_idx}')
        for col in new_cols:
            if col not in df_cols:
                self.samples[col] = ''

    def extract(self, clean_temp_dir: bool = True):
        """Extract constructions from dialogue excerpts (samples)."""
        # 1. create dialign input by converting samples to dialogues
        dialogues_paths = self.convert_samples_to_dialogues()

        # 2. run dialign to generate dialign output files (lexica)
        lexica_path = Path(self.temp_dir) / 'lexica'
        if not lexica_path.exists():
            lexica_path.mkdir()
        for dialogues_path in dialogues_paths:
            # get last directory name
            corpora = dialogues_path.parts[-1]
            lexicon_path = lexica_path / corpora
            if not lexicon_path.exists():
                lexicon_path.mkdir()
            self.run_dialign(dialogues_path, str(lexicon_path))
        self.check_dialign_output()

        # 3. merge lexica into samples dataframe
        for corpus in ['switchboard', 'maptask']:
            for sample_idx, sample in tqdm(self.samples.iterrows(), total=len(self.samples), desc=corpus):
                if corpus != sample['corpus']:
                    continue
                # read constructions from lexicon files
                constrs_human = self.get_constructions_from_lexicon(
                    str(lexica_path / corpus / f'{sample_idx}_0_tsv-lexicon.tsv'))
                for turn_idx in range(self.context_length - 1):
                    try:
                        self.samples.at[sample_idx, f'constrs_human_turn{turn_idx}'] = \
                            pipe(constrs_human[turn_idx][0])
                    except Exception as e:
                        print('SAMPLE INDEX', sample_idx)
                        print('TURN INDEX', turn_idx)
                        print(constrs_human)
                        print(f'Error: {e}')
                    self.samples.at[sample_idx, f'constr_freqs_human_turn{turn_idx}'] = \
                        pipe(constrs_human[turn_idx][1])
                if self.df_has_model_responses:
                    try:
                        constructions = [
                            self.get_constructions_from_lexicon(
                                str(lexica_path / corpus / f'{sample_idx}_{i + 1}_tsv-lexicon.tsv'))
                            for i in range(self.n_model_responses)
                        ]
                    except FileNotFoundError:
                        pass
                    for i in range(self.n_model_responses):
                        for turn_idx in range(self.context_length - 1):
                            self.samples.at[sample_idx, f'constrs_model_r{i}_turn{turn_idx}'] = \
                                pipe(constructions[i][turn_idx][0])
                            self.samples.at[sample_idx, f'constr_freqs_model_r{i}_turn{turn_idx}'] = \
                                pipe(constructions[i][turn_idx][1])

        # 4. save samples dataframe
        if self.output_file is not None:
            out_fp = Path(self.output_file)
        else:
            out_fp = Path(DATA_DIR) / 'corpora' / 'samples_constrs.tsv'
        self.samples.to_csv(out_fp, sep='\t', header=True, index=True)

        print('Finied extracting constructions.')
        print(f'Output file: {out_fp}')

        # 5. delete temporary working folder and its contents
        if clean_temp_dir:
            sp.run(['rm', '-rf', self.temp_dir])
            return None
        else:
            print(f'Not deleting temporary working folder: {self.temp_dir}')
            return self.temp_dir

    def convert_samples_to_dialogues(self, skip_if_exists: bool = True):
        """
        Create a list of dialogue tsv files from a samples dataframe.
        - Each sample (row) will generate n_generations+1 sample files.
        - Plus one is for the human response dialogue.
        - Conversation TSVs will be generated to out_dir.
        - There will be one subdirectory per corpus in the samples dataframe.
        """
        print('Converting samples to dialogue TSVs...')
        out_dir = Path(self.temp_dir) / 'dialogues'
        if not out_dir.exists():
            out_dir.mkdir()
        corpora = self.samples['corpus'].unique().tolist()
        out_dirs = []
        for corpora in corpora:
            # create a sub-directory for each corpora
            corpora_dir = out_dir / corpora
            if not corpora_dir.exists():
                corpora_dir.mkdir()
            out_dirs.append(corpora_dir)

            # if directory not empty, skip
            if len(os.listdir(corpora_dir)) > 0 and skip_if_exists:
                print(f'Skipping {corpora} because it is not empty.')
                continue

            # get all samples for this corpora
            corpus_samples = self.samples[self.samples['corpus'] == corpora]
            for sample_idx, sample in tqdm(
                corpus_samples.iterrows(),
                total=len(corpus_samples),
                desc=corpora,
                unit=' samples'
            ):
                human_response = sample['human_response'].strip()
                human_response = human_response.replace('A:', 'A:\t').replace('B:', 'B:\t')
                response_label = human_response.split('\t')[0].strip()
                context = sample['prompt'].strip()
                context = context.replace('\\n', '\n')
                context = context.replace('A:', 'A:\t').replace('B:', 'B:\t')
                dialogues = [f'{context}\n{human_response}']
                if self.df_has_model_responses:
                    model_responses = [sample[f'model_response_r{i}'] for i in range(self.n_model_responses)]
                    for model_response in model_responses:
                        # check if model response is empty
                        if pd.isna(model_response):
                            continue
                        model_response = model_response.replace('\\n', '')
                        model_response = model_response.strip()
                        model_response = response_label + '\t' + model_response
                        dialogues.append(f'{context}\n{model_response}')
                # write to file
                for i, dialogue in enumerate(dialogues):
                    with open(corpora_dir / f'{sample_idx}_{i}.tsv', 'w') as f:
                        f.write(dialogue)
        print('Done.')
        return out_dirs

    def run_dialign(
            self, 
            in_dir: str, 
            out_dir: str, 
            skip_if_exists: bool = True
    ):
        """Run Dialign JAR on a directory of dialogue TSV files."""
        if len(os.listdir(out_dir)) > 0 and skip_if_exists:
            print(f'Skipping {in_dir} because it is not empty.')
            return
        command = f'java -jar {self.dialign_path} -i "{in_dir}" -o "{out_dir}"'
        print(f'Running Dialign: `{command}`')
        ts = time.time()
        sp.run(command, shell=True)
        print(f"Done in {time.time() - ts:2f} seconds")

    def check_dialign_output(self):
        """Sanity check to make sure Dialign output files are consistent with the input dialogues."""
        da_in = Path(self.temp_dir) / 'dialogues'
        da_out = Path(self.temp_dir) / 'lexica'
        corpora = self.samples['corpus'].unique().tolist()
        had_warnings = False
        for corpus in corpora:
            samples_corpus = self.samples[self.samples['corpus'] == corpus]
            n_samples = len(samples_corpus)
            output_filenames = [f.name for f in (da_out / corpus).iterdir() if f.name.endswith('_tsv-lexicon.tsv')]
            n_files_da_out = len(output_filenames)
            if n_samples * ((self.n_model_responses if self.df_has_model_responses else 0) + 1) != n_files_da_out:
                print(f"WARNING: {corpus} has {n_samples} samples but {n_files_da_out} Dialign output files.")
                had_warnings = True
            input_filenames = [fn for fn in os.listdir(da_in / corpus) if fn.endswith('.tsv')]
            # find missing files
            missing_files = []
            inputs = [fn.replace('.tsv', '') for fn in input_filenames]
            outputs = [fn.replace('_tsv-lexicon.tsv', '') for fn in output_filenames]
            for i in inputs:
                if i not in outputs:
                    missing_files.append(i)
            if len(missing_files) > 0:
                print(f"WARNING: {corpus} is missing {len(missing_files)} Dialign output files.")
                print(f"Missing files: {missing_files}")
                had_warnings = True
        if not had_warnings:
            print('[CHECK PASS] Dialogues and Dialign output files are consistent.')
        else:
            print('[CHECK FAIL] Dialogues and Dialign output files are inconsistent.')

    def get_constructions_from_lexicon(self, lexicon_fp: str) -> list:
        """Retrieve the list of constructions from a Dialign lexicon file."""
        lex = pd.read_csv(lexicon_fp, sep='\t', header=0)

        # only interested in constructions that appear in the response:
        lex = lex[lex['Turns'].str.contains(str(self.context_length - 1))]
        if len(lex) == 0:
            return [([], []) for _ in range(self.context_length - 1)]

        # remove rows that have less than two alphanumeric characters
        lex = lex[lex['Surface Form'].str.len() > 2]

        # remove rows that have less than two words
        for i, row in lex.iterrows():
            if len(normalize(row['Surface Form'].strip()).strip().split()) < 2:
                lex.drop(i, inplace=True)

        # get constructions per turn
        constrs_per_turn = []
        for turn_idx in range(self.context_length - 1):
            constrs = lex[lex['Turns'].str.contains(str(turn_idx))]
            constr_strs = constrs['Surface Form'].tolist()
            constr_freqs = constrs['Freq.'].tolist()
            constrs_per_turn.append((constr_strs, constr_freqs))
        return constrs_per_turn

    def download_dialign_jar_from_github(self, version: str = '1.1'):
        """Download and extract the Dialign JAR file from GitHub."""
        
        print(f"Downloading Dialign JAR...")
        out_dir = Path(DATA_DIR) / "software"
        url = DIALIGN_GITHUB_URL(version)
        if not out_dir.exists():
            out_dir.mkdir()
        
        # check if dialign jar already exists
        jar_file = out_dir / f"dialign-{version}" / "dialign.jar"
        if jar_file.exists():
            print(f"Dialign JAR already exists: {jar_file}")
            return jar_file

        # download and extract
        print(f"URL: {url}")
        print(f"Output directory: {out_dir}")
        import requests
        import zipfile
        import io
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(out_dir)

        # remove everything except the JAR file
        for f in (out_dir / f"dialign-{version}").iterdir():
            if f.name != 'dialign.jar':
                f.unlink()
        
        # done
        print(f"Done.")
        return jar_file


# ==================================================================


def run_on_sample_level(
        input_file: str,
        output_file: str,
        n_model_responses: int = 5,
        context_length: int = 10,
        working_dir: str = None,
        delete_working_dir: bool = True
):
    extractor = ConstructionExtractor(
        samples_fp=input_file,
        output_file=output_file,
        n_model_responses=n_model_responses,
        context_length=context_length,
        working_dir=working_dir
    )
    temp_dir = extractor.extract(
        clean_temp_dir=delete_working_dir
    )
    if temp_dir is not None:
        print(f"Temporary working folder remains: {temp_dir}")
        # get last two directories
        temp_dir = Path(temp_dir)
        temp_dir = temp_dir.parts[-2:]
        # dialogues path
        dialogues_dir = temp_dir[0] + '/' + temp_dir[1] + '/dialogues/'
        print(f"Dialogues: {dialogues_dir}")


def test():
    extractor = ConstructionExtractor(
        samples_fp=str(Path(DATA_DIR) / 'corpora' / 'samples.tsv'),
        working_dir=str(Path(DATA_DIR) / '_tmp_dialign_qh42rk93')
    )
    temp_dir = extractor.extract(
        clean_temp_dir=False
    )
    print(temp_dir)


if __name__ == '__main__':
    import fire
    fire.Fire(run_on_sample_level)
