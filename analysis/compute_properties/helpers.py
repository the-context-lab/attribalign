"""String helper functions."""
import re
import string


def add_space_before_non_alphanum(input_str: str) -> str:
    """Add space before non-alphanumeric characters."""
    return re.sub(r'([^\s\w]|_)+', r' \1 ', input_str)


def rm_non_alphanum_from_str(input_str: str, sub: str = ' ') -> str:
    """Remove non-alphanumeric characters from string."""
    return (''.join(re.compile(r'[\W_]+').sub(sub, input_str))).strip()


def remove_punctuation(text):
    """Replace punctuation with space."""
    return text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))


def rm_extra_spaces(text):
    """Format string to only contain single spaces."""
    return ' '.join(text.split())


def normalize(text):
    """Clean/normalize string."""
    return rm_extra_spaces(remove_punctuation(text.strip().lower()).strip()).strip()


def find_subsequence(subsequence, sequence):
    """Find sub-string in string."""
    l = len(subsequence)
    ranges = []
    for i in range(len(sequence)):
        if sequence[i:i+l] == subsequence:
            if i - 1 < 0:
                space_before = True
            else:
                space_before = sequence[i-1] in " ',.!:;?"
            if i + l >= len(sequence):
                space_after = True
            else:
                space_after = sequence[i+l] in " ',.!:;?"
            if space_before and space_after:
                ranges.append((i, i+l))
    return ranges


def pipe(in_list: list):
    """Convert a list of a pipe-separated string."""
    return '|'.join([str(i) for i in in_list])


def unpipe(in_str: str):
    """Convert a pipe-separated string to a list."""
    if not isinstance(in_str, str):
        return []
    if in_str.strip() == '':
        return []
    dtype = str
    try:
        float(in_str.split('|')[0])
        dtype = float
    except ValueError:
        pass
    return [dtype(i) for i in in_str.split('|')]
