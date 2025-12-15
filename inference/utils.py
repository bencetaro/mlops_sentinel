
import re
from glob import glob
from pathlib import Path

def sorted_alphanumeric(files):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(files, key=alphanum_key)

def make_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p