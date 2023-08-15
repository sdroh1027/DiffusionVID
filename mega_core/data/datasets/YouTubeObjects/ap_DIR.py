import os
from glob import glob


def ap_Dir(path, mustbedir=False):
    # Get the list of a directory in a path with specific file extension (ex: .jpg)
    files = glob(path)
    files.sort()
    out = []
    for p in files:
        if (mustbedir and not os.path.isdir(p)) or p[0] == '.':
            continue
        out.append(p)

    return out
