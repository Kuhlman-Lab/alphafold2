import bz2
import pickle
import _pickle as cPickle
import hashlib

from typing import Any

# Saves the 'data' with the 'title' and adds the .pickle
def full_pickle(title: str, data: Any) -> None:
    pikd = open(title + '.pkl', 'wb')
    pickle.dump(data, pikd)
    pikd.close()

# Loads and returns a pickled object
def loosen(file: str) -> Any:
    pikd = open(file, 'rb')
    data = pickle.load(pikd)
    pikd.close()
    return data

# Pickle a file and then compress it into a file with extension
def compressed_pickle(title: str, data: Any) -> None:
    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        cPickle.dump(data, f)

# Load any compressed pickle file
def decompress_pickle(file: str) -> Any:
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data

# Look up hash of given string
def get_hash(x: str) -> str:
    return hashlib.sha1(x.encode()).hexdigest()
