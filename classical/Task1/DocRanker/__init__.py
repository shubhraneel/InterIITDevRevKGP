from . import utils
from .doc_db import DocDB
from .tokenizer import CoreNLPTokenizer
import os
DEFAULTS = {
    'corenlp_classpath': os.getenv('CLASSPATH')
}

