from . import docranker_utils
import os
DEFAULTS = {
    'corenlp_classpath': os.getenv('CLASSPATH')
}
from .doc_db import DocDB

