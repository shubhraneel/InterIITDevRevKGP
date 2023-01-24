import os

from . import docranker_utils

DEFAULTS = {"corenlp_classpath": os.getenv("CLASSPATH")}
from .doc_db import DocDB
