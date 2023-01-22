import os

from . import utils

DEFAULTS = {"corenlp_classpath": os.getenv("CLASSPATH")}
from .doc_db import DocDB
