# read version from installed package
from importlib.metadata import version
__version__ = version("cyclicityanalysis")

# populate package namespace
from .orientedarea import *
from .coom import *

