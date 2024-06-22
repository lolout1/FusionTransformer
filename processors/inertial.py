from typing import List
import numpy as np
from processors.base import Processor

class InertialProcessor(Processor):
    def __init__(self, mode, duration):
        super.__init__(mode, duration)
        