import numpy as np
import matplotlib.pyplot as plt

class Solver:
    def __init__(self, function, initial):
        self._function = function