class ProgressBar:
    def __init__(self):
        self._progress = 0.0
    
    def step(self, step=0.1):
        self._progress += step

    def show(self):
        print("\r[{0}{1}] {2:.2f}%".format('#'*int(self._progress*3//10), ' '*(30-int(self._progress*3//10)), self._progress), end='')
        if self._progress >= 100:
            print()

    def update_progress(self, progress):
        self._progress = progress