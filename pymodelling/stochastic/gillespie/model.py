import numpy as np
import matplotlib.pyplot as plt
from ...utilities import ProgressBar


class Model:
  def __init__(self, T=500, V=100):
    self.T = T # maximum elapsed time
    self.stateNames = list()
    self.stateInitialValues = np.empty([0], dtype=np.float)
    self.stateTransitionFunctions = list()
    self.stateW = list()
    # self.stateValues = np.empty([0], dtype=np.float)
    self.timeStamp = np.empty([1,0], dtype=np.float)
    self.V = V
    self.transitions = list()
    self.t = 0

  def addTransition(self, transitionFunction, weightFunction):
    self.stateTransitionFunctions.append(transitionFunction)
    self.stateW.append(weightFunction)

  def addState(self, name, initialValue=0):
    self.stateNames.append(name)
    self.stateInitialValues = np.insert(self.stateInitialValues, 0, initialValue)

  def run(self):
    progress = ProgressBar()
    stateInitialValues = np.flip(self.stateInitialValues, axis=0)
    self.stateHistoricalValues = np.empty([0,len(stateInitialValues)], dtype=np.float)
    stateValues = np.array(stateInitialValues)
    self.timeStamp = np.empty([1,0], dtype=np.float)

    while self.t < self.T:
      progress.update_progress(self.t/self.T*100)
      progress.show()
      state = dict(zip(self.stateNames, stateValues))
      w_array = [w(state, self) for w in self.stateW]
      W = sum(w_array)
      # print(w_array, W)

      if W == 0:
        self.t = self.T
        self.timeStamp = np.insert(self.timeStamp, 0, self.t)
        self.stateHistoricalValues = np.vstack((self.stateHistoricalValues, stateValues))
        break

      dt = -np.log(np.random.random_sample()) / W
      # print(dt)
      self.timeStamp = np.insert(self.timeStamp, 0, self.t)
      self.t += dt
      # print(self.t, dt)

      event = np.random.random_sample()
      sum_a = 0
      idx = 0

      # print(w_array)
      for w in w_array:
        if  sum_a / W <= event < (sum_a + w) / W:
            self.stateHistoricalValues = np.vstack((self.stateHistoricalValues, stateValues))
            newStates = self.stateTransitionFunctions[idx](state, dt, w, W)
            stateValues = [max(newStates[state], 0) for state in self.stateNames]
            break
        else:
            idx += 1
            sum_a += w
    
    progress.update_progress(self.t/self.T*100)
    progress.show()
    self.t = 0


  def plot(self):
    for i in range(0, len(self.stateNames)):
      plt.plot(self.timeStamp.reshape(1,len(self.timeStamp)).T, np.flip(self.stateHistoricalValues[:,i:i+1], axis=0))
    plt.title("Populations vs Time")
    plt.legend(self.stateNames)
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.show()
