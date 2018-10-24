import numpy as np
import matplotlib.pyplot as plt
from modelling.utilities import ProgressBar

class Solver:
    def __init__(self, function, initial=np.array([0,0])):
        self._function = function
        self._initial = initial
        self._solution = np.array([])
        self._solutions = np.zeros((len(function),1))
        self._time = np.array([])
        self._label = "none"
        self._fig, self._ax = plt.subplots()
        self._functions = function
    
    def __euler(self, start, end, h):
        x0 = self._initial[0]

        result_a = np.arange(x0, end, h)
        time_a = np.arange(x0, end, h)
        result_a[0] = self._initial[1]

        for (i, previous_result) in enumerate(result_a[:-1]):
            result_a[i+1] = previous_result + h*self._function(time_a[i], previous_result)

        result_b = np.arange(x0, start-h, -h)
        time_b = np.arange(x0, start-h, -h)
        result_b[0] = self._initial[1]

        for (i, previous_result) in enumerate(result_b[:-1]):
            result_b[i+1] = previous_result + -h*self._function(time_b[i], previous_result)

        result = np.concatenate((result_b[::-1], result_a[1:]))
        time = np.concatenate((time_b[::-1], time_a[1:]))

        self._solution = result
        self._time = time
        self._label = "Euler"

        return np.stack([time, result])
    
    def __heun(self, start, end, h):
        x0 = self._initial[0]

        result_a = np.arange(x0, end, h)
        time_a = np.arange(x0, end, h)
        result_a[0] = self._initial[1]

        for (i, previous_result) in enumerate(result_a[:-1]):
            y_i1 = previous_result + h*self._function(time_a[i], previous_result)
            result_a[i+1] = previous_result + h/2*(self._function(time_a[i], previous_result) + self._function(time_a[i+1], y_i1))

        result_b = np.arange(x0, start-h, -h)
        time_b = np.arange(x0, start-h, -h)
        result_b[0] = self._initial[1]

        for (i, previous_result) in enumerate(result_b[:-1]):
            y_i1 = previous_result - h*self._function(time_b[i], previous_result)
            result_b[i+1] = previous_result - h/2*(self._function(time_b[i], previous_result) + self._function(time_b[i+1], y_i1))

        result = np.concatenate((result_b[::-1], result_a[1:]))
        time = np.concatenate((time_b[::-1], time_a[1:]))

        self._solution = result
        self._time = time
        self._label = "Heun"

        return np.stack([time, result])

    def __rk4(self, start, end, h):
        result = np.arange(start, end, h)
        time = np.arange(start, end, h)
        result[0] = self._initial
        for (i,_) in enumerate(time[1:],1):
            k1 = h*self._function(time[i-1], result[i-1])
            k2 = h*self._function(time[i-1] + h/2, result[i-1] + k1/2)
            k3 = h*self._function(time[i-1] + h/2, result[i-1] + k2/2)
            k4 = h*self._function(time[i-1] + h, result[i-1] + k3)

            result[i] = result[i-1] + 1/6*(k1+2*k2+2*k3+k4)

        self._solution = result
        self._time = time
        self._label = "Runge-Kutta 4"
        
        return np.stack([time, result], axis=1)
    
    def __rk_4_vec(self, start, end, h):
        result = np.zeros([len(self._functions), int((end-start)/h)])
        time = np.arange(start, end, h)
        result[:,0] = self._initial[1:].T
        k_values = np.zeros((len(self._functions), 4))
        
        for (i,t) in enumerate(time[:-1],0):
            for (k, func) in enumerate(self._functions):
                k_values[k,0] = h*func(t, *result[:,i])
            for (k, func) in enumerate(self._functions):
                # print(result[:,i] + k_values[k,0]/2, k_values[:,0])
                k_values[k,1] = h*func(t + h/2, *(result[:,i] + k_values[:,0]/2))
            for (k, func) in enumerate(self._functions):
                k_values[k,2] = h*func(t + h/2, *(result[:,i] + k_values[:,1]/2))
            for (k, func) in enumerate(self._functions):
                k_values[k,3] = h*func(t + h, *(result[:,i] + k_values[:,2]))

            result[:,i+1] = result[:,i] + 1/6*(k_values[:,0]+2*k_values[:,1]+2*k_values[:,2]+k_values[:,3])

        self._solutions = result
        self._time = time
        self._label = "Runge-Kutta 4"
        return np.vstack([time, result])

    def __rk45_vec(self, start, end, error=0.0001, h=0.5, limit=100):
        progress = ProgressBar()
        t = self._initial[0]

        result = np.zeros((len(self._functions),1))
        result[:, 0] = self._initial[1:].T
        # print(result)
    
        time = np.array([t])

        while(t < end):
            h, y5 = self.__rk45_step_vec(limit, h, t, result[:,-1], error)
            t += h
            # print(result.shape, y5.shape)
            result = np.concatenate([result, y5], axis=1)
            # print(result)
            time = np.append(time, [t])
            # print(t)
            progress.step(h/(end-start)*100)
            progress.show()
        
        h = -h
        t = self._initial[0]
        while(start < t):
            h, y5 = self.__rk45_step_vec(limit, h, t, result[:,0], error)
            t += h
            result = np.append([y5], result)
            time = np.append([t], time)
            progress.step(-h/(end-start)*100)
            progress.show()


        self._solutions = result
        self._time = time
        self._label = "Runge-Kutta 45"

        return np.vstack([time, result])


    def __rk45(self, start, end, error=0.0001, h=0.5, limit=100):
        t = self._initial[0]
        result = np.array([self._initial[1]])
        time = np.array([t])

        while(t < end):
            h, y5 = self.__rk45_step(limit, h, t, result[-1], error)
            t += h
            result = np.append(result, [y5])
            time = np.append(time, [t])
        
        h = -h
        t = self._initial[0]
        while(start < t):
            h, y5 = self.__rk45_step(limit, h, t, result[0], error)
            t += h
            result = np.append([y5], result)
            time = np.append([t], time)

        self._solution = result
        self._time = time
        self._label = "Runge-Kutta 45"

        return np.stack([time, result])

    def __rk45_step_vec(self, limit, h, ti, yi, max_error, error_factor=2):
        i = 0
        k_values = np.zeros((len(self._functions), 6))
        while (i < limit):
            i += 1

            for (k, func) in enumerate(self._functions):
                k_values[k,0] = h*func(ti, *yi)
            for (k, func) in enumerate(self._functions):
                k_values[k,1] = h*func(ti + h/4, *(yi + k_values[:,0]))
            for (k, func) in enumerate(self._functions):
                k_values[k,2] = h*func(ti + h*3/8, *(yi + k_values[:,0]*3/32 + k_values[:,1]*9/32))
            for (k, func) in enumerate(self._functions):
                k_values[k,3] = h*func(ti + h*12/13, *(yi + k_values[:,0]*1932/2197 - k_values[:,1]*7200/2197 + k_values[:,2]*7296/2197))
            for (k, func) in enumerate(self._functions):
                k_values[k,4] = h*func(ti + h, *(yi + k_values[:,0]*439/216 - k_values[:,1]*8 + k_values[:,2]*3680/513 - k_values[:,3]*845/4104))
            for (k, func) in enumerate(self._functions):
                k_values[k,5] = h*func(ti + h/2, *(yi - k_values[:,0]*8/27) + k_values[:,1]*2 - k_values[:,2]*3544/2565 + k_values[:,3]*1859/4104 - k_values[:,4]*11/40)

            y5 = yi + 16/135*k_values[:,0] + 6656/12825*k_values[:,2] + 28561/56430*k_values[:,3] - 9/50*k_values[:,4] + 2/55*k_values[:,5]
            y4 = yi + 25/216*k_values[:,0] + 1408/2565*k_values[:,2] + 2197/4104*k_values[:,3] - 1/5*k_values[:,4]

            error = np.amax(np.abs(y5-y4))
            # print(error, max_error)
            if (error > max_error):
                h /= error_factor 
            elif (error < max_error/error_factor):
                h *= error_factor
                break
            else:
                break
        return h, np.reshape(y5, (len(self._functions),1))

    def __rk45_step(self, limit, h, ti, yi, error):
        i = 0
        while (i < limit):
            i += 1

            k1 = h*self._function(ti, yi)
            k2 = h*self._function(ti+h/4, yi+k1/4)
            k3 = h*self._function(ti+3*h/8, yi+3*k1/32+9*k2/32)
            k4 = h*self._function(ti+12*h/13, yi+1932/2197*k1-7200/2197*k2+7296/2197*k3)
            k5 = h*self._function(ti+h, yi+k1*439/216-k2*8+k3*3680/513-k4*845/4104)
            k6 = h*self._function(ti+h/2, yi-8/27*k1+2*k2-3544/2565*k3+1859/4104*k4-11/40*k5)

            y5 = yi + 16/135*k1+6656/12825*k3+28561/56430*k4-9/50*k5+2/55*k6
            y4 = yi + 25/216*k1+1408/2565*k3+2197/4104*k4-1/5*k5

            if (abs(y5-y4) > error):
                h /= 2
            else:
                break
        return h, y5

    def run(self, method, start, end, h):
        if (method == "euler"):
            return self.__euler(start, end, h)
        elif (method == "heun"):
            return self.__heun(start, end, h)
        elif (method == "rk4"):
            return self.__rk_4_vec(start, end, h)
        elif (method == "rk45"):
            return self.__rk45_vec(start, end, h=h)
        else:
            return None

    def plot(self, label=[]):
        if not label:
            label = self._label
        for (i, sol) in enumerate(self._solutions):
            self._ax.plot(self._time, sol, label=label[i])
        
    def show(self):
        self._ax.legend()
        plt.title("Populations vs Time")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Population")
        plt.show()