import math


def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

class Smoother2d():
    def __init__(self, min_cutoff = 0.001, beta = 0.1):
        self.x_smoother = []

        self.min_cutoff = min_cutoff
        self.beta = beta
    
    def reset(self):
        # del self.smoother
        self.smoother = []

    def smooth(self, idx, item):

        # 如果高速移动场景下延迟比较严重，那么增大 beta ， 如果低速抖动比较严重， 那么减小 min_cutoff
        # min_cutoff = 0.001
        # beta = 0.01

        shape = item.shape
        
        return_item = item.copy()
        if idx == 0:
            for i in range(shape[0]):
                self.x_smoother.append(OneEuroFilter(idx, item[i][0], min_cutoff=self.min_cutoff, beta=self.beta))
        else:
            for i in range(shape[0]):
                return_item[i][0] = self.x_smoother[i](idx, item[i][0])
        
        return return_item


class SmootherHighdim():
    def __init__(self, min_cutoff = 0.001, beta = 0.1):
        self.smoother = None

        self.min_cutoff = min_cutoff
        self.beta = beta

    def reset(self):
        # del self.smoother
        self.smoother = None
    
    def smooth(self, idx, item):

        # 如果高速移动场景下延迟比较严重，那么增大 beta ， 如果低速抖动比较严重， 那么减小 min_cutoff
        # min_cutoff = 0.001
        # beta = 0.01

        shape = item.shape
        
        return_item = item.copy()
        if idx == 0:
            self.smoother = OneEuroFilter(idx, item, min_cutoff=self.min_cutoff, beta=self.beta)
        else:
            return_item = self.smoother(idx, item)
        
        return return_item