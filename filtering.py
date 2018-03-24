#!/usr/bin/env python
""" filtering.py:
Multiple filtering techniques to be used on different applications.

Inspired by:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__date__ = "March 24, 2018"

# import
import numpy as np
import matplotlib.pyplot as plt

class Filtering():
    def __init__(self):
        pass

    def g_h_filter(self, data, x0, dx, g, h, dt):
        """ Performs g-h filter on 1 state variable with a fixed g and h.

        'data' contains the data to be filtered.
        'x0' is the initial value for our state variable
        'dx' is the initial change rate for our state variable
        'g' is the g-h's g scale factor
        'h' is the g-h's h scale factor
        'dt' is the length of the time step 
        """
        x_est = x0
        results = []
        x_preds = []
        for z in data:
            # prediction step
            x_pred = x_est + (dx*dt)
            x_preds.append(x_pred)
            dx = dx

            # update step
            residual = z - x_pred
            dx = dx + h * (residual) / dt
            x_est = x_pred + g * residual
            results.append(x_est)

        return np.array(results), np.array(x_preds)

    def _gen_data(self, x0, dx, count, noise_factor=1, accel=0.):
        """ Generate noisy data with zero mean and unit standard variation
        """
        zs = []
        truth = []
        for i in range(count):
            truth.append(x0 + dx*i)
            zs.append(x0 + dx*i + np.random.randn()*noise_factor)
            dx += accel
        return zs, truth

    def _test_g_h_filter(self):
        # generate random data points (measurements)
        # sample data points from a normal distribution plus a trend

        # generate measurements and truth value
        meas_x0 = 10
        meas_dx = 0
        n_data = 20
        noise_factor = 0
        accel = 2
        measurements, truth = self._gen_data(meas_x0, meas_dx, n_data, noise_factor, accel)

        # define filter initial conditions
        x0 = 10
        dx = 0
        g, h = 0.2, 0.02
        dt = 1

        # filter
        results, x_preds = self.g_h_filter(measurements, x0, dx, g, h, dt)

        # plot results
        plt.figure()
        plt.title('g-h Filter')
        plt.xlabel('Time Step []')
        plt.ylabel('Value []')
        plt.plot(measurements, 'kx', label='measurements')
        plt.plot(truth, 'k--', label='truth')
        plt.plot(results, label='filtered')
        # plt.plot(x_preds, 'x', label='pred')
        plt.grid()
        plt.legend(loc='best')
        plt.xlim([0, n_data - 1])
        plt.show()

    def _test_g_h_filter_bad_init(self):
        """ Example for bad initial conditions
        """
        # generate random data points (measurements)
        # sample data points from a normal distribution plus a trend

        # generate measurements
        meas_x0 = 5
        meas_dx = 2
        n_data = 50
        noise_factor = 10
        measurements, truth = self._gen_data(meas_x0, meas_dx, n_data, noise_factor, accel=.1)

        # define filter initial conditions
        x0 = 100
        dx = 2
        g, h = 0.2, 0.02
        dt = 1

        # filter
        results, x_preds = self.g_h_filter(measurements, x0, dx, g, h, dt)

        # plot results
        plt.figure()
        plt.title('g-h Filter')
        plt.xlabel('Time Step []')
        plt.ylabel('Value []')
        plt.plot(measurements, 'kx', label='measurements')
        plt.plot(truth, 'k--', label='truth')
        plt.plot(results, label='filtered')
        # plt.plot(x_preds, 'x', label='pred')
        plt.grid()
        plt.legend(loc='best')
        plt.xlim([0, n_data - 1])
        plt.show()


if __name__ == '__main__':
    filtering = Filtering()
    filtering._test_g_h_filter()