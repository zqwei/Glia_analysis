import numpy as np
from scipy.signal import convolve
from scipy.optimize import minimize


def pulse_resp(pulse_input, g_t, w_pulse):
    t_vec = np.where(pulse_input==1)[0]
    w_mat = np.zeros((len(t_vec), len(pulse_input)))
    for n in range(len(t_vec)):
        w_mat[n, t_vec[n]:t_vec[n]+len(w_pulse)] = w_pulse*g_t[n]
    return w_mat.sum(axis=0)


def mse_pulse_g_t_resp(g_t, y, x, w_pulse):
    y_est = y.copy()
    for n in range(y.shape[0]):
        y_est[n] = pulse_resp(x[n], g_t, w_pulse)
    return ((y-y_est)**2).sum()


def mse_pulse_w_pulse_resp(w_pulse, y, x, g_t):
    y_est = y.copy()
    for n in range(y.shape[0]):
        y_est[n] = pulse_resp(x[n], g_t, w_pulse)
    return ((y-y_est)**2).sum()


def eroll_min(x):
    from scipy.signal import find_peaks
    from scipy.interpolate import interp1d
    xmin_loc, _ = find_peaks(-x)
    xmin_loc = xmin_loc[x[xmin_loc]<x.max()*.2]
    return interp1d(xmin_loc, x[xmin_loc],kind = 'cubic',bounds_error = False, fill_value=0.0)(np.arange(len(x)))


def gp_smooth(y):
    from sklearn import gaussian_process
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    kernel = ConstantKernel()+ Matern(length_scale=1, nu=2.5)+ WhiteKernel(noise_level=1) #
    len_y = len(y)
    gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
    gp.fit(np.arange(len_y)[:, None], y)
    y_pred, _ = gp.predict(np.arange(len_y)[:, None], return_std=True)
    return y_pred


def motor_resp(swim_input, w_swim, t_pre):
    t_post = len(w_swim)-t_pre-1
    return convolve(swim_input, w_swim)[t_pre:-t_post]


def mse_motor_w_motor_resp(w_swim, y, x, t_pre):
    err = 0
    for n in range(len(y)):
        y_est = motor_resp(x[n], w_swim, t_pre)
        err += ((y[n]-y_est)**2).sum()
    return err


# def mse_pulse_g_t_resp(g_t, y, x, w_pulse):
#     y_est = y.copy()
#     for n in range(y.shape[0]):
#         y_est[n] = pulse_resp(x[n], g_t, w_pulse)
#     return ((y-y_est)**2).sum()