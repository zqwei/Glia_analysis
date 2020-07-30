import numpy as np
from scipy.signal import convolve
from scipy.optimize import minimize


def eroll_min(x):
    from scipy.signal import find_peaks
    from scipy.interpolate import interp1d
    xmin_loc, _ = find_peaks(-x)
    xmin_loc = xmin_loc[x[xmin_loc]<x.max()*.2]
    return interp1d(xmin_loc, x[xmin_loc],kind = 'cubic',bounds_error = False, fill_value=0.0)(np.arange(len(x)))


def pulse_resp(pulse_input, g_t, w_pulse):
    t_vec = np.where(pulse_input==1)[0]
    w_mat = np.zeros(len(pulse_input))
    for n in range(len(t_vec)):
        w_mat[t_vec[n]:t_vec[n]+len(w_pulse)] += w_pulse*g_t[n]
    return w_mat


def motor_resp(swim_input, w_swim, t_pre):
    t_post = len(w_swim)-t_pre-1
    return convolve(swim_input, w_swim)[t_pre:-t_post]


def motor_pulse_resp(swim_input, w_swim, t_pre, pulse_input, g_t, w_pulse):
    return motor_resp(swim_input, w_swim, t_pre) + pulse_resp(pulse_input, g_t, w_pulse)


def mse_motor_pulse_resp(y, swim_input_, w_swim, t_pre, pulse_input_, g_t, w_pulse):
    err = 0
    for n in range(len(y)):
        y_est = motor_pulse_resp(swim_input_[n], w_swim, t_pre, pulse_input_[n], g_t, w_pulse)
        err += ((y[n]-y_est)**2).sum()
    return err    


def mse_motor_w(w_swim, y, swim_input_, t_pre, pulse_input_, g_t, w_pulse):
    return mse_motor_pulse_resp(y, swim_input_, w_swim, t_pre, pulse_input_, g_t, w_pulse)


def mse_pulse_g_t(g_t, y, swim_input_, w_swim, t_pre, pulse_input_, w_pulse):
    return mse_motor_pulse_resp(y, swim_input_, w_swim, t_pre, pulse_input_, g_t, w_pulse)


def mse_pulse_w(w_pulse, y, swim_input_, w_swim, t_pre, pulse_input_, g_t):
    return mse_motor_pulse_resp(y, swim_input_, w_swim, t_pre, pulse_input_, g_t, w_pulse)


def mse_motor_pulse_w(w_ps, y, swim_input_, t_pre, pulse_input_, g_t, l_pulse):
    w_pulse=w_ps[:l_pulse]
    w_swim=w_ps[l_pulse:]
    return mse_motor_pulse_resp(y, swim_input_, w_swim, t_pre, pulse_input_, g_t, w_pulse)


# def mse_pulse_g_t_resp(g_t, y, x, w_pulse):
#     y_est = y.copy()
#     for n in range(y.shape[0]):
#         y_est[n] = pulse_resp(x[n], g_t, w_pulse)
#     return ((y-y_est)**2).sum()