import numpy as np
import pandas as pd
import tweedie
import scipy.stats as st

def pred_list_to_prob_array(pred_list, cumulative=False, total_days=30):
    prob_array = np.zeros((pred_list.shape[0], total_days))
    pred_list = np.clip(pred_list, 1, total_days)
    for row, e in enumerate(pred_list):
        if cumulative:
            prob_array[row, int(e-1)] = 1.
        else:
            prob_array[row, int(e-1):] = 1.
        
    if cumulative:
        prob_array = prob_array+1e-4 
        prob_array = np.divide(prob_array, prob_array.sum(axis=1).reshape(-1,1))
        prob_array = prob_array.cumsum(axis=1)

    return prob_array

def pred_list_to_prob_array_mc(pred_list, total_days=30):
    prob_array = np.zeros((pred_list.shape[0], total_days))
    pred_list = np.clip(pred_list, 1, total_days)
    for row, e in enumerate(pred_list):
        prob_array[row, int(e):] = 1.
        
    return prob_array

def rps(y, p, probs=False, total_days=30):
    y_array = pred_list_to_prob_array(y, total_days=total_days)
    if probs:
        p_array = p.cumsum(axis=1)
    else:
        p_array = pred_list_to_prob_array(p, cumulative=True, total_days=total_days)
    return ((p_array - y_array)**2).sum(axis=1).mean()


def rps_mc(y, p, probs=False, total_days=30):
    y_array = pred_list_to_prob_array_mc(y, total_days=total_days)
    if probs:
        p_array = p.cumsum(axis=1)
    return ((p_array - y_array)**2).sum(axis=1).mean()

def rps_raw(y, p, probs=False):
    y_array = pred_list_to_prob_array(y)
    if probs:
        p_array = p.cumsum(axis=1)
    else:
        p_array = pred_list_to_prob_array(p, cumulative=True)
    return ((p_array - y_array)**2).sum(axis=1)


def pred_list_to_tweedie(pred_list, phi=1, p=1.5):
    # has a bug in the first day, it's the wrong probability, but it's worse without the bug
    distros = dict()
    for mu in range(1,31):
        distros[mu] = [tweedie.tweedie(p=p, mu=mu, phi=phi).cdf(days) for days in range(1,31,1)]
        distros[mu][1:] = np.diff(distros[mu])
        distros[mu] = np.round(distros[mu] / np.sum(distros[mu]), 4)
    
    prob_array = np.zeros((pred_list.shape[0], 30))

    for row, mu in enumerate(pred_list):
        prob_array[row, :] = distros[mu]#.cumsum()
        #prob_array[row, -1] = 1.

    return prob_array



def pred_list_to_distro(pred_list, wei=False, total_days=30, phi=2, power=1.5):
    distros = dict()
    for mu in range(1,total_days+1):
        if wei:
            distros[mu] = [st.norm.cdf(days, loc=mu, scale=1) for days in range(0,total_days+1,1)]
        else:
            distros[mu] = [tweedie.tweedie(p=power, mu=mu, phi=phi).cdf(days) for days in range(0,total_days+1,1)]
        #distros[mu] = [st.lognorm.cdf(days, s=0.5, loc=mu, scale=0.5) for days in range(0,31,1)]
        #distros[mu] = [st.expon.cdf(days, loc=mu, scale=0.01) for days in range(0,31,1)]
        #distros[mu] = [st.gengamma.cdf(days, loc=mu, scale=1, a=mu, c=1) for days in range(1,31,1)]
        if np.sum(distros[mu]) > 0:
            distros[mu] = np.diff(distros[mu])
            distros[mu] = np.round(distros[mu] / np.sum(distros[mu]), 4)
        else:
            distros[mu] = distros[mu][1:]
            distros[mu][-1] = 1
        
    
    prob_array = np.zeros((pred_list.shape[0], total_days))

    for row, mu in enumerate(pred_list):
        prob_array[row, :] = distros[mu]#.cumsum()
        #prob_array[row, -1] = 1.

    return prob_array

def pred_list_to_distro_smooth(pred_list, total_days=30, phi=2, power=1.5, smooth_factor=0.3):
    distros = dict()
    for mu in range(1,total_days+1):
        distros[mu] = [tweedie.tweedie(p=power, mu=mu, phi=phi).cdf(days) for days in range(0,total_days+1,1)]
        if np.sum(distros[mu]) > 0:
            distros[mu] = np.diff(distros[mu])
            distros[mu] = np.round(distros[mu] / np.sum(distros[mu]), 4)
        else:
            distros[mu] = distros[mu][1:]
            distros[mu][-1] = 1
        
    
    prob_array = np.zeros((pred_list.shape[0], total_days))

    for row, mu in enumerate(pred_list):
        if mu == 1:
            prob_array[row, :] = (1-smooth_factor)*distros[mu] + smooth_factor*distros[mu+1]
        elif mu == total_days:
            prob_array[row, :] = smooth_factor*distros[mu-1] + (1-smooth_factor)*distros[mu]
        else:
            prob_array[row, :] = (smooth_factor/2)*distros[mu-1] + (1-smooth_factor)*distros[mu] + (smooth_factor/2)*distros[mu+1]

    return prob_array