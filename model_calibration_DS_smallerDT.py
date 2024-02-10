import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d

df = pd.read_csv("calibration_data/cali_14_332.csv")
area = df["Area"][0]
river = df["River"][0]
obs_ppt = df["PPT"].dropna()
obs_pet = df["PET"].dropna()
obs_q = df["Q(m3/month)"].dropna()
ups_q = df["UPS_Q(m3/month)"].dropna()

obs_ts = np.arange(0, 276, 1)
ppt = interp1d(obs_ts, obs_ppt)
pet = interp1d(obs_ts, obs_pet)
obs_q_i = interp1d(obs_ts, obs_q)

ts_i = np.arange(0, 275.015625, 0.015625)
ppt = ppt(ts_i)
ppt = [round(i, 2) for i in ppt]
pet = pet(ts_i)
pet = [round(i, 2) for i in pet]
obs_q_i = obs_q_i(ts_i)


def performance(sim_q, obs_q):
    # remove nan
    obs_q_nonan = []
    sim_q_nonan = []
    for i in range(len(obs_q)):
        if obs_q[i] >= 0:
            obs_q_nonan.append(obs_q[i])
            sim_q_nonan.append(sim_q[i])
    obs_q_nonan = np.array(obs_q_nonan)
    sim_q_nonan = np.array(sim_q_nonan)

    r, _ = pearsonr(sim_q_nonan, obs_q_nonan)  # pearson correlation (r)

    # NSE
    mean_obs_q = np.mean(obs_q_nonan)
    nse = 1 - (np.sum((sim_q_nonan - obs_q_nonan) ** 2) / np.sum((obs_q_nonan - mean_obs_q) ** 2))

    # percentage deviation (dv)
    dv = (np.mean(sim_q_nonan) - np.mean(obs_q_nonan)) / np.mean(obs_q_nonan) * 100

    return r, nse, dv


# ----- differential evolution ------


def model_v2(ppt, pet, area, river, ups_q, ss_depth, x_tf, rc_tf, x_per, rc_per, gws_depth, x_bf, rc_bf, rc_q):
    ss_t = round((area * 1000000) * ss_depth * 0.25, 2)
    gws_t = round((area * 1000000) * gws_depth * 0.1, 2)
    cs_t = round(river, 2)
    sim_q = []

    max_ss = (area * 1000000) * ss_depth * 0.25
    min_ss_tf = (area * 1000000) * ss_depth * 0.25 * x_tf
    min_ss_per = (area * 1000000) * ss_depth * 0.25 * x_per
    min_gws = (area * 1000000) * gws_depth * 0.25 * x_bf

    for t in range(len(obs_q_i)):
        # print(t)
        # print("ss_t: " + str(ss_t))
        # print("cs_t: " + str(cs_t))
        ppt_t = (ppt[t] / 1000) * (area * 1000000)
        pet_t = (pet[t] / 1000) * (area * 1000000)
        if ss_t < pet_t:
            pet_t = ss_t
        # print("ppt: " + str(ppt_t))
        # print("pet: " + str(pet_t))

        if ss_t > max_ss:  # OLF
            olf = round(ss_t - max_ss, 2)
        else:
            olf = 0

        if ss_t > min_ss_tf:  # Throughflow (TF)
            tf = ss_t * rc_tf
        else:
            tf = 0

        if ss_t > min_ss_per:  # Percolation (PER)
            per = ss_t * rc_per
        else:
            per = 0

        # print(">> olf: " + str(olf))
        # print(">> tf: " + str(tf))
        # print(">> per: " + str(per))
        ss_t = round(ss_t + (ppt_t - pet_t - olf - tf - per) * 0.015625, 2)  # update SS for next timestep
        # print(">>>> delta ss: " + str((ppt_t - pet_t - olf - tf - per) * 0.125))

        if gws_t > min_gws:  # Baseflow (BF)
            bf = gws_t * rc_bf
        else:
            bf = 0
        # print(">> bf: " + str(bf))

        gws_t = round(gws_t + (per - bf) * 0.015625, 2)  # update GWS for next timestep

        q = (cs_t + ups_q[t]) * rc_q  # Discharge (Q)
        # print(">>>> q: " + str(q))

        # print(">> ups_q: " + str(ups_q[t]))
        cs_t = round(cs_t + (olf + tf + bf + ups_q[t] - q) * 0.015625, 2)  # update CS for next timestep
        # print(">>>> delta cs: " + str((olf + tf + bf + ups_q[t] - q) * 0.125))

        sim_q.append(q)
        # print("------")
        # input()

    return sim_q


def diff_ev(x):
    ss_depth, x_tf, rc_tf, x_per, rc_per, gws_depth, x_bf, rc_bf, rc_q = x
    sim_q = model_v2(ppt, pet, area, river, ups_q, ss_depth, x_tf, rc_tf, x_per, rc_per, gws_depth, x_bf, rc_bf, rc_q)
    sim_q = sim_q[::64]
    r, nse, dv = performance(sim_q, obs_q)
    print(nse)
    # print("r: " + str(r))
    return -nse


"""# ----- test model to see if it matches with STELLA -----
# model_v2(ppt, pet, area, river, ups_q, ss_depth, x_tf, rc_tf, x_per, rc_per, gws_depth, x_bf, rc_bf, rc_q)
test_q = model_v2(ppt, pet, area, river, ups_q, 1.95, 0.33, 0.5, 0.3203, 0.9811, 7.2561, 1.7, 0.2, 0.9921)
test_q = test_q[::64]
print(obs_q.tolist())
print(test_q)
plt.plot(obs_q, label="obs")
plt.plot(test_q, label="test")
plt.legend()
plt.show()
# --------------------------------------------------------"""


# ss_depth, x_tf, rc_tf, x_per, rc_per, gws_depth, x_bf, rc_bf, rc_q
bounds = [(0.1, 5), (0, 3), (0, 1), (0, 3), (0, 1), (4, 9), (0, 3), (0, 1), (0, 1)]
res = differential_evolution(diff_ev, bounds=bounds, maxiter=10000)
print("-----\ndifferential evolution:")
print(res)
opt_ss_depth, opt_x_tf, opt_rc_tf, opt_x_per, opt_rc_per, opt_gws_depth, opt_x_bf, opt_rc_bf, opt_rc_q = res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], res.x[5], res.x[6], res.x[7], res.x[8]
if res.success:
    print(f'ss_depth: {opt_ss_depth}\n'
          f'x_tf: {opt_x_tf}\n'
          f'rc_tf: {opt_rc_tf}\n'
          f'x_per: {opt_x_per}\n'
          f'rc_per: {opt_rc_per}\n'
          f'opt_gws_depth: {opt_gws_depth}\n'
          f'x_bf: {opt_x_bf}\n'
          f'rc_bf: {opt_rc_bf}\n'
          f'rc_q: {opt_rc_q}')
    opt_q = model_v2(ppt, pet, area, river, ups_q, opt_ss_depth, opt_x_tf, opt_rc_tf,
                     opt_x_per, opt_rc_per, opt_gws_depth, opt_x_bf, opt_rc_bf, opt_rc_q)
    fig1 = plt.figure("v1")
    plt.plot(obs_q_i, label="obs")
    plt.plot(opt_q, label="pred")
    plt.legend()
    plt.ylim(0, 4.8e9)
    plt.show()
else:
    print("failed")
