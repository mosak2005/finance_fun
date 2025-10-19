'''
code inspired by and based on article: https://medium.com/@polanitzer/building-a-garch-1-1-model-in-python-step-by-step-f8503e868efa

Enjoy!
'''


import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
from scipy.optimize import minimize

end = dt.datetime.now()
start = end - dt.timedelta(days=5*365)

df = yf.download('WIG.WA', start=start, end=end, progress=False)
df.rename(columns={'Close': 'Si'}, inplace=True)
df = df[['Si']]

df["ui"] = 0.0
for i in range(1, len(df)):
    df.iloc[i, df.columns.get_loc("ui")] = (df.iloc[i, df.columns.get_loc("Si")] - df.iloc[i-1, df.columns.get_loc("Si")]) / df.iloc[i-1, df.columns.get_loc("Si")]

V = np.var(df["ui"][1:])


df_opt = df[1:].copy().reset_index(drop=True)  
n = len(df_opt)


df_opt["vi"] = 0.0  
df_opt["L"] = 0.0   

def garch_likelihood(params, df_opt, V, n):
    df_copy = df_opt.copy()     
    w = params[0] * 0.00001
    a = params[1] * 0.1
    b = params[2]
    
    df_copy.loc[0, "vi"] = V  
    for i in range(1, n):
        df_copy.loc[i, "vi"] = w + a * (df_copy.loc[i-1, "ui"]**2) + b * df_copy.loc[i-1, "vi"]

    for i in range(1, n):
        vi = df_copy.loc[i, "vi"]
        df_copy.loc[i, "L"] = -np.log(vi) - (df_copy.loc[i, "ui"]**2 / vi)

    log_L = np.sum(df_copy["L"][1:])
    
    return -log_L

start_params = [0.001, 0.1, 0.8]
bnds = ((0.00000001, 1), (0.00000001, 1), (0.00000001, 1))
cons = ({'type': 'ineq', 'fun': lambda params: 1.0 - params[1]*0.1 - params[2]}) 

result = minimize(garch_likelihood, start_params, args=(df_opt, V, n), method='SLSQP', bounds=bnds, constraints=cons)
w_opt = result.x[0] * 0.00001
a_opt = result.x[1] * 0.1
b_opt = result.x[2]

print(f"w: {w_opt}")
print(f"a: {a_opt}")
print(f"b: {b_opt}")
VLT = w_opt / (1 - a_opt - b_opt) if (1 - a_opt - b_opt) > 0 else np.inf
print(f"long term volotility: {np.sqrt(VLT):.8f} % ")