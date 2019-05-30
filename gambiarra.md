import numpy
import pandas as pd

y =  [1.85,1.90,1.98,1.90,2.00,2.00,2.00,1.90,1.90,1.89]
y_pred       = [1.90,1.95,2.00,1.91,2.03,2.05,2.03,1.96,1.91,1.90]


df_pricing = pd.DataFrame({'y':y,'y_pred':y_pred})

df_pricing['residuo'] = df_pricing.y - df_pricing.y_pred

df_pricing

n = df_pricing.shape[0]
n

nok = df_pricing[df_pricing.residuo > 0].shape[0]
nok

## até agora temos 30 % de acertividade
nok/n

# a meta é chegar em 80%
notok = df_pricing[df_pricing.residuo < 0]
notok


from random import random
from scipy.optimize import minimize,minimize_scalar
import numpy as np


def objective(teta,y,y_pred):
    cnt = 0
    n = len(y)
    for i in range(0,n):
        if((y_pred[i]-teta) <= y[i]):
            cnt = cnt+1
    return cnt/n


objective(0.04,df_pricing.y.values,df_pricing.y_pred.values)




 
