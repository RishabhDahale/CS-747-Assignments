import pandas as pd
import statistics


data = pd.read_csv('../outputDataT2-v2.csv')
regts = []
regtsh = []

algos = ['thompson-sampling', 'thompson-sampling-with-hint']
horizons = [1000, 2000, 5000]

for h in horizons:
    chunk = data.loc[(data['algo']==algos[0]) & (data['horizon']==h)]
    regts.append(statistics.mean(data['reg']))

for h in horizons:
    chunk = data.loc[(data['algo']==algos[1]) & (data['horizon']==h)]
    regtsh.append(statistics.mean(data['reg']))

print(regts[0])
print(regtsh[0])
