import random
import pandas as pd
import numpy as np

csv_path = 'syn_pat.csv'

df = pd.read_csv(csv_path)

# Drop dead pat
df = df[df['DEATHDATE'].isnull()]

# Define day ranges
days_in_hosp_range = [np.arange(0,6), np.arange(6,11), np.arange(11,16), np.arange(16,21), np.arange(21,26)]
# Define probability of picking from above ranges for k samples
choices = random.choices(days_in_hosp_range, weights=(75, 10, 7, 5, 3), k=5000)
days_in_hosp = []
for choice in choices:
    days_in_hosp.append(random.choice(choice))

# Define day ranges
days_in_icu_range = [[0], np.arange(1,4), np.arange(4,6), np.arange(6,10)]
# Define probability of picking from above ranges for k samples
choices = random.choices(days_in_icu_range, weights=(85, 10, 3, 2), k=5000)
days_in_icu = []
for choice in choices:
    days_in_icu.append(random.choice(choice))

df['Days in Hospital'] = days_in_hosp
df['Days in ICU'] = days_in_icu

df.to_csv('syn_pat_target.csv', index=False)