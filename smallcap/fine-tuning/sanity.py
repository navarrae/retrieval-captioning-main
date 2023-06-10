import pandas
from pandas import *

data = read_csv("datastore/mimic_test_reports.csv")
target = data['target'].tolist()
impression = data['impression'].tolist()
findings = data['findings'].tolist()

print('target:', target[4])
print('impression:', impression[4])
print('findings:', findings[4])