import pandas as pd
import os
from pathlib import Path
import re

print(os.getcwd())
dta_folder = Path(os.getcwd()).parent / 'NFWP_BaselineData_WomenMen_forVicki_March10_2023'
glob_patter = '**/*.dta'
dta_files = dta_folder.glob(glob_patter)
# dta_file = dta_files.__next__()

def stata2csv(dta_file):
    '''
    :param dta_file: .dta files in specific folder
    :return: csv files
    '''
    data = pd.read_stata(dta_file)
    pattern = re.compile(r'\.dta$')
    filename = dta_file.name
    new_filename = re.sub(pattern, '.csv', filename)
    data.to_csv(new_filename)

try:
    while True:
        dta_file = dta_files.__next__()
        stata2csv(dta_file)
except StopIteration:
    pass

# data = pd.read_stata("../NFWP_BaselineData_WomenMen_forVicki_March10_2023/frmt_women_combined_outcomes_mar10_2023.dta")
# data.to_csv('frmt_women_combined_outcomes_mar10_2023.csv')
