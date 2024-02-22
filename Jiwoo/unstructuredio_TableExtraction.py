import docx
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
import os
from pathlib import Path
import pandas as pd

print(os.getcwd())
data_folder = Path(os.getcwd()).parent / 'Financial_Diaries'
glob_pattern = '**/*.pdf'
filepath_gen = data_folder.glob(glob_pattern)
file = filepath_gen.__next__()
print(file.parent)

#%%

# fname = '../Financial_Diaries/Baseline Financial Diaries/Taraba/FDs/BALI/Bali non WAG/FINANCIAL DIARY FOR NON WAG MEMBER BALI LGA (URBAN) 8122021pdf.pdf'
filename = '../Financial_Diaries/Baseline Financial Diaries/Taraba/FDs/BALI/Bali non WAG/FINANCIAL DIARY FOR NON WAG MEMBER BALI LGA (URBAN) 8122021.docx'
elements = partition(filename=filename)

tables = [el for el in elements if el.category == "Table"]
print(tables[0].metadata.text_as_html)

html_table = tables[0].metadata.text_as_html
table1 = pd.read_html(html_table, header=None)[0]
print(table1)
