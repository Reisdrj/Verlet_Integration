import pandas as pd 

file = 'data/HIST_PAINEL_COVIDBR_2020_Parte1_02dez2023.csv'

df = pd.read_csv(file, on_bad_lines = 'skip')