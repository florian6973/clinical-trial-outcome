file = '/nlp/projects/llama/outputs/top5_snomed_matches.csv'

import pandas as pd

csv = pd.read_csv(file)
print(csv)

# dask array

#                                                     Title  Match1_ConceptId  ... Match5_Depth  Match5_Similarity
# 0                           Change in Mean Cortisol Level         166478001  ...            4           0.536243
# 1       Change From Baseline in Low-Contrast Visual Ac...         170721002  ...            4           0.488271
# 2       Percentage of Participants With a Solicited Sy...         413378005  ...            2           0.506054
# 3                              Anti-mumps Antibody Titers         371112003  ...            2           0.609260
# 4       Double Blind Phase: Percentage of Participants...         170850000  ...            3           0.442935
# ...                                                   ...               ...  ...          ...                ...
# 438021  Part B: Titers of Anti-N SARS-CoV-2 Antibodies...         897034005  ...            6           0.518610
# 438022  Number of Participants With Potentially Clinic...         224994002  ...            4           0.476222
# 438023      Broberg Morrey Composite Elbow Function Score         128133004  ...            5           0.521342
# 438024  Headache Severity at Treatment, 30 Minutes and...          25064002  ...            3           0.535522
# 438025  Number of Local Adverse Reactions to 2-dose an...         420113004  ...            3           0.542915
# [438026 rows x 21 columns]