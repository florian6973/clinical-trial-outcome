import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# csv = pd.read_csv('outcomes.txt', sep='|')
outcomes = pd.read_csv('outcomes.txt', sep='|')

outcomes['length'] = outcomes['title'].apply(lambda x: len(x))
outcomes.sort_values('length', ascending=False, inplace=True)

rows = outcomes['title']

print(len(rows))
rows = rows.apply(lambda x: x.lower())
rows = rows.drop_duplicates()
rows = rows.reset_index(drop=True)
print(len(rows))
print(rows)


rows_length = rows.apply(lambda x: len(x))
print(rows_length.describe())
rows_length.plot.hist(density=True)
plt.xlabel("Number of characters")
plt.tight_layout()
plt.savefig('test.png')

print(rows)

plt.figure()
rows_words = rows.apply(lambda x: len(x.split(' ')))
print(rows_words.describe())
rows_words.plot.hist(density=True)
plt.xlabel("Number of words")
plt.tight_layout()
plt.savefig('test-words.png')


rows.iloc[:1300].to_csv('check.csv')

random_rows = rows.sample(n=200, random_state=0)
random_rows = random_rows.reset_index(drop=True)
print(random_rows)
random_rows.to_csv('train_raw.csv')


