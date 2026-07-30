import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def test_track_nctid():
    outcomes = pd.read_csv('../../snomed/raw/outcomes.txt', sep='|')
    outcomes['length'] = outcomes['title'].apply(lambda x: len(x))
    outcomes.sort_values('length', ascending=False, inplace=True)
    print(outcomes.columns)
    rows = outcomes[['title', 'id', 'nct_id']]
    print(len(rows))
    rows.loc[:, 'title'] = rows['title'].apply(lambda x: x.lower())
    # rows = rows.drop_duplicates()
    rows = rows.reset_index(drop=True)
    print(len(rows))
    print(rows)

def read_outcomes():
    # csv = pd.read_csv('outcomes.txt', sep='|')
    outcomes = pd.read_csv('../../snomed/raw/outcomes.txt', sep='|')

    outcomes['length'] = outcomes['title'].apply(lambda x: len(x))
    outcomes.sort_values('length', ascending=False, inplace=True)
    outcomes.set_index('id')
    rows = outcomes['title']

    print(len(rows))
    rows = rows.apply(lambda x: x.lower())
    rows.value_counts()[:1000].to_csv('natural_normalization.csv')
    print(rows.value_counts())
    rows = rows.drop_duplicates()
    ids = rows.index
    rows = rows.reset_index(drop=True)
    print(ids)

    return ids, rows

if __name__ == '__main__':
    test_track_nctid()
    exit()

    ids, rows = read_outcomes()
    
    print(len(ids), len(rows))
    print(rows)
    exit()


    rows['length'] = rows.apply(lambda x: len(x))
    print(rows['length'].describe())
    rows['length'].plot.hist()
    plt.savefig('test.png')

    rows.iloc[:1300].to_csv('check.csv')