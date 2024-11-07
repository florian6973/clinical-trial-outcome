import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_outcomes():
    # csv = pd.read_csv('outcomes.txt', sep='|')
    outcomes = pd.read_csv('outcomes.txt', sep='|')

    outcomes['length'] = outcomes['title'].apply(lambda x: len(x))
    outcomes.sort_values('length', ascending=False, inplace=True)

    rows = outcomes['title']

    print(len(rows))
    rows = rows.apply(lambda x: x.lower())
    rows = rows.drop_duplicates()
    rows = rows.reset_index(drop=True)

    return rows

if __name__ == '__main__':
    rows = read_outcomes()
    
    print(len(rows))
    print(rows)


    rows['length'] = rows.apply(lambda x: len(x))
    print(rows['length'].describe())
    rows['length'].plot.hist()
    plt.savefig('test.png')

    rows.iloc[:1300].to_csv('check.csv')