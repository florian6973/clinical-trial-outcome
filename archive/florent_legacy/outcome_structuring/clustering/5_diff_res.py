import pandas as pd

def compare_csv_files(file1, file2, output_differences_file):
    # Read the two CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Compare rows in both DataFrames
    diff1 = df1[~df1.isin(df2).all(axis=1)]  # Rows in df1 but not in df2
    diff2 = df2[~df2.isin(df1).all(axis=1)]  # Rows in df2 but not in df1

    # Combine differences
    differences = pd.concat([diff1, diff2]).drop_duplicates()

    # Output differences to a new CSV file
    differences.to_csv(output_differences_file, index=False)
    print(f"Differences saved to {output_differences_file}")

# Example usage
file1 = 'results_cls-without.csv'
file2 = 'results_cls.csv'
output_differences_file = 'differences.csv'

compare_csv_files(file1, file2, output_differences_file)