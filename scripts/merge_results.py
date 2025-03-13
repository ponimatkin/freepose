import argparse
from pathlib import Path
import pandas as pd


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str)
    args.add_argument("--split", type=str, default='test')
    args = args.parse_args()

    results_path = Path('./data/results/').resolve() / args.dataset

    for folder in results_path.iterdir():
        # test if folder is json or csv
        if folder.is_file():
            continue
        
        csv_files = []
        for file in folder.iterdir():
            csv_df = pd.read_csv(file)
            # if dataframe is not empty
            if not csv_df.empty:
                csv_files.append(csv_df)

        df = pd.concat(csv_files)
        df = df.dropna()
        out_name = folder.name.replace(f'_{args.dataset}-{args.split}', '').replace('_', '-') + f'_{args.dataset}-{args.split}.csv'
        df.to_csv(out_name, header=True, index=False)