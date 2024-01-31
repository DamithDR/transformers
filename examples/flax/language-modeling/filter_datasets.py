import argparse

import pandas as pd
from datasets import load_dataset

from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='''convert dataset''')
parser.add_argument('--cache_dir', type=str, default=None, required=False,
                    help='model_name')
args = parser.parse_args()

config = 'en_all'
if args.cache_dir:
    dataset = load_dataset('joelito/Multi_Legal_Pile', config, split='train', cache_dir=args.cache_dir, streaming=True)
else:
    dataset = load_dataset('joelito/Multi_Legal_Pile', config, split='train', cache_dir=args.cache_dir, streaming=True)

filtered_list = []
part_number = 1
count = 0
removed = 0
for data in dataset:
    if len(data['text'].split(" ")) <= 1024:
        count += 1
        filtered_list.append(data['text'])
        if len(filtered_list) == 500000:
            df = pd.DataFrame()
            df['text'] = filtered_list
            df.to_csv(f'en_all_filtered_1024_part_{part_number}.tsv', sep='\t', index=False)
            part_number += 1
            filtered_list = []
        print(f'processing part number : {part_number} | count number : {count}')
    else:
        removed += 1
df = pd.DataFrame()
df['text'] = filtered_list
df.to_csv(f'data_files/en_all_filtered_1024_part_{part_number}.tsv', sep='\t', index=False)
print(f'used count {count} | removed count {removed}')

files_list = [f'en_all_filtered_1024_part_{i}.tsv' for i in range(1, 18)]
dataframes = [pd.read_csv(f'data_files/{file}', sep='\t') for file in files_list]
df = pd.concat(dataframes, ignore_index=True)
df = df.dropna(how='all')

train, validation = train_test_split(df, test_size=0.1, random_state=777, shuffle=True)

train.to_csv('training_file.tsv', sep='\t', index=False)
validation.to_csv('validation_file.tsv', sep='\t', index=False)

# used count 8377505 | removed count 9859559 - missed last part
# used count 8377505 | removed count 9859559 - add them together to get total


