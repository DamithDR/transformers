import argparse
import time
from datetime import timedelta

import datasets

from t5_tokenizer_model import SentencePieceUnigramTokenizer
from transformers import T5Config


def run(args):
    vocab_size = 32_000
    input_sentence_size = None

    # Initialize a dataset
    if args.data_config:
        dataset = datasets.load_dataset(args.dataset, name=args.data_config, split="train")
    else:
        dataset = datasets.load_dataset(args.dataset, split="train")

    tokenizer = SentencePieceUnigramTokenizer(unk_token="<unk>", eos_token="</s>", pad_token="<pad>")

    # Build an iterator over this dataset
    def batch_iterator(input_sentence_size=None):
        if input_sentence_size is None:
            input_sentence_size = len(dataset)
        batch_length = 100
        for i in range(0, input_sentence_size, batch_length):
            yield dataset[i: i + batch_length][args.field]

    start = time.time()
    # Train tokenizer
    tokenizer.train_from_iterator(
        iterator=batch_iterator(input_sentence_size=input_sentence_size),
        vocab_size=vocab_size,
        show_progress=True,
    )

    end = time.time()

    time_taken_timedelta = timedelta(seconds=end - start)

    # Print the result in a human-readable format
    print(f'Time taken: {time_taken_timedelta}')

    # Save files to disk
    tokenizer.save("./LegalT5-base/tokenizer.json")

    # from transformers imporLegalT5-baset T5Config

    config = T5Config.from_pretrained("google/t5-v1_1-base", vocab_size=tokenizer.get_vocab_size())
    config.save_pretrained("./LegalT5-base")

    while True:
        print("process finished")
        time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''evaluates models on legal instruction finetuning''')
    parser.add_argument('--dataset', type=str, required=True, help='dataset_name')
    parser.add_argument('--data_config', type=str, required=False, help='data_config')
    parser.add_argument('--field', type=str, required=False, help='field')
    args = parser.parse_args()
    run(args)
