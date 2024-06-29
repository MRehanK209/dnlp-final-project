import argparse
import random

import numpy as np
import pandas as pd
import torch
from sacrebleu.metrics import BLEU
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration

from optimizer import AdamW

TQDM_DISABLE = False


def transform_data(dataset, tokenizer, max_length=256, batch_size=16, data_type='train'):
    """
    Turn the data to the format you want to use.
    Use AutoTokenizer to obtain encoding (input_ids and attention_mask).
    Tokenize the sentence pair in the following format:
    sentence_1 + SEP + sentence_2 segment location + SEP + paraphrase types.
    Return DataLoader.
    """
    inputs = []
    targets = []

    for idx, row in dataset.iterrows():
        if data_type in ('train', 'dev'):
            input_text = row['sentence1']
            target_text = row['sentence2']
            inputs.append(input_text)
            targets.append(target_text)
        else:
            input_text = row['sentence1']
            inputs.append(input_text)

    encodings = tokenizer(inputs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    if data_type in ('train', 'dev'):
        target_encodings = tokenizer(targets, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        dataset = TensorDataset(encodings.input_ids, encodings.attention_mask, target_encodings.input_ids)
    else:
        dataset = TensorDataset(encodings.input_ids, encodings.attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def train_model(model, train_data, dev_data, device, tokenizer, num_epochs=1, lr=5e-5):
    """
    Train the model. Return and save the model.
    """
    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in tqdm(train_data, desc=f"Training Epoch {epoch + 1}", disable=TQDM_DISABLE):
            optimizer.zero_grad()

            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1} finished. Training loss: {epoch_loss / len(train_data):.4f}")

    return model


def test_model(test_data, test_ids, device, model, tokenizer):
    """
    Test the model. Generate paraphrases for the given sentences (sentence1) and return the results
    in form of a Pandas dataframe with the columns 'id' and 'Generated_sentence2'.
    The data format in the columns should be the same as in the train dataset.
    Return this dataframe.
    """
    model.eval()
    generated_sentences = []

    with torch.no_grad():
        for batch in tqdm(test_data, desc="Generating paraphrases", disable=TQDM_DISABLE):
            input_ids, attention_mask = [b.to(device) for b in batch]

            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=5,
                early_stopping=True,
            )

            generated_texts = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in outputs
            ]
            generated_sentences.extend(generated_texts)

    results = pd.DataFrame({'id': test_ids, 'Generated_sentence2': generated_sentences})
    return results


def evaluate_model(model, test_data, device, tokenizer):
    """
    You can use your train/validation set to evaluate models performance with the BLEU score.
    """
    model.eval()
    bleu = BLEU()
    predictions = []
    references = []

    with torch.no_grad():
        for batch in test_data:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Generate paraphrases
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=5,
                early_stopping=True,
            )

            pred_text = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in outputs
            ]
            ref_text = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in labels
            ]

            predictions.extend(pred_text)
            references.extend([[r] for r in ref_text])

    model.train()

    # Calculate BLEU score
    bleu_score = bleu.corpus_score(predictions, references)
    return bleu_score.score


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()
    return args


def finetune_paraphrase_generation(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

    train_dataset = pd.read_csv("data/processed_etpc_paraphrase_train.csv", on_bad_lines='warn', sep="\t")
    dev_dataset = pd.read_csv("data/processed_etpc_paraphrase_dev.csv", on_bad_lines='warn', sep="\t")
    test_dataset = pd.read_csv("data/processed_etpc_paraphrase_generation_test_student.csv", on_bad_lines='warn',
                               sep="\t")

    # You might do a split of the train data into train/validation set here
    # ...

    train_data = transform_data(train_dataset, tokenizer, data_type='train')
    dev_data = transform_data(dev_dataset, tokenizer, data_type='dev')
    test_data = transform_data(test_dataset, tokenizer, data_type='test')

    print(f"Loaded {len(train_dataset)} training samples.")

    model = train_model(model, train_data, dev_data, device, tokenizer)

    print("Training finished.")

    bleu_score = evaluate_model(model, dev_data, device, tokenizer)
    print(f"The BLEU-score of the model is: {bleu_score:.3f}")

    test_ids = test_dataset["id"]
    test_results = test_model(test_data, test_ids, device, model, tokenizer)
    test_results.to_csv(
        "predictions/bart/etpc-paraphrase-generation-test-output.csv", index=False, sep="\t"
    )


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_generation(args)
