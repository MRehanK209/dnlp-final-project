import argparse
import random
import numpy as np
import pandas as pd
import torch
from sacrebleu.metrics import BLEU
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration
import torch.optim.lr_scheduler as lr_scheduler
from transformers import get_linear_schedule_with_warmup
from optimizer import AdamW
from nltk.corpus import wordnet
import torch.nn.functional as F

TQDM_DISABLE = False


def augment_sentence_with_synonyms(sentence, min_n=1, max_n=5):
    """
    Replace 'n' words in the sentence with their synonyms to create augmented data.
    Randomly select n between min_n and max_n for each sentence.
    """
    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set(words))
    random.shuffle(random_word_list)

    num_replaced = 0
    n = random.randint(min_n, max_n)

    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    return ' '.join(new_words)


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    if word in synonyms:
        synonyms.remove(word)
    return synonyms


def augment_dataset_with_original(dataset, augment_factor=1):
    augmented_data = []
    for idx, row in dataset.iterrows():
        original_sentence = row['sentence1']
        for _ in range(augment_factor):
            augmented_sentence = augment_sentence_with_synonyms(original_sentence)
            augmented_row = row.copy()
            augmented_row['sentence1'] = augmented_sentence
            augmented_data.append(augmented_row)
    augmented_df = pd.DataFrame(augmented_data)
    return pd.concat([dataset, augmented_df], ignore_index=True)


def transform_data(dataset, max_length=256, batch_size=64, shuffle=True):
    inputs = []
    targets = []

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)

    for idx, row in dataset.iterrows():
        input_text = row['sentence1']
        inputs.append(input_text)

        if 'sentence2' in row:  # for train/dev data
            target_text = row['sentence2']
            targets.append(target_text)

    encodings = tokenizer(inputs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    if targets:
        target_encodings = tokenizer(targets, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        dataset = TensorDataset(encodings.input_ids, encodings.attention_mask, target_encodings.input_ids)
    else:
        dataset = TensorDataset(encodings.input_ids, encodings.attention_mask)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def contrastive_loss(input_ids, generated_ids, model, device):
    """
    Implement contrastive loss to penalize when the generated sentence is too similar to the input sentence.
    """
    input_embeds = model.get_input_embeddings()(input_ids).mean(dim=1)
    generated_embeds = model.get_input_embeddings()(generated_ids).mean(dim=1)

    # Calculate cosine similarity between input and generated embeddings
    cosine_sim = F.cosine_similarity(input_embeds, generated_embeds, dim=-1)

    loss = torch.mean(cosine_sim)
    return loss


def train_model(model, train_data, dev_data, device, num_epochs=20, lr=3e-5, scheduler_type='cosine',
                contrastive_weight=0.3, early_stopping_patience=3):
    """
    Train the model with early stopping based on validation performance.

    Args:
        model (BartForConditionalGeneration): Pretrained BART model.
        train_data (DataLoader): DataLoader for training data.
        dev_data (DataLoader): DataLoader for validation data.
        device (torch.device): Device on which to perform computations.
        num_epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        scheduler_type (str): Type of learning rate scheduler.
        contrastive_weight (float): Weight for the contrastive loss component.
        early_stopping_patience (int): Number of epochs to wait for improvement before stopping.

    Returns:
        BartForConditionalGeneration: Trained model.
    """
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.005)

    if scheduler_type == 'reduce_on_plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    elif scheduler_type == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    elif scheduler_type == 'warmup':
        total_steps = len(train_data) * num_epochs
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
    else:
        scheduler = None

    model.train()

    best_score = float('-inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in tqdm(train_data, desc=f"Training Epoch {epoch + 1}", disable=TQDM_DISABLE):
            optimizer.zero_grad()

            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=6,
                num_beam_groups=2,
                diversity_penalty=1.0
            )
            contrastive = contrastive_loss(input_ids, generated_ids, model, device)

            loss += contrastive_weight * contrastive

            loss.backward()
            optimizer.step()

            if scheduler and scheduler_type != 'reduce_on_plateau':
                scheduler.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1} finished. Training loss: {epoch_loss / len(train_data):.4f}")

        penalized_bleu = evaluate_model(model, dev_data, device)
        print(f"Penalized BLEU Score on validation data: {penalized_bleu:.3f}")

        if scheduler_type == 'reduce_on_plateau':
            scheduler.step(penalized_bleu)

        if penalized_bleu > best_score:
            best_score = penalized_bleu
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs. Best penalized BLEU score: {best_score:.3f}")
            model.load_state_dict(torch.load('best_model.pt'))
            break

    return model


def test_model_with_diverse_beam_search(test_data, test_ids, device, model, num_beams=6, num_beam_groups=2,
                                        diversity_penalty=1.5):
    model.eval()
    generated_sentences = []

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)

    with torch.no_grad():
        for batch in tqdm(test_data, desc="Generating paraphrases", disable=TQDM_DISABLE):
            input_ids, attention_mask = [b.to(device) for b in batch]

            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=num_beams,
                num_beam_groups=num_beam_groups,
                diversity_penalty=diversity_penalty,
                early_stopping=True
            )

            generated_texts = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in outputs
            ]
            generated_sentences.extend(generated_texts)

    results = pd.DataFrame({'id': test_ids, 'Generated_sentence2': generated_sentences})
    return results


def evaluate_model(model, test_data, device):
    model.eval()
    bleu = BLEU()
    predictions = []

    dataloader = transform_data(test_data, shuffle=False)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=6,
                num_beam_groups=2,
                diversity_penalty=1.0,
                early_stopping=True
            )

            pred_text = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in outputs
            ]

            predictions.extend(pred_text)

    inputs = test_data["sentence1"].tolist()
    references = test_data["sentence2"].tolist()

    model.train()
    bleu_score_reference = bleu.corpus_score(references, [predictions]).score
    bleu_score_inputs = 100 - bleu.corpus_score(inputs, [predictions]).score

    print(f"BLEU Score: {bleu_score_reference}", f"Negative BLEU Score with input: {bleu_score_inputs}")

    penalized_bleu = bleu_score_reference * bleu_score_inputs / 52
    print(f"Penalized BLEU Score: {penalized_bleu}")

    return penalized_bleu


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
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", local_files_only=True)
    model.to(device)

    train_dataset = pd.read_csv("data/processed_etpc_paraphrase_train.csv", sep="\t")
    dev_dataset = pd.read_csv("data/processed_etpc_paraphrase_dev.csv", sep="\t")
    test_dataset = pd.read_csv("data/etpc-paraphrase-generation-test-student.csv", sep="\t")

    train_dataset = augment_dataset_with_original(train_dataset, augment_factor=3)

    train_data = transform_data(train_dataset, shuffle=True)
    dev_data = dev_dataset
    test_data = transform_data(test_dataset, shuffle=False)

    print(f"Loaded {len(train_dataset)} training samples.")

    scheduler_type = 'warmup'

    model = train_model(model, train_data, dev_data, device, scheduler_type=scheduler_type)

    print("Training finished.")

    bleu_score = evaluate_model(model, dev_data, device)
    print(f"The penalized BLEU-score of the model is: {bleu_score:.3f}")

    test_ids = test_dataset["id"]
    test_results = test_model_with_diverse_beam_search(test_data, test_ids, device, model)
    test_results.to_csv("predictions/bart/etpc-paraphrase-generation-test-output.csv", index=False, sep="\t")


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_generation(args)
