import argparse
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartModel
from optimizer import AdamW


TQDM_DISABLE = False


class BartWithClassifier(nn.Module):
    def __init__(self, num_labels=7):
        super(BartWithClassifier, self).__init__()

        self.bart = BartModel.from_pretrained("facebook/bart-large", local_files_only=True)
        self.classifier = nn.Linear(self.bart.config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None):
        # Use the BartModel to obtain the last hidden state
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]

        # Add an additional fully connected layer to obtain the logits
        logits = self.classifier(cls_output)

        # Return the probabilities
        probabilities = self.sigmoid(logits)
        return probabilities


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=int, default=1e-5)
    args = parser.parse_args()
    return args


def accuracy_binary(predicted_labels_np,true_labels_np):
    accuracies = []
    for label_idx in range(true_labels_np.shape[1]):
        correct_predictions = np.sum(
            true_labels_np[:, label_idx] == predicted_labels_np[:, label_idx]
        )
        total_predictions = true_labels_np.shape[0]
        label_accuracy = correct_predictions / total_predictions
        accuracies.append(label_accuracy)
    return np.mean(accuracies)


def transform_data(dataset, max_length=512):
    """
    dataset: pd.DataFrame

    Turn the data to the format you want to use.

    1. Extract the sentences from the dataset. We recommend using the already split
    sentences in the dataset.
    2. Use the AutoTokenizer from_pretrained to tokenize the sentences and obtain the
    input_ids and attention_mask.
    3. Currently, the labels are in the form of [2, 5, 6, 0, 0, 0, 0]. This means that
    the sentence pair is of type 2, 5, and 6. Turn this into a binary form, where the
    label becomes [0, 1, 0, 0, 1, 1, 0]. Be careful that the test-student.csv does not
    have the paraphrase_types column. You should return a DataLoader without the labels.
    4. Use the input_ids, attention_mask, and binary labels to create a TensorDataset.
    Return a DataLoader with the TensorDataset. You can choose a batch size of your
    choice.
    """
    sentences1 = dataset['sentence1'].tolist()
    sentences2 = dataset['sentence2'].tolist()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    encodings = tokenizer(sentences1, sentences2, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    if 'paraphrase_types' in dataset.columns:
        binary_labels = dataset['paraphrase_types'].apply(lambda x: [1 if int(i) != 0 else 0 for i in x.strip('[]').split(', ')])
        binary_labels = torch.tensor(binary_labels.tolist())
        dataset = TensorDataset(input_ids, attention_mask, binary_labels)

    else :
        dataset = TensorDataset(input_ids, attention_mask)
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader


def train_model(model, train_data, dev_data, device, args):
    """
    Train the model. You can use any training loop you want. We recommend starting with
    AdamW as your optimizer. You can take a look at the SST training loop for reference.
    Think about your loss function and the number of epochs you want to train for.
    You can also use the evaluate_model function to evaluate the
    model on the dev set. Print the training loss, training accuracy, and dev accuracy at
    the end of each epoch.

    Return the trained model.
    """
    ### TODO

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        all_pred = []
        all_labels = []
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(train_data, desc=f"Training Epoch {epoch + 1}", disable=TQDM_DISABLE):
            optimizer.zero_grad()

            input_ids, attention_mask, labels = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids,attention_mask)
            loss = criterion(logits, labels.float())

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

            predicted_labels = (logits > 0.5).int()
            all_pred.append(predicted_labels)
            all_labels.append(labels)

        all_predictions = torch.cat(all_pred, dim=0)
        all_true_labels = torch.cat(all_labels, dim=0)

        true_labels_np = all_true_labels.cpu().numpy()
        predicted_labels_np = all_predictions.cpu().numpy()

        training_acc = accuracy_binary(predicted_labels_np,true_labels_np)
        epoch_train_loss = train_loss/num_batches
        print(f"Training Epoch {epoch + 1} Training Loss: {epoch_train_loss:.4f}")
        print(f"Training Epoch {epoch + 1} Training Accuracy: {training_acc:.4f}")
        validation_acc = evaluate_model(model, dev_data, device)
        print(f"Training Epoch {epoch + 1} Validation Accuracy: {validation_acc:.4f}")

    return model


def test_model(model, test_data, test_ids, device):
    """
    Test the model. Predict the paraphrase types for the given sentences and return the results in form of
    a Pandas dataframe with the columns 'id' and 'Predicted_Paraphrase_Types'.
    The 'Predicted_Paraphrase_Types' column should contain the binary array of your model predictions.
    Return this dataframe.
    """
    ### TODO
    model.eval()  # Set model to evaluation mode
    all_predictions = []

    with torch.no_grad():
        for batch in test_data:
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            logits = model(input_ids, attention_mask=attention_mask)
            predictions = (logits > 0.5).int()
            all_predictions.append(predictions.cpu().numpy())

    all_predictions = [item for sublist in all_predictions for item in sublist]

    result_df = pd.DataFrame({
        'id': test_ids,
        'Predicted_Paraphrase_Types': all_predictions
    })

    return result_df
    

def evaluate_model(model, test_data, device):
    """
    This function measures the accuracy of our model's prediction on a given train/validation set
    We measure how many of the seven paraphrase types the model has predicted correctly for each data point.
    So, if the models prediction is [1,1,0,0,1,1,0] and the true label is [0,0,0,0,1,1,0], this predicition
    has an accuracy of 5/7, i.e. 71.4% .
    """
    all_pred = []
    all_labels = []
    model.eval()

    with torch.no_grad():
        for batch in test_data:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predicted_labels = (outputs > 0.5).int()

            all_pred.append(predicted_labels)
            all_labels.append(labels)

    all_predictions = torch.cat(all_pred, dim=0)
    all_true_labels = torch.cat(all_labels, dim=0)

    true_labels_np = all_true_labels.cpu().numpy()
    predicted_labels_np = all_predictions.cpu().numpy()

    accuracy1 = accuracy_binary(predicted_labels_np,true_labels_np)
    return accuracy1


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def finetune_paraphrase_detection(args):
    model = BartWithClassifier()
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model = model.to(device)

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv", sep="\t")
    test_dataset = pd.read_csv("data/etpc-paraphrase-detection-test-student.csv", sep="\t")

    # TODO You might do a split of the train data into train/validation set here
    # (or in the csv files directly)

    train_dataset = train_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    train_ratio = 0.7
    train_size = int(train_ratio * len(train_dataset))
    train_df = train_dataset[:train_size]
    dev_df = train_dataset[train_size:]
    
    train_data = transform_data(train_df)
    dev_data = transform_data(dev_df)
    test_data = transform_data(test_dataset)

    print(f"Loaded {len(train_dataset)} training samples.")

    model = train_model(model, train_data, dev_data, device,args)

    print("Training finished.")

    accuracy = evaluate_model(model, dev_data, device)
    print(f"The accuracy of the model is: {accuracy:.3f}")

    test_ids = test_dataset["id"]
    test_results = test_model(model, test_data, test_ids, device)
    test_results.to_csv(
        "predictions/bart/etpc-paraphrase-detection-test-output.csv", index=False, sep="\t"
    )

    
if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_detection(args)
