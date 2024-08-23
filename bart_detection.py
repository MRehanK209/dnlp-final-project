import argparse
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartModel
from sklearn.metrics import matthews_corrcoef
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=int, default=1e-5)
    parser.add_argument("--use_weight", type=str, default="None", choices=["None", "Deterministic","Fixed"])
    
    args = parser.parse_args()
    return args


def accuracy_binary(predicted_labels_np,true_labels_np):
    accuracies = []
    matthews_coefficients = []
    for label_idx in range(true_labels_np.shape[1]):
        correct_predictions = np.sum(
            true_labels_np[:, label_idx] == predicted_labels_np[:, label_idx]
        )
        total_predictions = true_labels_np.shape[0]
        label_accuracy = correct_predictions / total_predictions
        accuracies.append(label_accuracy)
        #compute Matthwes Correlation Coefficient for each paraphrase type
        matth_coef = matthews_corrcoef(true_labels_np[:,label_idx], predicted_labels_np[:,label_idx])
        matthews_coefficients.append(matth_coef)
    return np.mean(accuracies), np.mean(matthews_coefficients)

def convert_labels_to_binary(paraphrase_types, num_labels=7):
    binary_labels = []
    for types in paraphrase_types:
        binary_label = [0] * num_labels
        for t in eval(types):
            if t != 0:
                binary_label[t-1] = 1 
        binary_labels.append(binary_label)
    return binary_labels

def smoothness_inducing_loss(model, inputs, labels, criterion, epsilon=1e-6, lambda_reg=0.01):
    
    # Get the input embeddings from the BART model
    input_embeddings = model.bart.encoder.embed_tokens(inputs['input_ids'])
    
    # Add the small Gaussian noise to the embeddings
    noise = torch.randn_like(input_embeddings) * epsilon
    perturbed_embeddings = input_embeddings + noise

    # Forward pass with the original inputs
    original_output = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    
    # Forward pass with the perturbed embeddings
    perturbed_output = model.bart(
        inputs_embeds=perturbed_embeddings, 
        attention_mask=inputs['attention_mask'],
        decoder_input_ids=inputs['input_ids']  # or decoder_inputs_embeds=decoder_embeddings if you want to use embeddings
    )
    perturbed_output = model.classifier(perturbed_output.last_hidden_state[:, 0, :])
    perturbed_output = model.sigmoid(perturbed_output)

    original_loss = criterion(original_output, labels.float())
    
    # Calculate the smoothness regularization term (L2 norm)
    smoothness_loss = torch.mean(torch.norm(original_output - perturbed_output, p=2, dim=-1))
    
    # Combine the original loss with the smoothness regularization term
    total_loss = original_loss + lambda_reg * smoothness_loss
    
    return total_loss

def bregman_proximal_point_update(optimizer, model, original_params, eta=1e-5):

    # Apply the Bregman proximal point update
    with torch.no_grad():
        for param, original_param in zip(model.parameters(), original_params):
            param.copy_(param - eta * (param - original_param))

def transform_data(dataset, max_length=512, batch_size = 64):
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
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    encodings = tokenizer(sentences1, sentences2, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    if 'paraphrase_types' in dataset.columns:
        binary_labels = torch.tensor(convert_labels_to_binary(dataset["paraphrase_types"].tolist()))
        dataset = TensorDataset(input_ids, attention_mask, binary_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else :
        dataset = TensorDataset(input_ids, attention_mask)
        dataloader = DataLoader(dataset, batch_size=batch_size)
    
    return dataloader


def train_model(model, train_data, dev_data,weight_tensor, device, args):
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

    # criterion = FocalLoss()
    criterion = nn.BCELoss(weight = weight_tensor)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lambda_reg = 0.01
    eta = 1e-5
    for epoch in range(args.epochs):
        model.train() 
        total_loss = 0
        all_pred = []
        all_labels = []

        for batch in tqdm(train_data, desc=f"Training Epoch {epoch + 1}", disable=TQDM_DISABLE):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()

            original_params = [param.clone() for param in model.parameters()]
            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            loss = smoothness_inducing_loss(model, inputs, labels, criterion)
            loss.backward()
            optimizer.step()
            bregman_proximal_point_update(optimizer, model, original_params, eta=eta)

            total_loss += loss.item()
            all_pred.extend((model(**inputs) > 0.5).int().cpu().numpy())
            all_labels.extend(labels.int().cpu().numpy())

        avg_loss = total_loss / len(train_data)
        train_accuracy, train_mathhews_coefficient = accuracy_binary(np.array(all_pred), np.array(all_labels))
        validation_accuracy, validation_mathhews_coefficient = evaluate_model(model, dev_data, device)

        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"Training Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Training Mathew Coefficient: {train_mathhews_coefficient:.4f}")
        print(f"Validation Accuracy: {validation_accuracy:.4f}, Validation Mathew Coefficient: {validation_mathhews_coefficient:.4f}")
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

    accuracy1, mathhews_coefficient1 = accuracy_binary(predicted_labels_np,true_labels_np)
    return accuracy1,mathhews_coefficient1


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
    # model.bart.gradient_checkpointing_enable()

    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model = model.to(device)

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv", sep="\t")
    test_dataset = pd.read_csv("data/etpc-paraphrase-detection-test-student.csv", sep="\t")

    # TODO You might do a split of the train data into train/validation set here
    # (or in the csv files directly)

    np.random.seed(42)
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(np.arange(train_dataset.shape[0]))
    data_shuffled = train_dataset.iloc[shuffled_indices]
    train_size = int(train_ratio * len(data_shuffled))
    train_df = data_shuffled.iloc[:train_size]
    dev_df = data_shuffled.iloc[train_size:]

    train_data = transform_data(train_df)
    dev_data = transform_data(dev_df)
    test_data = transform_data(test_dataset)

    if args.use_weight == "Fixed":
        class_weights = [1.5,1.0,1.5,1.5,1.5,1.0,1.0]
        weight_tensor = torch.tensor(class_weights, dtype=torch.float)
        weight_tensor = weight_tensor.to(device) 
    elif args.use_weight == "Deterministic":
        class_frequencies = np.mean(convert_labels_to_binary(train_df["paraphrase_types"]), axis=0)
        inverse_class_frequencies = 1.0 / class_frequencies
        weight_tensor = torch.tensor(inverse_class_frequencies, dtype=torch.float)
        weight_tensor = weight_tensor.to(device) 
    elif args.use_weight == "None":
        weight_tensor = None

    print(f"Loaded {len(train_dataset)} training samples.")

    
    model = train_model(model, train_data, dev_data, weight_tensor, device,args)

    print("Training finished.")

    accuracy, matthews_corr = evaluate_model(model, train_data, device)
    print(f"The accuracy of the model is: {accuracy:.3f}")
    print(f"Matthews Correlation Coefficient of the model is: {matthews_corr:.3f}")

    test_ids = test_dataset["id"]
    test_results = test_model(model, test_data, test_ids, device)
    test_results.to_csv(
        "predictions/bart/etpc-paraphrase-detection-test-output.csv", index=False, sep="\t"
    )

    
if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_detection(args)

