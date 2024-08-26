# DNLP SS23 Final Project - Multitask BERT and BART Fintune

  
### Group Tutor 
Yassir, Yassir

<div align="left">

<b> Algorithmic Advancement Alliances </b> <br/>

Karim, Injamam <br/>

Sorathiya, Smitesh <br/>

Ahmad, Saad <br/>

Khalid, Muhammad Rehan <br/>

Arsalane, Mohamed Reda <br/>

</div>

  

## Introduction

[![Final](https://img.shields.io/badge/Status-Final-purple.svg)](https://https://img.shields.io/badge/Status-Final-blue.svg)

[![Black Code Style](https://img.shields.io/badge/Code%20Style-Black-black.svg)](https://black.readthedocs.io/en/stable/)

[![AI-Usage Card](https://img.shields.io/badge/AI_Usage_Card-pdf-blue.svg)](./AI-Usage-Card.pdf/)

  

This repository is our official implementation of the Multitask BERT project and BART Fine Tuning for the Deep Learning for Natural Language

Processing course at the University of Göttingen.

## Setup Requirements

To install requirements and all dependencies and create the environment:


```sh

./setup.sh

```

If setting up the environment on GWDG, then run:

 ```sh

./setup_gwdg.sh

```

The environment is activated with `conda activate dnlp`.
  
The script will create a new conda environment called `dnlp` and install all required packages.

  
## BERT - Quora Question Pair Paraphrase Detection

### Introduction

This project focuses on building a robust model for paraphrase detection using the Quora Question Pairs dataset. The primary goal is to determine whether two given questions have the same intent or meaning, thereby detecting paraphrases. The task aims to improve the model's accuracy and reliability in differentiating between paraphrased and non-paraphrased question pairs. Various deep learning techniques, including BERT-based models, were employed, with fine-tuning and hyperparameter optimization used to achieve the best results.

### Setup Instructions

To set up the environment and run the model, follow these steps:

1. Run `./setwp_gwdf.sh` to install all the required dependencies and set up the environment.
2. Use `sbatch run_train_bert.sh` with 'qqp' or 'multitask' argument on a GPU-enabled cluster to initiate model training using Slurm.

The setup script installs necessary libraries such as TensorFlow, PyTorch, and HuggingFace Transformers. A GPU-enabled environment with CUDA support is recommended for faster training.

### Data

The project uses the Quora Question Pairs dataset, which is available in the `data` directory. The following datasets are utilized:

1. `quora-paraphrase-train.csv`: Contains 141,506 training question pairs along with labels indicating whether they are paraphrases.
2. `quora-paraphrase-dev.csv`: Contains 20,215 validation question pairs for tuning hyperparameters and preventing overfitting.
3. `quora-paraphrase-test-student.csv`: Contains 40,431 test question pairs to evaluate the model's performance.

Model outputs and metrics are saved in the 'quota-paraphrase-dev-output.csv' and 'quota-paraphrase-test-output.csv' inside the prediction folder.

### Methodology

This project utilizes a BERT-based model for paraphrase detection, leveraging the HuggingFace Transformers library. The input question pairs are tokenized using BERT's tokenizer, and embeddings are generated for each question pair. 
The following enhancements were implemented:


1. **Loss Function**: Binary cross-entropy with logit loss was used to train the model. To make up for imbalanced data like 62% of data 0-class and 38% of data belonging to the 1-class, some modifications like pos_weight factor or using focal_loss to  a loss element that penalizes false positives and negatives.
2. **Learning Rate Scheduling**: Both cosine decay and learning rate warmup strategies were explored to optimize training efficiency.
3. **Early Stopping**: Early stopping was implemented to terminate training if no improvement in validation loss was observed, thereby preventing overfitting.

### Hyperparameter Tuning Summary

| Hyperparameter                | Description                                                                                   |
|-------------------------------|-----------------------------------------------------------------------------------------------|
| `--epochs`                    | Number of training epochs. Determines the number of complete passes through the training data, directly influencing how long the model will train.
| `--learning_rate`             | Learning rate. Controls how much to change the model's weights during each step of training. A higher learning rate can speed up training but may lead to overshooting minima, while a lower rate allows for more precise but slower convergence. |
| `--pos_weight`                | Positive class weight. Used to handle class imbalance by giving more importance to the minority class during training, typically in binary classification tasks. |
| `--alpha`                     | A hyperparameter is used in some loss functions, like Focal Loss, to control the balance between different classes or aspects of the loss. It adjusts the importance of positive and negative samples in the training process. |
| `--gamma`                     | A parameter used in Focal Loss to reduce the relative loss for well-classified examples, focusing more on hard, misclassified examples. This helps in addressing the class imbalance. |
| `--label_smoothing_factor`    | Label smoothing factor. A regularization technique that prevents the model from becoming too confident by distributing some of the probability from the true class to the other classes, reducing overfitting. |
| `--lambda` (Regression Factor) | Regularization parameter for controlling the strength of regularization, such as L1 or L2, to prevent overfitting by penalizing large weights. |
| `--noise (Epsilon)`           | A small amount of noise is added to the input data or parameters during training to improve generalization and robustness against small perturbations. |
| `--optimizer`                 | The optimization algorithm used to update the model's weights during training, such as AdamW, SGD, or RMSprop. Different optimizers can significantly affect the speed and quality of convergence. |
| `--weight_decay`              | Weight decay (also known as L2 regularization) adds a penalty to the loss function based on the size of the weights, helping to prevent overfitting by discouraging overly complex models. |
| `--regularization`            | General term for techniques like L1, L2, or dropout that help prevent overfitting by penalizing large weights or randomly dropping units during training. |
| `--batch_size`                | The number of training examples used in one iteration of model training. Larger batch sizes can lead to faster training but may require more memory, while smaller batches provide more frequent updates. |
| `--dropout`                   | Dropout rate. A regularization technique where a fraction of neurons is randomly set to zero during each training step, which helps prevent the model from becoming too reliant on specific neurons and improves generalization. |

This summary provides an overview of each hyperparameter's role in the training process, helping to understand how they influence the model's performance and generalization capabilities.

### Experiments

The table below summarizes the key hyperparameters, results, and insights from each experiment conducted:


Abbreviations and Terms:
Exp: Experiment Number
Ep: Epochs - Number of training epochs
LR: Learning Rate
PosWt: Positive Weight - Class weighting for handling imbalance
α (Alpha): Alpha - Parameter in Focal Loss
γ (Gamma): Gamma - Parameter in Focal Loss
ε (Epsilon): Epsilon - Label Smoothing Factor(In the code this is called label smoothing)
λ (Lambda): Lambda - Regression Factor for L2 Regularization
δ (Delta): Small perturbation added during training for robustness(In code this is called Epsilon)
Opt: Optimizer used (e.g., AdamW)
WD: Weight Decay - Regularization technique
Reg: Type of Regularization used ( L2, Dropout)
BS: Batch Size - Number of training samples per batch
DO: Dropout rate
Time: Time Consumption for the training run
Dev Acc: Development (Validation) Accuracy
Train Acc: Training Accuracy


| Exp | Ep | LR   | PosWt | α   | γ   | ε    | λ    | δ      | Opt  | WD   | Reg  | BS | DO | Time  | Dev Acc | Train Acc |    
|-----|----|------|-------|-----|-----|-----|------|--------|------|------|------|----|----|-------|---------|-----------|
| B   | 3  | 1e-5 | None  | N/A | N/A | N/A | N/A  | N/A    | AdamW| 0.01 | None | 64 | 0.1| 40m   | 77.0%   | 93.1%     |
| 1   | 5  | 5e-5 | 2.0   | N/A | N/A | N/A | N/A  | N/A    | AdamW| 0.01 | None | 64 | 0.1| 1h    | 73.2%   | 91.7%     |
| 2   | 5  | 3e-5 | None  | 0.6 | 3   | N/A | N/A  | N/A    | AdamW| 0.001| Drop | 64 | 0.3| 1h 05m| 74.1%   | 97.8%     |
| 3   | 10 | 3e-5 | None  | 0.5 | 10  | N/A | N/A  | N/A    | AdamW| 0.001| Drop | 64 | 0.3| 1h 55m| 75.7%   | 97.3%     |
| 4   | 10 | 5e-5 | None  | 0.1 | 3   | 0.5 | N/A  | N/A    | AdamW| 0.01 | Drop | 64 | 0.3| 2h    | 77.3%   | 98.3%     |
| 5   | 10 | 8e-5 | None  | 0.1 | 3   | 0.09| N/A  | N/A    | AdamW| 0.1  | Drop | 64 | 0.5| 1h 45m| 77.4%   | 97.9%     |
| 6   | 10 | 8e-5 | None  | 0.1 | 3   | 0.09| 1e-2 | 1e-6   | AdamW| 0    | L2   | 32 | 0.1| 6h    | 70.6%   | 72.5%     |
| 7   | 5  | 3e-5 | None  | N/A | N/A | N/A | 1e-4 | 1e-6  | AdamW| 0    | L2   | 32  | 0.1| 3h 05m| 75.5%   | 77.9%     |
| 8   | 10 | 8e-5 | None  | N/A | N/A | N/A | 1e-6 | 1e-10  | AdamW| 0    | L2   | 32 | 0.1| 7h    | 77.8%   | 79.4%     |
| 9  | 20 | 1e-5 | None  | none | none   | none | None | None  | AdamW| 0.01 | None  | 64| 0.1| 4h 30m| 79.0%     | 99.6%    |


\[
\text{Loss} = -\alpha (1 - p_t)^\gamma \cdot \left[ y \log(\hat{y}) + (1-y) \log(1-\hat{y}) \right] + \epsilon \cdot \left( \frac{1}{2} \right) + \lambda \cdot \sum_{j=1}^{n} \left( f(x_j + \delta) - f(x_j) \right)^2
\]
 

### Detailed Results Analysis

1. **Baseline Experiment**:
   - **Goal**: Establish a benchmark with simple settings using binary cross-entropy loss without any advanced regularization techniques.
   - **Configuration**:
     - Epochs: 3, Learning Rate: 1e-5, Optimizer: AdamW, Weight Decay: 0.01, Batch Size: 64, Dropout: 0.1
   - **Results**: Dev Accuracy: 77.0%, Train Accuracy: 93.1%
   - **Discussion**: The baseline model showed a significant difference between training and validation accuracy, indicating potential overfitting. The high training accuracy suggests the model was able to learn patterns from the training data, but the relatively lower validation accuracy shows it had difficulty generalizing to new data.

2. **Experiment 1**:
   - **Goal**: Evaluate the effect of positive class weighting (Pos Weight = 2.0) on handling class imbalance.
   - **Configuration**:
     - Epochs: 5, Learning Rate: 5e-5, Optimizer: AdamW, Weight Decay: 0.01, Batch Size: 64, Dropout: 0.1, Pos Weight: 2.0
   - **Results**: Dev Accuracy: 73.2%, Train Accuracy: 91.7%
   - **Discussion**: Introducing positive class weighting slightly reduced validation accuracy. This might indicate that while the model became more sensitive to the positive class, this sensitivity did not improve overall generalization and possibly overemphasized positive examples.

3. **Experiment 2**:
   - **Goal**: Use dropout regularization and introduce alpha-gamma hyperparameters to improve robustness.
   - **Configuration**:
     - Epochs: 5, Learning Rate: 3e-5, Alpha: 0.6, Gamma: 3, Optimizer: AdamW, Weight Decay: 0.001, Batch Size: 64, Dropout: 0.3
   - **Results**: Dev Accuracy: 74.1%, Train Accuracy: 97.8%
   - **Discussion**: The introduction of dropout and focal loss components (alpha and gamma) increased training accuracy significantly. However, only a slight improvement in dev accuracy was observed, suggesting more tuning is needed to balance training and validation performance.

4. **Experiment 3**:
   - **Goal**: Increase the number of epochs to examine effects on model stability and accuracy.
   - **Configuration**:
     - Epochs: 10, Learning Rate: 3e-5, Alpha: 0.5, Gamma: 10, Optimizer: AdamW, Weight Decay: 0.001, Batch Size: 64, Dropout: 0.3
   - **Results**: Dev Accuracy: 75.7%, Train Accuracy: 97.3%
   - **Discussion**: Training for more epochs slightly increased dev accuracy. However, the high training accuracy with a gap in validation accuracy suggests ongoing overfitting despite regularization efforts.

5. **Experiment 4**:
   - **Goal**: Implement label smoothing to reduce overconfidence in predictions and improve generalization.
   - **Configuration**:
     - Epochs: 10, Learning Rate: 5e-5, Alpha: 0.1, Gamma: 3, Label Smoothing Factor: 0.5, Optimizer: AdamW, Weight Decay: 0.01, Batch Size: 64, Dropout: 0.3
   - **Results**: Dev Accuracy: 77.3%, Train Accuracy: 98.3%
   - **Discussion**: Label smoothing provided noticeable improvements in both training and validation accuracies, indicating a positive impact on reducing model overconfidence.

6. **Experiment 5**:
   - **Goal**: Test the effect of higher learning rates and varying regularization techniques.
   - **Configuration**:
     - Epochs: 10, Learning Rate: 8e-5, Alpha: 0.1, Gamma: 3, Label Smoothing Factor: 0.09, Optimizer: AdamW, Weight Decay: 0.1, Batch Size: 64, Dropout: 0.5
   - **Results**: Dev Accuracy: 77.4%, Train Accuracy: 97.9%
   - **Discussion**: Increasing the learning rate and dropout further improved dev accuracy, showing that higher regularization and a more aggressive learning rate can be beneficial for the model's generalization ability.


7. **Experiment 6**:
   - **Goal**: Combine high dropout and L2 regularization with a noise factor to assess performance and stability.
   - **Configuration**:
     - Epochs: 10, Learning Rate: 8e-5, Alpha: 0.1, Gamma: 3, Label Smoothing Factor: 0.09, Lambda: 1e-2, Delta: 1e-6, Optimizer: AdamW, Weight Decay: 0, Batch Size: 32, Dropout: 0.1
   - **Results**: Dev Accuracy: 70.6%, Train Accuracy: 72.5%
   -**Complication**: The Batch Size: 64 raises the 'out of memory' error. Batch Size: 32 works but takes almost double the time, I had to cut short the epoch in the middle of the task.
   - **Discussion**: This setup demonstrated a balance between training and validation accuracies, indicating improved stability and robustness when using combined regularization techniques. But to incorporate L2 regularisation I might have complexity the loss function a little as the val acc drops.

8. **Experiment 7**:
   - **Goal**: To simplify the loss function with a not-so-aggressive regularisation.
   - **Configuration**:
     - Epochs: 10, Learning Rate: 3e-5, Optimizer: AdamW, Lambda: 1e-4, Delta: 1e-6, Weight Decay: 0, Batch Size: 32, Dropout: 0.5
   - **Results**: Dev Accuracy: 75.5%, Train Accuracy: 77.9%
   -**Complication**: Almost the same complication as the previous one.
   - **Discussion**: This experiment showed modest results, suggesting that or fine-tuning of regularization parameters could be beneficial for better generalization.

9. **Experiment 8**:
    - **Goal**: Ease up on the regression factor(Lambda) and the noise (epsilon) hoping to get a better results by reducing the L2 regularization a little further.
    - **Configuration**:
      - Epochs: 10, Learning Rate: 8e-5, Lambda: 1e-6, Delta: 1e-10, Optimizer: AdamW, Weight Decay: 0, Batch Size: 32, Dropout: 0.2
    - **Results**: Dev Accuracy: 77.8%, Train Accuracy: 79.4%
    -**Complication**: Those complications are still there by the way.
    - **Discussion**: Demonstrating effective results but not by much.

10. **Experiment 9**:
    - **Goal**: Try to get to get as much as possible by the time left.
    - **Configuration**:
      - Epochs: 25, Learning Rate: Schedular ReduceLROnPlateau(with factor 0.8), Alpha: 0.0, Gamma: 0, Label Smoothing Factor: 0, Lambda: 0, Delta: 0, Optimizer: AdamW, Weight Decay: 0.01, Batch Size: 64, Dropout: 0.1
    - **Results**: Dev Accuracy: 79.0%, Train Accuracy: 99.4%. This can be replicated with the simple model with lr=1e-5 with 20 epochs or so.
    - **Discussion**: Due to the time consumption, with not much time left I get back to the simple model and rely on the learning rate to give the best result it could. I use here the This configuration achieved the highest validation accuracy, but there, the overfitting problem has risen in the absence of regularization.

### Complications: 

I have included the 'Time' consumption column to show a big complication in the task. It has almost 400k sentences to  flow through the model, so it is time-consuming. That is why I could not experiment few more techniques to their full potential like POS and NER tagging which helps the model to better understand the features and thereby improve the semantic understanding of the model. Also, the L2 regularisation which I calculate adds noise to the data(δ). You can see that in the multitask_classifier file in the Testing folder. Also, I could not experiment with the SophiaG optimizer as much as I should or wanted to. 

### Conclusion

The series of experiments conducted highlights the critical role of balancing various hyperparameters, regularization techniques, and optimization strategies to enhance model performance in paraphrase detection tasks. The baseline experiment established a strong starting point, but it clearly indicated overfitting issues due to high training accuracy and relatively lower validation accuracy. This was evident across several experiments, showing the importance of carefully managing model complexity to prevent overfitting.

Experiments 2 to 5 explored the introduction of dropout regularization, alpha-gamma parameters, and label smoothing, which incrementally improved the model's robustness and generalization capabilities. The implementation of focal loss components (alpha and gamma) helped focus the model's learning on harder examples, but consistent validation accuracy improvements were modest, suggesting that fine-tuning these hyperparameters is essential for optimal performance.

Regularization strategies such as L2 regularization, combined with noise injection (experiments 6 to 8), demonstrated the potential to stabilize the learning process and improve generalization. However, these configurations also revealed practical complications, including increased computational time and memory constraints, which limited the ability to fully explore more complex techniques. Experiment 7 specifically aimed to simplify the loss function to manage these challenges better, showing a balance between dev and training accuracy but not significantly improving results.

The final experiment (Experiment 9) underscored the trade-off between achieving higher accuracy and managing overfitting. By reverting to a simpler model configuration and relying on adaptive learning rates, the highest validation accuracy was achieved, albeit with a notable overfitting issue. This highlights the importance of regularization, even when focusing on achieving the best possible accuracy.

Overall, these experiments confirm that fine-tuning dropout rates, using regularization techniques (like L2 and label smoothing), and adjusting learning rates dynamically are vital strategies for controlling overfitting and enhancing model generalization. However, real-world constraints, such as time and computational resources, significantly impact the scope of experimentation. Future work should explore optimizing these hyperparameters in a more resource-efficient manner, potentially incorporating advanced optimizers like SophiaG, POS, and NER tagging techniques to improve the model's semantic understanding and performance.
 
### Evaluation Metrics

1. **Accuracy**: Measures the proportion of correctly classified question pairs.

These metrics offer insights into the model's effectiveness in correctly identifying paraphrases and avoiding false positives.


### Future Work
The following enhancements can also be tested, as a part of additional work.
1. **Data Augmentation**: Synonym replacement and back-translation techniques were employed to increase training data diversity.
2. Experimenting with other language models such as Roberta and XLNet.
3. Experiment with metrics like Recall, Precision, and F1 to monitor the model's class-specific performance.
4. I want  to further experiments with regularization like L1, L2, or the Kl divergence loss.
This README provides a comprehensive overview of the Quora Question Pair Paraphrase Detection project, including setup instructions, methodology, experiments, results, and future work directions.

## BERT - Stanford Sentiment Treebank (SST) Dataset

### Introduction

This project focuses on improving a BERT-based model for multitask (sst, sts, and qqp). We tuned various hyperparameters to achieve this balance, and the results of our experiments are summarized in this document. Understanding a text in SST involves determining its polarity (positive, negative, or neutral). Sentiment analysis can reveal personal feelings about products, politicians, or news reports. Each phrase is labeled as negative, slightly negative, neutral, slightly positive, or positive.

### Setup Instructions

To set up the environment and run the model, follow these steps:

1. Run `./setwp_gwdf.sh` to install all the required dependencies and set up the environment.
2. Use `sbatch run_train_bert.sh` with 'sst' or 'multitask' argument on a GPU-enabled cluster to initiate model training using Slurm.

The setup script installs necessary libraries such as TensorFlow, PyTorch, and HuggingFace Transformers. A GPU-enabled environment with CUDA support is recommended for faster training.

### Methodology

This section describes the methodology used in our experiments to extend the training of the multitask BERT model for Sentiment.

### Hyperparameter Tuning Summary

| Parameter               | Description                                                                    |
| ----------------------- | ------------------------------------------------------------------------------ |
| `--batch_size`          | Batch size.                                                                    |
| `--epochs`              | Number of epochs.                                                              |
| `--hidden_dropout_prob` | Dropout probability for hidden layers.                                         |
| `--lr`                  | Learning rate.                                                                 |
| `--use_gpu`             | Whether to use the GPU.                                                        |
| `--seed`                | Random seed for reproducibility.                                               |
| `--batch_size`          | Defining batch size for training model.                                        |
| `--task`                | Task to be trained (sst, qqp, sts, etpc) or multitask to train togeather.      |

### Experiments:

#### Baseline

For the baseline model, we chose the following hyperparameters. 

- mode: `finetune`
- epochs: `10`
- learning rate: `1e-5`
- optimizer: `AdamW`
- batch size: `32`

#### Smoothness-Inducing Regularization

Initially, the baseline score was 52.2%, after trying some methods to improve the performance like data augmentation, the model was showing strong overfitting, which we tried to overcome using LRscheduling and EarlyStopping, but it was greatly complemented by the Smoothness-inducing regularization technique, which did not help in improving the validation accuracy but helped the model to avoid too strong overfitting on data.

#### AdamW, SophiaH, Bregman proximal Point optimization

Since various optimizers play a pivotal role in task definition, we tried a few of them to improve the model performance. There was no noticeable difference between AdamW and SophiaH they both performed the same with Bregman proximal Point optimization, it has addressed overfitting greatly but also hindered the overall learning of the model, using the Bregman proximal Point optimization model was getting an accuracy of roughly 25% accuracy in an initial epoch which is early half of that we got in AdamW and SophiaH i.e. roughly 45%

Sophia: [Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training](https://arxiv.org/abs/2305.14342).

Bregman proximal Point: [Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models Principled Regularized Optimization](https://aclanthology.org/2020.acl-main.197.pdf)

#### Data Augmentation (Synonym Replacement and Random Noise) & Synthetic Dataset

Given recent advances in imitation learning, particularly the demonstrated ability of compact language models to emulate the performance of larger, proprietary counterparts ([Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)), we investigated the role of synthetic data in improving multitask classification models. Our focus was on sentiment classification. WE generated around 6000 more data points adding them to the original dataset to have a larger training dataset. OpenAI's GPT-3 and GPT-4 were trained on undisclosed datasets, which raises concerns about data overlap with our sentiment classification set. Even though these models are unlikely to reproduce specific test set instances, the issue persists and should be addressed. But, it just contributed to overfitting the model. A similar case was with the data augmentation techniques.

### Experiment Results

| Model name                      | Parameters                                | Accuracy |
| ------------------------------- | ----------------------------------------- | -------- |
| Baseline                        |                                           | 52.2%    |
| Increasing model complexity     | `--lr 1e-05 --hidden_dropout_prob 0.1`    | 52.6%    |
| Bregman proximal Point optim.   | `--lr 1e-05 --hidden_dropout_prob 0.3`    | 46.2%    |
| bert projected attention layer  | `--lr 1e-05 --hidden_dropout_prob 0.3`    | 46.2%    |
| SophiaH                         |                                           | 45.4%    |
| Synthetic Data                  | `--sst_train data/synthetic_data.csv`     | 51.3%    |
| Synonym Replacement             |                                           | 46.6%    |

### Conclusion

Even though we tried many things to improve the model accuracy there was negligible improvement from the baseline when we introduced more layers in a model like residual connection, layer normalization, and dropout. the accuracy went from 52.2% to 52.6%, and that's the best accuracy we got for the first task after trying many things.

## BART - Paraphrase Type Generation

### Introduction

This project focuses on improving a BART-based model for paraphrase generation. The goal is to balance accuracy (measured by the BLEU score) and diversity (measured by the Negative BLEU score) of the generated paraphrases. We tuned various hyperparameters to achieve this balance, and the results of our experiments are summarized in this document.

### Setup Instructions

To set up the environment and run the model, follow these steps:

1. Run `./setup_gwdg_bart_generation.sh` to set up the environment.
2. Use `sbatch run_train_bart_generation.sh` on the GWDG cluster to start model training via Slurm.

All necessary libraries are listed in the setup script. The project is designed to run on a GPU-enabled environment with CUDA support for faster processing.

### Data

The data required for this project is stored in the `data` folder. The datasets used are:

1. `processed_etpc_paraphrase_train.csv`: Training data.
2. `processed_etpc_paraphrase_dev.csv`: Validation data.
3. `etpc-paraphrase-generation-test-student.csv`: Test data.

Model predictions are saved in the `predictions` folder.

### Methodology

This project uses a BART model from HuggingFace’s Transformers library for paraphrase generation. We augmented the data by replacing words with synonyms to increase the diversity of the training data. The model combines standard loss with contrastive loss to penalize paraphrases that are too similar to the input. We also implemented early stopping and learning rate scheduling to improve training efficiency. Throughout the experiments, weight decay was consistently set to 0.005 to prevent overfitting, which helped maintain a balance between model complexity and generalization.

#### Penalized BLEU Score Calculation

The Penalized BLEU score is calculated by combining the standard BLEU score with the Negative BLEU score, which measures diversity. The formula used is: 

**Penalized BLEU Score = (BLEU Score * Negative BLEU Score) / 52**

This formula balances the trade-off between generating accurate and diverse paraphrases, aiming for a model that can produce outputs that are both semantically correct and sufficiently varied.

### Hyperparameter Tuning Summary

| Hyperparameter                        | Description                                                                                   |
|---------------------------------------|-----------------------------------------------------------------------------------------------|
| `--epochs`                            | Number of training epochs. Determines how many times the model sees the entire training data. |
| `--lr`                                | Learning rate. Controls how quickly the model updates during training.                        |
| `--scheduler_type`                    | Type of learning rate scheduler. Options: `cosine`, `reduce_on_plateau`, `warmup`.            |
| `--contrastive_weight`                | Weight for the contrastive loss. Balances accuracy and diversity of generated paraphrases.     |
| `--early_stopping_patience`           | Number of epochs to wait before stopping if no improvement is seen. Prevents overfitting.     |
| `--num_beams`                         | Number of beams used in beam search. Higher values can improve output quality.                |
| `--num_beam_groups`                   | Number of groups for diverse beam search. Increases output diversity.                         |
| `--diversity_penalty`                 | Penalty to encourage diversity in beam search. Higher values lead to more diverse paraphrases.|
| `--augment_factor`                    | Amount of data augmentation through synonym replacement.                                      |
| `--use_gpu`                           | Whether to use GPU for training.                                                              |
| `--seed`                              | Random seed for reproducibility.                                                              |

### Experiments

The following table summarizes the key hyperparameters, results, and insights from each experiment:

| Experiment | Epochs | Learning Rate | Contrastive Loss Weight | Scheduler Type   | Beam Search (num_beams, num_beam_groups, diversity_penalty)        | BLEU Score | Penalized BLEU Score |
|------------|--------|---------------|--------------------------|------------------|---------------------------------------------------------------------|------------|-----------------------|
| Baseline   | 3      | 5e-5          | N/A                      | None             | None                                                                | 46.6373    | 3.541                 |
| 1          | 5      | 5e-5          | 0.1                      | Cosine           | num_beams=6, num_beam_groups=2, diversity_penalty=1.0               | 45.1888    | 13.306                |
| 2          | 10     | 3e-5          | 0.05                     | Cosine           | num_beams=5, num_beam_groups=2, diversity_penalty=0.8               | 46.1133    | 11.234                |
| 3          | 10     | 3e-5          | 0.1                      | ReduceOnPlateau  | num_beams=6, num_beam_groups=2, diversity_penalty=1.0               | 45.9262    | 10.647                |
| 4          | 5      | 5e-5          | 0.05                     | Cosine           | num_beams=4, num_beam_groups=2, diversity_penalty=0.7               | 46.2459    | 12.792                |
| 5          | 10     | 5e-5          | 0.1                      | Warmup           | num_beams=6, num_beam_groups=3, diversity_penalty=1.2               | 46.3249    | 14.087                |
| 6          | 12     | 2e-5          | 0.4                      | Cosine           | num_beams=6, num_beam_groups=2, diversity_penalty=1.0               | 40.6632    | 22.401                |
| 7          | 15     | 5e-5          | 0.3                      | Warmup           | num_beams=6, num_beam_groups=2, diversity_penalty=1.3               | 38.5582    | 24.192                |
| 8          | 15     | 3e-5          | 0.25                     | Warmup           | num_beams=6, num_beam_groups=2, diversity_penalty=1.3               | 42.3426    | 20.097                |
| 9          | 19     | 3e-5          | 0.3                      | Warmup           | num_beams=6, num_beam_groups=2, diversity_penalty=1.0               | 41.6671    | 22.820                |

### Detailed Results Analysis

1. **Baseline Experiment**: 
   - **Goal**: Establish a baseline without contrastive loss, scheduler, or other advanced features.
   - **Results**: BLEU score: 46.6373, Penalized BLEU score: 3.541
   - **Discussion**: The baseline model achieved good BLEU scores across epochs, but the Penalized BLEU scores indicate a lack of diversity in generated paraphrases.

2. **Experiment 1**: 
   - **Goal**: Introduce contrastive loss with a focus on accuracy.
   - **Scheduler**: Cosine scheduler was used for its smooth decay, which helps avoid sudden drops that could cause premature convergence.
   - **Results**: BLEU score: 45.1888, Penalized BLEU score: 13.306.
   - **Discussion**: The model achieved a good BLEU score but with limited diversity, leading to a moderate Penalized BLEU score.

3. **Experiment 2**: 
   - **Goal**: Improve diversity without sacrificing BLEU score.
   - **Scheduler**: Cosine scheduler provided a gradual learning rate decay, which allowed the model to adapt slowly and maintain stability.
   - **Results**: BLEU score: 46.1133, Penalized BLEU score: 11.234.
   - **Discussion**: Slight improvements in diversity were achieved with a higher BLEU score, but further tuning was needed to enhance the Penalized BLEU score.

4. **Experiment 3**: 
   - **Goal**: Test dynamic learning rate adjustment with ReduceOnPlateau scheduler.
   - **Scheduler**: ReduceOnPlateau was chosen to dynamically lower the learning rate when improvements plateaued, aiming for finer model adjustments.
   - **Results**: BLEU score: 45.9262, Penalized BLEU score: 10.647.
   - **Discussion**: The model's BLEU score remained stable, but the diversity did not increase as desired, leading to a lower Penalized BLEU score.

5. **Experiment 4**: 
   - **Goal**: Focus on accuracy with lower beams and a conservative diversity penalty.
   - **Scheduler**: Cosine scheduler was retained for its ability to provide smooth and consistent learning rate decay.
   - **Results**: BLEU score: 46.2459, Penalized BLEU score: 12.792.
   - **Discussion**: The model achieved a balanced outcome with a high BLEU score but less diverse paraphrases, reflected in a moderate Penalized BLEU score.

6. **Experiment 5**: 
   - **Goal**: Explore the impact of warmup scheduling on diversity.
   - **Scheduler**: Warmup scheduler was used to gradually increase the learning rate, allowing the model to stabilize before fully engaging in the learning process.
   - **Results**: BLEU score: 46.3249, Penalized BLEU score: 14.087.
   - **Discussion**: This led to smoother learning, resulting in a higher Penalized BLEU score with slightly better diversity.

7. **Experiment 6**: 
   - **Goal**: Maximize diversity with a lower learning rate and higher contrastive loss.
   - **Scheduler**: Cosine scheduler continued to provide smooth decay as diversity was pushed further.
   - **Results**: BLEU score: 40.6632, Penalized BLEU score: 22.401.
   - **Discussion**: The model achieved a high Negative BLEU score but at the cost of accuracy, resulting in a lower BLEU score.

8. **Experiment 7**: 
   - **Goal**: Combine warmup scheduling with moderate contrastive loss for a balanced outcome.
   - **Scheduler**: Warmup scheduler allowed the model to gradually adjust to the increased diversity.
   - **Results**: BLEU score: 38.5582, Penalized BLEU score: 24.192.
   - **Discussion**: The highest Penalized BLEU score was achieved, indicating a successful balance between accuracy and diversity.

9. **Experiment 8**: 
    - **Goal**: Fine-tune the balance between accuracy and diversity.
    - **Scheduler**: Warmup scheduler helped stabilize early learning, ensuring consistent improvement.
    - **Results**: BLEU score: 42.3426, Penalized BLEU score: 20.097.
    - **Discussion**: The model achieved the most balanced result with strong BLEU and Penalized BLEU scores.

10. **Experiment 9**:
    - **Goal**: Explore the impact of warmup scheduling on the stability of the learning process.
    - **Scheduler**: Warmup scheduler was used to stabilize the model’s learning process over a longer training period, with early stopping in place to prevent overfitting.
    - **Results**: BLEU score: 41.6671, Penalized BLEU score: 22.820.
    - **Discussion**: This experiment demonstrated a steady improvement in Penalized BLEU score, peaking after 19 epochs.

### Conclusion

The experiments revealed that balancing accuracy and diversity is key to optimizing the Penalized BLEU score. The warmup scheduler, in particular, proved effective in stabilizing the learning process, allowing the model to adapt more effectively to changes in hyperparameters. Fine-tuning the contrastive loss weight, learning rate, and beam search configuration enabled the achievement of a strong final model performance.

Future work could focus on exploring alternative data augmentation strategies and further optimizing the contrastive loss component to improve both accuracy and diversity.

  
## BART - Paraphrase Type Detection

### Introduction

This project focuses on improving a BART-based model for paraphrase type detection. The goal is to fine-tune the BART model to improve the performance of the validation data.
For evaluation Accuracy was used, but models mostly collapsed due to the bad metrics used and the imbalance of multiclass labels in the dataset. Finally ** Matthews Correlation Coefficient** will 
used for evaluation per paraphrase class.

### Setup Instructions

To set up the environment and run the model, follow these steps:

1. Run `./setup_gwdg.sh` to set up the environment.
2. Use `sbatch run_bart_detection.sh` on the GWDG cluster to start model training via Slurm.
3. Monitor the progress using slurm files in slurm_folder.
   
All necessary libraries are listed in the setup script. The project is designed to run on a GPU-enabled environment with CUDA support for faster processing.

### Data

The data required for this project is stored in the `data` folder. The datasets used are:

1. `etpc_paraphrase_train.csv`: Training data.
2. `etpc_paraphrase_dev.csv`: Validation data.
3. `etpc-paraphrase-detection-test-student.csv`: Test data.

### Methodology

This project uses a BART model from HuggingFace’s Transformers library for paraphrase type detection. 

#### Baseline

For the Baseline, I have used the BCELoss function, Since the BART class was returning the probabilities. I have trained the model for 10 epochs.

#### Improvements

##### Weights With BCELoss:
  The idea is to penalize the majority classes and putt more weight on the minority classes. I have defined the Weights in 3 categories

| Category | Description  |
| ----------------------- | ------------------------------------------------------------------------------ |
| `None`  | With this argument no weight will be used.  |
| `Fixed`  | With this argument we used the fixed weight for each class that we define in the code.  |
| `Deterministic`  | With this argument, weight is calculated from the data.  |
  
##### Focal Loss

In many real-world applications of deep learning, there is often a significant class imbalance in the dataset. For example, in the case of detecting rare objects in images, the number of background examples greatly outnumbers the positive examples. Traditional loss functions like cross-entropy are not well-suited for these scenarios because they tend to be overwhelmed by the numerous easy examples, causing the model to underperform on minority examples. That's why I chose this loss function since in the Type Detection case our   data is imbalanced.

Focal Loss was introduced to address the imbalance challenge by focusing the learning on hard examples while down-weighting the contribution of easy examples. This is achieved by modulating the standard cross-entropy loss with a factor `(1 - pt)^gamma`, where `pt` is the model's estimated probability for the true class. When the model is confident about the prediction (i.e., `pt` is high), the factor is small, reducing the loss contribution from easy examples. Conversely, when the model struggles (i.e., `pt` is low), the factor is large, increasing the loss contribution and encouraging the model to focus on these harder examples.

###### Parameters

`alpha`: is a weighting factor that balances the importance between classes, especially in imbalanced datasets. It helps prevent the model from being biased toward the majority class by assigning a higher weight to the minority class

`gamma`: adjusts the loss contribution from easy examples by reducing their impact, focusing learning on harder examples. A higher `gamma` places more emphasis on difficult-to-classify cases, making the model prioritize them during training.

##### SMART Regularization

Smart regularization is an advanced technique designed to improve the generalization of deep learning models by adding constraints or modifications to the loss function. This approach helps the model learn smoother, more stable representations that are less likely to overfit the training data. It has two components.

###### Smoothness-Inducing Loss

The goal of smoothness-inducing loss is to make the model’s predictions stable and consistent, even when small perturbations are applied to the input data. This encourages the model to learn robust features that do not change drastically due to minor variations in input.

Components:

1. **Perturbation with Noise:** Small Gaussian noise is added to the input embeddings, which creates a perturbed version of the inputs. This simulates slight variations in data, helping the model become insensitive to minor changes.
2. **Smoothness Loss:** This is calculated as the L2 norm of the difference between the outputs from the original and perturbed inputs. It penalizes large deviations in model predictions caused by small input changes, enforcing stability.
3. **Total Loss:** The final loss combines the original BCELoss with the smoothness loss, weighted by a regularization parameter (lambda_reg). This ensures that the model not only learns to minimize the loss but also maintains smoothness.

###### Bregman Proximal Point Update
This method is used to stabilize the training process by applying a Bregman divergence-based constraint on the model parameters. It helps maintain a balance between the updated parameters and the original parameters after each optimization step.

Components:

1. **Original Parameters:** Before updating the model’s parameters, a copy of the current parameters is saved as original_params.
2. **Proximal Update:** After the standard optimization step (e.g., gradient descent), the Bregman proximal point update is applied. It adjusts the parameters by pulling them closer to the original parameters using a small step size (eta). This serves as a regularization mechanism that prevents the model from diverging too much from the current solution, ensuring more stable and consistent updates.

This technique is adapted from Jiang et al., 2020 ([SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization](https://aclanthology.org/2020.acl-main.197) (Jiang et al., ACL 2020))

### Hyperparameter Tuning Summary

| Parameter | Description  |
| ----------------------- | ------------------------------------------------------------------------------ |
| `--epochs`  | Number of epochs  |
| `--lr`  | Learning rate  |
| `--seed`  | Random Seed  |
| `--use_gpu` | Whether to use the GPU  |
| `--use_weight`  | Weight use with BCE Loss, it could be None, Fix or Deterministic  |

### Experiment Details

In this study, we conducted a series of experiments to evaluate the effectiveness of Smart regularization strategy, loss functions, and weight assignment methods on the model's performance, measured by the Matthews Correlation Coefficient (MCC) after 10 epochs. The experiments were designed to explore the impact of the SMART regularization technique, different weighting schemes, and loss functions on the model's ability to generalize.

#### Experiment Configurations

1. **SMART Regularization:**
   - **Yes:** Indicates that SMART regularization was applied during training. SMART is designed to improve model robustness by inducing smoothness in the loss function.
   - **No:** Indicates that SMART regularization was not applied.

2. **Weight:**
   - **None:** No additional weighting was applied to the loss function.
   - **Deterministic:** A deterministic weighting scheme was applied, based on the class distribution.
   - **Fixed:** A fixed weighting scheme was applied, where the weights are constant throughout the training.

3. **Loss Function:**
   - **BCE:** Binary Cross-Entropy loss, a standard loss function for binary classification tasks.
   - **Focal:** Focal loss, designed to address class imbalance by focusing more on hard-to-classify samples.

### Results Summary

| **SMART** | **Weight**      | **Loss** | **MCC Value** |
|-----------|-----------------|----------|---------------|
| No        | None            | BCE      | 0.131         |
| No        | Deterministic   | BCE      | 0.1424        |
| No        | Fixed           | BCE      | 0.1515        |
| **Yes**       | **None**            | **BCE**      | **0.18**        |
| Yes       | Deterministic   | BCE      | 0.1212        |
| Yes       | Fixed           | BCE      | 0.1244        |
| No        | None            | Focal    | 0.1514        |
| Yes       | None            | Focal    | 0.155         |


### Key Observations

1. **SMART Regularization Impact:**
   - Applying SMART regularization with no weighting (`Weight = None`) and using Binary Cross-Entropy (BCE) as the loss function resulted in the highest MCC value of 0.18, indicating that the SMART technique significantly improved model performance without additional weighting.
   - However, when deterministic or fixed weights were combined with SMART, the MCC values decreased, It suggests that SMART may be more effective without additional weighting mechanisms.

2. **Loss Function Comparison:**
   - The Focal loss function generally performed better than BCE in scenarios without SMART regularization, particularly with no weights (`Weight = None`).
   - With SMART regularization, BCE still slightly outperformed Focal loss.

3. **Weighting Schemes:**
   - The Fixed weighting scheme provided the best MCC value among experiments without SMART regularization, indicating that it helps in scenarios where SMART is not used.
   - Interestingly, the deterministic and fixed weights negatively impacted the MCC when SMART was applied, highlighting the importance of selecting the right combination of regularization and weighting strategies.

### Conclusion
The experiments demonstrate that SMART regularization combined with no additional weighting yields the best performance with Binary Cross-Entropy loss. However, the combination of regularization techniques, loss functions, and weighting strategies requires careful consideration to achieve optimal results.


## Contributors


| Injamam  | Rehan | Saad | Reda | Smitesh |
| ---------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| Tagging  | Sophia Optimizer  | Synthetic Data  | Synthetic Data  | Synthetic Data  |
| Layer Unfreeze | Hyperparameter Tuning | | Synthetic Data  |
| Classifier Model | Repository  | | Synthetic Data  |

  
## AI-Usage Card


Artificial Intelligence (AI) aided the development of this project. For transparency, we provide our [AI-Usage Card](./AI-Usage-Card.pdf/) at the top. The card is based on [https://ai-cards.org/](https://ai-cards.org/).


## Acknowledgement


The project description, partial implementation, and scripts were adapted from the default final project for the
Stanford [CS 224N class](https://web.stanford.edu/class/cs224n/) developed by Gabriel Poesia, John, Hewitt, Amelie Byun,
John Cho, and their (large) team (Thank you!)


The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon

University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),


created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig  (Thank you!)

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers)
