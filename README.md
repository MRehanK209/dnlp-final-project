# DNLP SS23 Final Project - Multitask BERT and BART for PTD and PTG

  

<div align="left">

<b> Algorithmic Advancement Alliances </b> <br/>

Karim, Injamam <br/>

Sorathiya, Smitesh <br/>

Ahmad, Saad <br/>

Khalid, Muhammad Rehan <br/>

Arsalane, Mohamed Reda <br/>

</div>

  

## Introduction

  

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org/)

[![Apache License 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)

[![Final](https://img.shields.io/badge/Status-Final-purple.svg)](https://https://img.shields.io/badge/Status-Final-blue.svg)

[![Black Code Style](https://img.shields.io/badge/Code%20Style-Black-black.svg)](https://black.readthedocs.io/en/stable/)

[![AI-Usage Card](https://img.shields.io/badge/AI_Usage_Card-pdf-blue.svg)](./AI-Usage-Card.pdf/)

  

This repository our official implementation of the Multitask BERT project and BART Fine Tuning for the Deep Learning for Natural Language

Processing course at the University of Göttingen.

BERT: ([BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805))

A pre-trained BART:

The BART model was used as the basis for our experiments for PTD and PTG tasks. The model was fine-tuned separately for the datasets of PTG and PTD tasks.


## Requirements

To install requirements and all dependencies and create the environment:


```sh

./setup.sh

```

If setting up the environment on GWDG, then run:

 ```sh

./setup_gwdg.sh

```

The environment is activated with `conda activate dnlp`.


Additionally, the POS and NER tags need to be downloaded. This can be done by running `python -m spacy download en_core_web_sm`.

  

Alternatively, use the provided script `setup.sh`.

The script will create a new conda environment called `dnlp2` and install all required packages.


  
## BERT

To train the model, activate the environment and run this command:

  

```sh

python -u multitask_classifier.py --use_gpu

```

  

There are a lot of parameters that can be set. To see all of them, run `python multitask_classifier.py --help`. The most

important ones are:

  

| Parameter | Description  |
| ----------------------- | ------------------------------------------------------------------------------ |
| `--additional_input`  | Activates the usage for POS and NER tags for the input of BERT |
| `--batch_size`  | Batch size.  |
| `--clip`  | Gradient clipping value. |
| `--epochs`  | Number of epochs.  |
| `--hidden_dropout_prob` | Dropout probability for hidden layers. |
| `--hpo_trials`  | Number of trials for hyperparameter optimization.  |
| `--hpo` | Activate hyperparameter optimization.  |
| `--lr`  | Learning rate. |
| `--optimizer` | Optimizer to use. Options are `AdamW` and `SophiaH`. |
| `--option`  | Determines if BERT parameters are frozen (`pretrain`) or updated (`finetune`). |
| `--samples_per_epoch` | Number of samples per epoch. |
| `--scheduler` | Learning rate scheduler to use. Options are `plateau`, `cosine`, and `none`. |
| `--unfreeze_interval` | Number of epochs until the next BERT layer is unfrozen |
| `--use_gpu` | Whether to use the GPU.  |
| `--weight_decay`  | Weight decay for optimizer.  |


## BART - Paraphrase Type Generation

### Introduction

This project focuses on improving a BART-based model for paraphrase generation. The goal is to balance accuracy (measured by the BLEU score) and diversity (measured by the Negative BLEU score) of the generated paraphrases. We tuned various hyperparameters to achieve this balance, and the results of our experiments are summarized in this document.

### Setup Instructions

To set up the environment and run the model, follow these steps:

1. Run `./setup_gwdg_bart_generation.sh` to set up the environment.
2. Use `sbatch run_train_bart_generation.sh` on the GWDG cluster to start model training via Slurm.
3. Monitor the progress using slurm files in slurm_folder.
   
All necessary libraries are listed in the setup script. The project is designed to run on a GPU-enabled environment with CUDA support for faster processing.

### Data

The data required for this project is stored in the `data` folder. The datasets used are:

1. `processed_etpc_paraphrase_train.csv`: Training data.
2. `processed_etpc_paraphrase_dev.csv`: Validation data.
3. `etpc-paraphrase-generation-test-student.csv`: Test data.

Model predictions are saved in the `predictions` folder.

### Methodology

This project uses a BART model from HuggingFace’s Transformers library for paraphrase generation. We augmented the data by replacing words with synonyms to increase the diversity of the training data. The model combines standard loss with contrastive loss to penalize paraphrases that are too similar to the input. We also implemented early stopping and learning rate scheduling to improve training efficiency.

### Hyperparameter Tuning Summary

| Hyperparameter                | Description                                                                                   |
|-------------------------------|-----------------------------------------------------------------------------------------------|
| `--epochs`                    | Number of training epochs. Determines how many times the model sees the entire training data. |
| `--lr`                        | Learning rate. Controls how quickly the model updates during training.                        |
| `--scheduler_type`            | Type of learning rate scheduler. Options: `cosine`, `reduce_on_plateau`, `warmup`.            |
| `--contrastive_weight`        | Weight for the contrastive loss. Balances accuracy and diversity of generated paraphrases.     |
| `--early_stopping_patience`   | Number of epochs to wait before stopping if no improvement is seen. Prevents overfitting.  |
| `--num_beams`                 | Number of beams used in beam search. Higher values can improve output quality.                |
| `--num_beam_groups`           | Number of groups for diverse beam search. Increases output diversity.                         |
| `--diversity_penalty`         | Penalty to encourage diversity in beam search. Higher values lead to more diverse paraphrases.|
| `--augment_factor`            | Amount of data augmentation through synonym replacement.                                      |
| `--use_gpu`                   | Whether to use GPU for training.                                                              |
| `--seed`                      | Random seed for reproducibility.                                                              |

### Experiments

The following table summarizes the key hyperparameters, results, and insights from each experiment:

| Experiment | Epochs | Learning Rate | Contrastive Loss Weight | Scheduler Type   | Beam Search (num_beams, num_beam_groups, diversity_penalty)        | BLEU Score | Penalized BLEU Score |
|------------|--------|---------------|--------------------------|------------------|---------------------------------------------------------------------|------------|-----------------------|
| 1          | 5      | 5e-5          | 0.1                      | Cosine           | num_beams=6, num_beam_groups=2, diversity_penalty=1.0               | 45.1888    | 13.306                |
| 2          | 10     | 3e-5          | 0.05                     | Cosine           | num_beams=5, num_beam_groups=2, diversity_penalty=0.8               | 46.1133    | 11.234                |
| 3          | 10     | 3e-5          | 0.1                      | ReduceOnPlateau  | num_beams=6, num_beam_groups=2, diversity_penalty=1.0               | 45.9262    | 10.647                |
| 4          | 10     | 5e-5          | 0.05                     | Cosine           | num_beams=4, num_beam_groups=2, diversity_penalty=0.7               | 46.2459    | 12.792                |
| 5          | 10     | 5e-5          | 0.1                      | Warmup           | num_beams=6, num_beam_groups=3, diversity_penalty=1.2               | 46.3249    | 14.087                |
| 6          | 15     | 5e-5          | 0.3                      | Cosine           | num_beams=6, num_beam_groups=2, diversity_penalty=1.5               | 45.9954    | 3.278                 |
| 7          | 15     | 2e-5          | 0.4                      | Cosine           | num_beams=6, num_beam_groups=2, diversity_penalty=1.0               | 40.6632    | 22.401                |
| 8          | 15     | 5e-5          | 0.3                      | Warmup           | num_beams=6, num_beam_groups=2, diversity_penalty=1.3               | 38.5582    | 24.192                |
| 9          | 15     | 3e-5          | 0.25                     | Warmup           | num_beams=6, num_beam_groups=2, diversity_penalty=1.3               | 42.3426    | 20.097                |

### Detailed Results Analysis

1. **Experiment 1**: Established a baseline with a moderate learning rate and low contrastive loss. The model achieved a good BLEU score but with limited diversity.
2. **Experiment 2**: Reduced the contrastive loss weight, leading to better BLEU scores but only slight improvements in diversity.
3. **Experiment 3**: Tested dynamic learning rate adjustment with ReduceOnPlateau, but results were less impactful in improving diversity.
4. **Experiment 4**: Lowered the number of beams and diversity penalty, focusing on more accurate but less diverse outputs.
5. **Experiment 5**: Introduced warmup scheduling with increased diversity, leading to a more balanced outcome in both accuracy and diversity.
6. **Experiment 6**: Pushed for higher diversity with an increased contrastive loss and diversity penalty, but the BLEU score dropped significantly.
7. **Experiment 7**: Further increased the contrastive loss to maximize diversity, leading to a notable drop in accuracy.
8. **Experiment 8**: Combined warmup scheduling with a moderate contrastive loss, yielding the highest Penalized BLEU score by effectively balancing accuracy and diversity.
9. **Experiment 9**: Fine-tuned the parameters to find the best balance between accuracy and diversity, achieving strong BLEU and Penalized BLEU scores.

### Conclusion

The experiments showed that balancing accuracy and diversity in paraphrase generation requires careful tuning of hyperparameters. The best results were achieved with a moderate learning rate, a carefully adjusted contrastive loss weight, and a warmup scheduler. Future work could further refine these parameters and explore additional data augmentation techniques to improve diversity without compromising accuracy.
  
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


## Evaluation for BERT

  

The model is evaluated after each epoch on the validation set. The results are printed to the console and saved in

the `logdir` directory. The best model is saved in the `models` directory.


## Evaluation for BART Detection:

The model is evaluated after each epoch on the validation set. The results are printed to the console. In BART Detection, 
the model will run for all the epochs and return the model after training on the last epoch.

## Results for BERT

  

As a Baseline of our model we chose the following hyperparameters. These showed to be the best against overfitting (which was our main issue) in our hyperparameter search and provided a good starting point for further improvements.

  

- mode: `finetune`

- epochs: `20`

- learning rate: `8e-5`

- scheduler: `ReduceLROnPlateau`

- optimizer: `AdamW`

- clip norm: `0.25`

- batch size: `64`

  

This allowed us to evaluate the impact of the different improvements to the model. The baseline model was trained at 10.000 samples per epoch until convergence. For further hyperparameter choices, see the default values in the [training script](./multitask_classifier.py).

  

---

  

Our multitask model achieves the following performance on:

  

### [Paraphrase Identification on Quora Question Pairs](https://paperswithcode.com/sota/paraphrase-identification-on-quora-question)

  

Paraphrase Detection is the task of finding paraphrases of texts in a large corpus of passages.

Paraphrases are “rewordings of something written or spoken by someone else”; paraphrase

detection thus essentially seeks to determine whether particular words or phrases convey

the same semantic meaning.

  

| Model name | Parameters  | Accuracy |
| -------------- | ----------------------------------------- | -------- |
| data2Vec | State-of-the-art single task model  | 92.4%  |
| Baseline | | 87.0%  |
| Tagging  | `--additional_input`  | 86.6%  |
| Synthetic Data | `--sst_train data/ids-sst-train-syn3.csv` | 86.5%  |
| SophiaH  | `--optimizer sophiah` | 85.3%  |

  

### [Sentiment Classification on Stanford Sentiment Treebank (SST)](https://paperswithcode.com/sota/sentiment-analysis-on-sst-5-fine-grained)

  

A basic task in understanding a given text is classifying its polarity (i.e., whether the expressed

opinion in a text is positive, negative, or neutral). Sentiment analysis can be utilized to

determine individual feelings towards particular products, politicians, or within news reports.

Each phrase has a label of negative, somewhat negative,

neutral, somewhat positive, or positive.

  

| Model name  | Parameters  | Accuracy |

| ------------------------------- | ----------------------------------------- | -------- |
| Heinsen Routing + RoBERTa Large | State-of-the-art single task model  | 59.8%  |
| Tagging | `--additional_input`  | 50.4%  |
| SophiaH | `--optimizer sophiah` | 49.4%  |
| Baseline  | | 49.4%  |
| Synthetic Data  | `--sst_train data/ids-sst-train-syn3.csv` | 47.6%  |

  

### [Semantic Textual Similarity on STS](https://paperswithcode.com/sota/semantic-textual-similarity-on-sts-benchmark)

  

The semantic textual similarity (STS) task seeks to capture the notion that some texts are

more similar than others; STS seeks to measure the degree of semantic equivalence [Agirre

et al., 2013]. STS differs from paraphrasing in it is not a yes or no decision; rather STS

allows for 5 degrees of similarity.

  

| Model name | Parameters  | Pearson Correlation |
| -------------- | ----------------------------------------- | ------------------- |
| MT-DNN-SMART | State-of-the-art single task model  | 0.929 |
| Synthetic Data | `--sst_train data/ids-sst-train-syn3.csv` | 0.875 |
| Tagging  | `--additional_input`  | 0.872 |
| SophiaH  | `--optimizer sophiah` | 0.870 |
| Baseline | | 0.866 |
  
## Results for BART Detection

## Methodologies for BERT

  

This section describes the methodology used in our experiments to extend the training of the multitask BERT model to the

three tasks of paraphrase identification, sentiment classification, and semantic textual similarity.

  

---

  

### POS and NER Tag Embeddings

  

Based on Bojanowski, et al. ([Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)), which showed that the

addition of subword information to word embeddings can improve performance on downstream tasks, we extended our approach

by incorporating Part-of-Speech (POS) and Named Entity Recognition (NER) tag embeddings into the input representation.

The primary goal was to investigate whether the inclusion of linguistic information could lead to improved performance

on the tasks.

  

#### Tagging

  

For the efficient and accurate tagging of POS and NER, we used the [spaCy](https://spacy.io/) library. The tagging

process occurs during data preprocessing, where each sentence is tokenized into individual words. The spaCy pipeline is

then invoked to annotate each word with its corresponding POS tag and NER label. The resulting tags and labels are

subsequently converted into embeddings.

  

To increase training efficiency, we implemented a caching mechanism where the computed tag embeddings were stored and

reused across multiple epochs.

  

#### Experimental Results

  

Contrary to our initial expectations, the inclusion of POS and NER tag embeddings did not yield the desired improvements

across the three tasks. Experimental results indicated that the performance either remained stagnant or only slightly

improved compared to the baseline BERT model without tag embeddings.

  

#### Impact on Training Process

  

An additional observation was the notable increase in training time when incorporating POS and NER tag embeddings. This

extended training time was attributed to the additional computational overhead required for generating and embedding the

tags.

  

#### Conclusion

  

Although the integration of POS and NER tag embeddings initially seemed promising, our experiments showed that this

approach did not contribute significantly to the performance across the tasks. The training process was noticeably slowed down by the

inclusion of tag embeddings.

  

As a result, we concluded that the benefits of incorporating POS and NER tags were not substantial enough to justify the

extended training time. Future research could explore alternative ways of effectively exploiting linguistic features

while minimising the associated computational overhead.

  

One possible explanation for the lack of performance improvements could be that the BERT model already encodes some

syntactic information in its word

embeddings. Hewitt and Manning ([A Structural Probe for Finding Syntax in Word Representations](https://aclanthology.org/N19-1419.pdf))

showed that some syntactic information is already encoded in the word embeddings of pretrained BERT models, which could

explain why the inclusion of POS and NER tags did not lead to performance improvements.

  

---

  

### Sophia

  

We implemented the Sophia (**S**econd-**o**rder Cli**p**ped Stoc**h**astic Opt**i**miz**a**tion) optimizer completly

from scratch, which is a second-order optimizer for language model pre-training. The paper promises convergence twice as

fast as AdamW and better generalisation performance. It uses a light weight estimate of the diagonal of the Hessian

matrix to approximate the curvature of the loss function. It also uses clipping to control the worst-case update size.

By only updating the Hessian estimate every few iterations, the overhead is negligible.

  

The optimizer was introduced recently in the

paper [Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training](https://arxiv.org/abs/2305.14342).

  

#### Implementation

  

The paper describes the optimizer in detail, but does not provide any usable code. We implemented the optimizer from

scratch in PyTorch. The optimizer is implemented in the [`optimizer.py`](optimizer.py) file and can be used in the

multitask classifier by setting the `--optimizer` parameter.

  

There are two ways of estimating the Hessian. The first option is to use the Gauss-Newton-Bartlett approximation, which

is computed using an average over the minibatch gradients. However, this estimator requires the existence of a

multi-class classification problem from which to sample. This is not the case for some of our tasks, e.g. STS, which is

a regression task. The estimator is still implemented as `SophiaG`.

  

The second option is to use Hutchinson's unbiased estimator of the Hessian diagonal by sampling from a spherical

Gaussian distribution. This estimator is implemented as `SophiaH`. This estimator can be used for all tasks. It requires

a Hessian vector product, which is implemented in most modern deep learning frameworks, including PyTorch.

  

#### Convergence

  

While the implementation of this novel optimizer was a challenge, in the end, we were able to implement it **successfully** and the model was able to train well. However, we did not observe any improvements in performance. The optimizer did

not converge faster than AdamW, and the performance was comparable. This could be due to the fact that the optimizer was

designed for pre-training language models, which is a different task to ours.

  

A more recent paper studing different training algorithms for transformer-based language

models by Kaddour et al. ([No Train No Gain: Revisiting Efficient Training Algorithms For Transformer-based Language Models](https://arxiv.org/pdf/2307.06440.pdf))

comes to the conclusion that the training algorithm gains vanish with a fully decayed learning rate. They show

performance being about the same as the baseline (AdamW), which is what we observed.

  

---

  

### Synthetic Data Augmentation

  

Given recent advances in imitation learning - in particular, the demonstrated ability of compact language models to emulate the performance of their larger, proprietary counterparts ([Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)) - we investigated the impact of synthetic data in improving multitask classification models. Our focus lied on sentiment classification, where we were the weakest and had the fewest training examples.

  

#### LLM Generation

  

A custom small language model produced suboptimal data at the basic level, displaying instances beyond its distribution, and struggled to utilise the training data, resulting in unusual outputs.

  

We employed OpenAI's GPT-2 medium model variant ([Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)) and adapted it with a consistent learning rate using our sentiment classification training dataset. This modified model subsequently produced 100,000 training occurrences, which were ten times greater than the primary dataset. Although the produced illustrations were more significant to the context than the earlier technique, they still had infrequent coherence discrepancies.

  

For our third plan, we asked [GPT-4](https://arxiv.org/abs/2303.08774) to produce new examples.

The data obtained from GPT-4 are of the highest quality. could only collect a restricted amount of data (~500 instances) due to ChatGPT's limitations and GPT-4's confidential nature.

  

#### Caution: Synthetic Data

  

OpenAI's GPT-2 and GPT-4, were trained on undisclosed datasets, posing potential concerns about data overlaps with our sentiment classification set. Even though these models are unlikely to reproduce particular test set instances, the concern remains and should be addressed.

  

#### Results with Synthetic Data

  

It's important to mention that our model didn't overfit on the training set, even after 30 epochs with 100.000 synthetic instances from GPT2. The methods used didn't improve the validation accuracy beyond what our best model already achieved. Additionally, performance worsened on the task with synthetic data.

However, we believe that the synthetic data augmentation approach has potential and could be further explored in future research, especially with larger models like GPT-4.

  

---

  

### Details

  

The first dataset we received for training was heavily unbalanced, with one set containing an order of magnitude more samples than the other. This imbalance can make models unfairly biased, often skewing results towards the majority class. Instead of using every available sample in each training epoch, which would be both time consuming and inefficient, we made modifications to the dataloader. In each epoch, we randomly select a fixed number of samples. This number was chosen to be 10.000, which is a good compromise between training time and performance.

  

<details>

<summary>A lot more about the model architecture.</summary>

  

#### Classifier

  

The design and selection of classifiers are crucial in multi-task learning, especially when the tasks are deeply

intertwined. The performance of one classifier can cascade its effects onto others, either enhancing the overall results

or, conversely, dragging them down. In our endeavor, we dedicated significant time to experimentation, aiming to ensure

not only the individual performance of each classifier but also their harmonious interaction within the multi-task

setup.

  

Some of the components of our multitask classifier are described in more detail below. Each classifier's architecture is

tailored to the unique characteristics of its task, enabling our multi-task learning framework to address multiple NLP

challenges simultaneously.

  

##### Attention Layer

  

The attention mechanism plays a major role in capturing and emphasizing salient information within the output embeddings

generated by the BERT model. We implemented

an `AttentionLayer` ([Attention Is All You Need](https://arxiv.org/abs/1706.03762)) that accepts the last hidden state

of the BERT output and applies a weighted sum mechanism to enhance the importance of certain tokens while suppressing

others. This layer aids in creating a more focused representation of the input sentence, which is crucial for downstream

tasks.

  

##### Sentiment Analysis Classifier

  

This classifier architecture consists of several linear layers that refine the BERT embeddings into logits corresponding

to each sentiment class. These logits are then used to compute the predicted sentiment label. Achieving a balance here

was crucial, as any inefficiencies could potentially impact the overall performance of our multi-task framework.

  

##### Paraphrase Detection Classifier

  

The paraphrase detection classifier uses a two-step process. First, the BERT embeddings for each input sentence are

processed separately by a linear layer. We then compute the absolute difference and the absolute sum of these processed

embeddings. These two concatenated features are then fed through additional linear layers to generate logits for

paraphrase prediction. Iterative refinement was crucial here, ensuring that the classifier neither overshadowed nor was

overshadowed by the other tasks.

  

##### Semantic Textual Similarity Estimator

  

For the Semantic Textual Similarity task, our approach relies on cosine similarity. The BERT embeddings for the input

sentences are generated and then compared using cosine similarity. The resulting similarity score is scaled to range

between 0 and 5, providing an estimate of how semantically similar the two sentences are.

  

#### Layer Unfreeze

  

Layer unfreezing is a technique employed during fine-tuning large pre-trained models like BERT. The idea behind

this method is to gradually unfreeze layers of the model during the training process. Initially, the top layers are trained while the bottom layers are frozen. As training progresses, more layers are incrementally

unfrozen, allowing for deeper layers of the model to be adjusted.

  

One of the motivations to use layer unfreezing is to prevent *catastrophic forgetting*—a phenomenon where the model

rapidly forgets its previously learned representations when fine-tuned on a new

task ([Howard and Ruder](https://arxiv.org/abs/1801.06146)). By incrementally unfreezing the layers, the hope is to

preserve valuable pretrained representations in the earlier layers while allowing the model to adapt to the new task.

  

In our implementation, we saw a decrease in performance. One possible

reason for this could be the interaction between the layer thaw schedule and the learning rate scheduler (plateau). As the

learning rate scheduler reduced the learning rate, not all layers were yet unfrozen. This mismatch may have hindered

the model's ability to make effective adjustments to the newly unfrozen layers. As a result, the benefits expected from the

unfreezing layers may have been offset by this unintended interaction.

  

#### Mixture of Experts

  

Inspired by suggestions that GPT-4 uses a Mixture of Experts (MoE) structure, we also investigated the possibility of integrating MoE into our multitasking classification model. Unlike traditional, single-piece structures, the MoE design is made up of multiple of specialised "expert" sub-models, each adjusted to handle a different section of the data range.

  

Our use of the MoE model includes three expert sub-models, each using a independent BERT architecture. Also, we use a fourth BERT model for three-way classification, which acts as the gating mechanism for the group.

Two types of gating were studied - Soft Gate, which employs a Softmax function to consider the contributions of each expert, and Hard Gate, which only permits the expert model with the highest score to affect the final prediction.

  

Despite the theoretical benefits of a MoE approach, our experimental findings did not result in any enhancements in performance over our top-performing standard models and we quickly abandoned the idea.

  

#### Automatic Mixed Precision

  

The automatic mixed precision (AMP) feature of PyTorch was used to speed up training and reduce memory usage. This feature changes the precision of the model's weights and activations during training. The model was trained in `bfloat16` precision, which is a fast 16-bit floating point format. The AMP feature of PyTorch automatically casts the model parameters. This reduces the memory usage and speeds up training.

  

</details>

## Methodologies for BART Detection  



## Experiments for BERT

  

We used the default datasets provided for training and validation with no modifications.

  

The baseline for our comparisons includes most smaller improvements to the BERT model listed above. The baseline model is further described in the [Results](#results) section. The baseline model was trained for 10 epochs at 10.000 samples per epoch.

  

The models were trained and evaluated on the Grete cluster. The training was done on a single A100 GPU. The training time for the baseline model was approximately 1 hour.

  

We used [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) to perform hyperparameter tuning. This allowed us to efficiently explore the hyperparameter space and find the best hyperparameters for our model. We used [Optuna](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.search.optuna.OptunaSearch.html) to search the hyperparameter space and [AsyncHyperBandScheduler](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.AsyncHyperBandScheduler.html) as the scheduler. The hyperparameters were searched for the whole model, not for each task individually. This was done to avoid overfitting to a single task. We searched for hyperparameters trying to minimize the overfitting of the model to the training data.

  

<div align="center"><img src="https://media.discordapp.net/attachments/1146522094067269715/1146523064763437096/image.png?width=1440&height=678" alt="Hyperparameter Search to find a baseline" width="600"/></div>

  

The trained models were evaluated on the validation set. The best model was selected based on the validation results ('dev'). The metrics used for the evaluation were accuracy only for paraphrase identification and sentiment classification, and Pearson correlation for semantic textual similarity.


## Experiments for BART Generation


## Experiments for BART Detection


## Contributors


| Injamam  | Rehan | Saad | Reda | Smitesh |
| ---------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| Tagging  | Sophia Optimizer  | Synthetic Data  | Synthetic Data  | Synthetic Data  |
| Layer Unfreeze | Hyperparameter Tuning | | Synthetic Data  |
| Classifier Model | Repository  | | Synthetic Data  |

  

### Grete Cluster

  
To run the multitask classifier on the Grete cluster you can use the `run_train.sh` script. You can change the

parameters in the script to your liking. To submit the script use

  
for BERT


````sh

sbatch run_train.sh

````


for BART Detection

  
````sh

sbatch run_bart_detection.sh

````


To check on your job you can use the following command


```sh

squeue --me

```

The logs of your job will be saved in the `slurm_files` directory. The best model will be saved in the `models` directory.


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
