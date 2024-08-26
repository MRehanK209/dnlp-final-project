This project focuses on improving a BERT-based model for multitask (sst, sts and qp). We tuned various hyperparameters to achieve this balance, and the results of our experiments are summarized in this document.
## Requirements

To install requirements and all dependencies using conda, run:

```sh
./setup_gwdg.sh
```

The environment is activated with `conda activate dnlp`.

Alternatively, use the provided script `setup.sh`.
The script will create a new conda environment called `dnlp` and install all required packages based on weather you machine has gpu or not

## Training

To train the model, activate the environment and run this command:

```sh
python -u multitask_classifier.py --use_gpu
```

There are a lot of parameters that can be set. To see all of them, run `python multitask_classifier.py --help`. The most
important ones are:

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


## Evaluation

The model is evaluated after each epoch on the validation set. The results are printed to the console and saved in
the `logdir` directory. The best model is saved in the `models` directory.

## Results

As a Baseline of our model we chose the following hyperparameters. 

- mode: `finetune`
- epochs: `10`
- learning rate: `1e-5`
- optimizer: `AdamW`
- batch size: `32`

---

Our multitask model achieves the following performance on:

### [Sentiment Classification on Stanford Sentiment Treebank (SST)](https://paperswithcode.com/sota/sentiment-analysis-on-sst-5-fine-grained)

Understanding a text involves determining its polarity (positive, negative, or neutral). Sentiment analysis can reveal personal feelings about products, politicians, or news reports.
Each phrase is labeled as negative, slightly negative, neutral, slightly positive, or positive.

| Model name                      | Parameters                                | Accuracy |
| ------------------------------- | ----------------------------------------- | -------- |
| Baseline                        |                                           | 52.2%    |
| Increasing model complexity     | `--lr 1e-05 --hidden_dropout_prob 0.1`    | 52.6%    |
| Bregman proximal Point optim.   | `--lr 1e-05 --hidden_dropout_prob 0.3`    | 46.2%    |
| bert projected attention layer  | `--lr 1e-05 --hidden_dropout_prob 0.3`    | 46.2%    |
| SophiaH                         |                                           | 45.4%    |
| Synthetic Data                  | `--sst_train data/synthetic_data.csv`     | 51.3%    |
| Synonym Replacement             |                                           | 46.6%    |

We tried to play around various value of hyperparameters but the best output (52.6%) was obtained using learning rate 1e-05 and hidden deopout probability 0.1, using extra layer in model i.e. residual connection,layer normalization and dropout.  Output logs are in `slurm_files/highest.out`

## Methodology

This section describes the methodology used in our experiments to extend the training of the multitask BERT model to the
three tasks of paraphrase identification, sentiment classification, and semantic textual similarity.

---

#### Smoothness-Inducing Regularization

Initially the baseline score was 52.2%, after trying some methods to improve the perfomance like data augmentation, model was showing strong overfitting, that we tried to overcome using LRscheduling and EarlyStopping, but it was greatly complemented by the Smoothness-inducing regularization technique, which did not helped in improving the validation accuracy but helped model to avoid too strong overfitting on data.

#### AdamW, SophiaH, Bregman proximal Point optimization

Since various optimizers plat pivotal role based on task definition, we tried few of them to improve the model perfomance. There was not noticible difference between AdamW and SophiaH they both performed same while Bregman proximal Point optimization, it has addressed overfitting greatly but also hindered the overall learning of model, using Bregman proximal Point optimization model was getting accuracy of roughly 25% accuracy in initial epoch which is early half of that we got in AdamW and SophiaH i.e. roughly 45%

Sophia: [Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training](https://arxiv.org/abs/2305.14342).

Bregman proximal Point: [Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models Principled Regularized Optimization](https://aclanthology.org/2020.acl-main.197.pdf)

#### Data Augmentation (Synonym Replacement and Random Noise) & Synthetic Dataset

Given recent advances in imitation learning, particularly the demonstrated ability of compact language models to emulate the performance of larger, proprietary counterparts ([Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)), we investigated the role of synthetic data in improving multitask classification models. Our focus was on sentiment classification. WE geneated around 6000 more data points adding them to original dataset to have a larger training dataset. OpenAI's GPT-3 and GPT-4 were trained on undisclosed datasets, which raises concerns about data overlap with our sentiment classification set. Even though these models are unlikely to reproduce specific test set instances, the issue persists and should be addressed.But, it just contributed to overfit the model. Similar case was with the data augmentation techniques.

#### Conclusion

Even though we tried many things to improve the model accuracy there was negligible improvement from baseline when we introduced more layers in model like residual connection, layer normalization and dropout. the accuracy went from 52.2% to 52.6%, and thats the best accuracy we got for sst task after trying many things.

---