Our multitask model achieves the following performance on:

### [Semantic Textual Similarity on STS]

The semantic textual similarity (STS) task seeks to capture the notion that some texts are
more similar than others; STS seeks to measure the degree of semantic equivalence [Agirre
et al., 2013]. Unlike paraphrasing, which is a binary decision of whether two texts are the same or not, STS allows for 5 degrees of similarity.

| Model name                      | Parameters                                | Pearson Correlation |
| ------------------------------- | ----------------------------------------- | -------- |
| Baseline                        |                                           | 0.375    |
| Regularization                  | `--lr 1e-05 --hidden_dropout_prob 0.3`    | 0.370    |
| Increased Learning rate | `--lr 8e-05 --hidden_dropout_prob 0.1`           |  0.363    |
| Scalling Similarity scores between 0 and 1  | `--lr 1e-05 --hidden_dropout_prob 0.3`    | 0.375   |
| SophiaH                         |                                           | 0.365  |
| Adding additional dense layers on top of the BERT embeddings                         |                                           | 0.370  |

TEXT


#### Regularization & Learning rate

#### AdamW, SophiaH

#### Data scalling 

#### Bert Architecture

#### Conclusion


---
