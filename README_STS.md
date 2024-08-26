Our multitask model achieves the following performance on:

### [Semantic Textual Similarity on STS]

The semantic textual similarity (STS) task seeks to capture the notion that some texts are
more similar than others; STS seeks to measure the degree of semantic equivalence [Agirre
et al., 2013]. Unlike paraphrasing, which is a binary decision of whether two texts are the same or not, STS allows for 5 degrees of similarity.

| Model name                      | Parameters                                | Pearson Correlation |
| ------------------------------- | ----------------------------------------- | -------- |
| Baseline                        |                                           | 0.375    |
| Scalling Similarity scores between 0 and 1  |    | 0.375   |
| Adding additional dense layers on top of the BERT embeddings                         |                    | 0.370  |
| Regularization                  | `--lr 1e-05 --hidden_dropout_prob 0.3`    | 0.370    |
| SophiaH                         |                                           | 0.365  |
| Increased Learning rate | `--lr 8e-05 --hidden_dropout_prob 0.1`           |  0.363    |

We experimented with various improvements to enhance the model's performance beyond the baseline:

### Regularization & Learning Rate:
Regularization techniques were applied to prevent overfitting and stabilize the model's training process. We specifically adjusted the hidden_dropout_prob parameter to 0.3 and tried different learning rates. Despite these efforts, the best Pearson correlation achieved was 0.370, which did not surpass the baseline.

### AdamW & SophiaH Optimizers:
We explored alternative optimizers like SophiaH, to optimize the training process. While AdamW was part of our baseline setup, SophiaH was evaluated as a potential improvement. However, SophiaH led to a slightly lower correlation of 0.365, indicating it did not outperform the baseline.

#### Data Scaling:
We attempted to scale the similarity scores from the original range of 0 to 5 down to a continuous range between 0 and 1. The rationale was to normalize the data and potentially improve the model's learning. Unfortunately, this adjustment did not result in any performance gains, as the correlation remained at the baseline level of 0.375.

### BERT Architecture:
We experimented with modifying the BERT architecture by adding additional dense layers on top of the BERT embeddings. The goal was to enhance the model's capacity to capture complex relationships between sentences. However, this approach not only failed to improve the results but also significantly increased the computational cost, making it less feasible for practical applications.

### Conclusion:
Although our experiments did not yield improvements in performance, they provided valuable insights suggesting that more sophisticated techniques or task-specific adaptations might be necessary for meaningful gains. We also experimented with MT-DNN, but it did not work properly and failed to provide usable results. Despite the lack of improvement, these experiments offered valuable insights that could guide future research and optimization efforts.

---
