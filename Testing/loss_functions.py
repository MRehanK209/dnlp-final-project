from torch.nn import functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# In the training loop:
criterion = FocalLoss()
l2_lambda = 0.01  # L2 regularization strength

# ... (rest of the loop)
loss = criterion(logits, b_labels.float())

# Add L2 regularization
l2_reg = torch.tensor(0.).to(device)
for param in model.parameters():
    l2_reg += torch.norm(param)
loss += l2_lambda * l2_reg

loss.backward()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# In the training loop:
criterion = LabelSmoothingLoss(classes=2, smoothing=0.1)

# ... (rest of the loop)
loss = criterion(logits, b_labels.long())


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

# In the training loop:
criterion = ContrastiveLoss()

# ... (rest of the loop)
embeddings1 = model.get_embeddings(b_ids_1, b_mask_1)
embeddings2 = model.get_embeddings(b_ids_2, b_mask_2)
loss = criterion(embeddings1, embeddings2, b_labels.float())


class SMARTLossParaphrase(torch.nn.Module):
    def __init__(self, model, epsilon=1e-3, num_steps=1, step_size=1e-3):
        super().__init__()
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, labels):
        # Original predictions
        orig_logits = self.model.predict_paraphrase(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)

        # Generate adversarial perturbations
        embed_1 = self.model.bert.embeddings.word_embeddings(input_ids_1)
        embed_2 = self.model.bert.embeddings.word_embeddings(input_ids_2)
        delta_1 = torch.zeros_like(embed_1).uniform_(-self.epsilon, self.epsilon)
        delta_2 = torch.zeros_like(embed_2).uniform_(-self.epsilon, self.epsilon)
        delta_1.requires_grad_()
        delta_2.requires_grad_()

        for _ in range(self.num_steps):
            adv_logits = self.model.predict_paraphrase(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, delta_1, delta_2)
            
            adv_loss = F.binary_cross_entropy_with_logits(adv_logits, torch.sigmoid(orig_logits.detach()))
            delta_grad_1, delta_grad_2 = torch.autograd.grad(adv_loss, (delta_1, delta_2))
            
            delta_1.data = delta_1.data + self.step_size * delta_grad_1.sign()
            delta_2.data = delta_2.data + self.step_size * delta_grad_2.sign()
            delta_1.data = torch.clamp(delta_1.data, -self.epsilon, self.epsilon)
            delta_2.data = torch.clamp(delta_2.data, -self.epsilon, self.epsilon)

        # Final adversarial logits
        adv_logits = self.model.predict_paraphrase(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, delta_1.detach(), delta_2.detach())

        # Symmetric KL divergence
        sym_kl = F.kl_div(
            F.logsigmoid(adv_logits),
            torch.sigmoid(orig_logits),
            reduction='batchmean'
        ) + F.kl_div(
            F.logsigmoid(orig_logits),
            torch.sigmoid(adv_logits),
            reduction='batchmean'
        )

        # Original loss (binary cross-entropy)
        orig_loss = F.binary_cross_entropy_with_logits(orig_logits, labels.float())

        return orig_loss + sym_kl
    
    def get_embeddings(self, input_ids, attention_mask, delta=None):
        if delta is not None:
            embeddings = self.bert.embeddings.word_embeddings(input_ids)
            embeddings = embeddings + delta
            outputs = self.bert(inputs_embeds=embeddings, attention_mask=attention_mask)
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs['pooler_output']
    def predict_paraphrase(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, delta_1=None, delta_2=None):
        embeddings_1 = self.get_embeddings(input_ids_1, attention_mask_1, delta_1)
        embeddings_2 = self.get_embeddings(input_ids_2, attention_mask_2, delta_2)
        concatenated = torch.cat((embeddings_1, embeddings_2), dim=1)
        logits = self.paraphrase_classifier(concatenated)
        return logits.squeeze()
    

    class BertCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, query_states, key_states, attention_mask=None):
        # Compute query, key, and value
        query = self.query(query_states)
        key = self.key(key_states)
        value = self.value(key_states)

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / (self.config.hidden_size ** 0.5)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Apply softmax and dropout
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # Compute weighted sum of values
        context_layer = torch.matmul(attention_probs, value)

        return context_layer


class SMARTFocalLoss(Module):
    def __init__(self, model, epsilon=1e-6, num_steps=5, step_size=1e-3, alpha=1, gamma=2, reduction='mean'):
        super(SMARTFocalLoss, self).__init__()
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels):
        # Calculate the original logits
        logits = self.model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)

        # Calculate the focal loss on the original logits
        focal_loss = self.focal_loss(logits, b_labels)

        # SMART perturbation
        b_ids_1_perturbed, b_ids_2_perturbed = self.perturb_inputs(b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels)
        perturbed_logits = self.model.predict_paraphrase(b_ids_1_perturbed, b_mask_1, b_ids_2_perturbed, b_mask_2)

        # Focal loss on the perturbed logits
        focal_loss_perturbed = self.focal_loss(perturbed_logits, b_labels)

        # SMART regularization: Kullback-Leibler divergence between original and perturbed logits
        kl_div_loss = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(perturbed_logits, dim=-1), reduction='batchmean')

        # Total loss: original focal loss + SMART regularization term + perturbed focal loss
        total_loss = focal_loss + kl_div_loss + focal_loss_perturbed

        return total_loss

    def perturb_inputs(self, b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels):
        b_ids_1_perturbed = b_ids_1.detach().clone().float().requires_grad_(True)
        b_ids_2_perturbed = b_ids_2.detach().clone().float().requires_grad_(True)

        for _ in range(self.num_steps):
            logits = self.model.predict_paraphrase(b_ids_1_perturbed, b_mask_1, b_ids_2_perturbed, b_mask_2)
            loss = F.cross_entropy(logits, b_labels)
            loss.backward()

            # Perturbation for b_ids_1
            grad_1 = b_ids_1_perturbed.grad
            grad_1 = torch.sign(grad_1) * self.step_size
            b_ids_1_perturbed = b_ids_1_perturbed + grad_1
            b_ids_1_perturbed = torch.clamp(b_ids_1_perturbed, min=-self.epsilon, max=self.epsilon)
            b_ids_1_perturbed = b_ids_1_perturbed.detach().clone().float().requires_grad_(True)

            # Perturbation for b_ids_2
            grad_2 = b_ids_2_perturbed.grad
            grad_2 = torch.sign(grad_2) * self.step_size
            b_ids_2_perturbed = b_ids_2_perturbed + grad_2
            b_ids_2_perturbed = torch.clamp(b_ids_2_perturbed, min=-self.epsilon, max=self.epsilon)
            b_ids_2_perturbed = b_ids_2_perturbed.detach().clone().float().requires_grad_(True)

        return b_ids_1_perturbed, b_ids_2_perturbed

    def focal_loss(self, logits, targets):
        targets = targets.float()
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
        

        def forward(self, input_ids, attention_mask):
        """
        input_ids: [batch_size, seq_len], seq_len is the max length of the batch
        attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
        """
        # get the embedding for each input token
        embedding_output = self.embed(input_ids=input_ids)
        masked_input_ids, mask = self.create_dynamic_mask(input_ids)
        embedding_output = self.embed(input_ids=masked_input_ids)
        # feed to a transformer (a stack of BertLayers)
        sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

        # get cls token hidden state
        first_tk = sequence_output[:, 0]
        first_tk = self.pooler_dense(first_tk)
        first_tk = self.pooler_af(first_tk)

        return {"last_hidden_state": sequence_output, "pooler_output": first_tk}



def predict_paraphrase(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        embeddings_1 = self.get_embeddings(input_ids_1, attention_mask_1)
        embeddings_2 = self.get_embeddings(input_ids_2, attention_mask_2)
        concatenated = torch.cat((embeddings_1, embeddings_2), dim=1)
        logits = self.paraphrase_classifier(concatenated)
        #probs = F.softmax(logits, dim=-1)
        return logits.squeeze()
        #return probs


if args.task == "qqp" or args.task == "multitask":
            # Train the model on the qqp dataset.
            all_pred=[]
            all_labels=[]
            for batch in tqdm(
                    quora_train_dataloader, desc=f"train-{epoch + 1:02}", disable=TQDM_DISABLE
            ):
                b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (
                    batch["token_ids_1"],
                    batch["attention_mask_1"],
                    batch["token_ids_2"],
                    batch["attention_mask_2"],
                    batch["labels"],
                )

                b_ids_1 = b_ids_1.to(device)
                b_mask_1 = b_mask_1.to(device)
                b_ids_2 = b_ids_2.to(device)
                b_mask_2 = b_mask_2.to(device)
                b_labels = b_labels.to(device)

                # Apply mixup to both sentences
                #mixed_input_ids_1, mixed_attention_mask_1, y_a_1, y_b_1, lam_1 = mixup_data(b_ids_1, b_mask_1, b_labels)
                #mixed_input_ids_2, mixed_attention_mask_2, y_a_2, y_b_2, lam_2 = mixup_data(b_ids_2, b_mask_2, b_labels)

                # Average the lambdas
                #lam = (lam_1 + lam_2) / 2

                optimizer.zero_grad()
                # Store original parameters before update
                original_params = [param.clone() for param in model.parameters()]
                #embeddings_1 = model.get_embeddings(b_ids_1, b_mask_1)
                #embeddings_2 = model.get_embeddings(b_ids_2, b_mask_2)
                #concatenated_embeddings = torch.cat((embeddings_1, embeddings_2), dim=1)
                 # Convert binary labels to {-1, 1} for CosineEmbeddingLoss
                #target = (2 * b_labels - 1).float()
        
                #loss = nn.CosineEmbeddingLoss()(embeddings1, embeddings2, target)
                
                logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                #def multiple_negative_ranking_loss(logits, labels, margin=1.0):
                #labels=torch.tensor(b_labels)
                #margin=1.0
    # Separate positive logits (where label is 1) and negative logits (where label is 0)
                #positive_logits = logits[labels == 1]
                #negative_logits = logits[labels == 0]
    
    # Compute pairwise differences between positive and negative logits
                #loss_MNR = torch.relu(margin + negative_logits.unsqueeze(0) - positive_logits.unsqueeze(1))
    
    # Average over all the samples
                #labels = labels.float() * 2 - 1  # Convert to -1/1
    
    # Calculate the hinge loss
                #loss_hinge = torch.clamp(1 - labels * logits, min=0)
                #loss = loss_hinge.mean()
                    #return loss
                #loss = F.kl_div(logits, b_labels.float(), reduction= 'batchmean')
                #pos_weight = torch.tensor([2.0]).to(device)
                #loss = F.binary_cross_entropy_with_logits(logits, b_labels.float(), pos_weight=pos_weight)
                #loss=criterion(logits, b_labels.float())
                  # Or your chosen loss function)
                #loss = model.smart_loss_qqp(concatenated_embeddings, logits)
                #loss_f = keras.losses.BinaryFocalCrossentropy(
                    #gamma=2, from_logits=True)
                #loss=loss_f(logits, b_labels.float())
                #loss = multiple_negative_ranking_loss(logits, b_labels, margin=1.0)
                #smart_loss_paraphrase = SMARTLossParaphrase(model, epsilon=args.smart_epsilon, num_steps=args.smart_num_steps, step_size=args.smart_step_size)
                #loss = smart_loss_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels)
                
                # Forward pass with mixed inputs
                #logits = model.predict_paraphrase(mixed_input_ids_1, mixed_attention_mask_1, mixed_input_ids_2, mixed_attention_mask_2)
                #criterion_1 = F.binary_cross_entropy_with_logits(logits, y_a_1.float())
                #criterion_2 = F.binary_cross_entropy_with_logits(logits, y_a_2.float())
                # Compute loss using mixup criterion
                #loss = lam*criterion_1 +(1-lam)*criterion_2
                #model = MultitaskBERT(config)
                #model.to(device)
                #criterion_SMART = SMARTFocalLoss(model, epsilon=1e-6, num_steps=5, step_size=1e-3, alpha=0.25, gamma=2, reduction='mean')
                #loss = criterion(logits, b_labels.float())
                loss=criterion_LabelSmoothing(logits, b_labels.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

        train_loss = train_loss / num_batches


    def predict_paraphrase(self, input_ids_1, attention_mask_1, input_ids_2,  attention_mask_2):
        # Concatenate the input IDs and attention masks with a [SEP] token
        sep_token_id = self.bert.config.sep_token_id
        cls_token_id = self.bert.config.bos_token_id
        maxlength=max(attention_mask_1.size(1), attention_mask_2.size(1)) + 3 
        # Create concatenated input ids
        input_ids = torch.cat(
            [torch.full((input_ids_1.size(0), 1), cls_token_id, device=input_ids_1.device), 
             input_ids_1, 
             torch.full((input_ids_1.size(0), 1), sep_token_id, device=input_ids_1.device), 
             input_ids_2, 
             torch.full((input_ids_1.size(0), 1), sep_token_id, device=input_ids_1.device)], 
            dim=1
        )
         # Ensure that attention masks are 2D (batch_size, sequence_length)
        attention_mask_1 = attention_mask_1 if attention_mask_1.dim() == 2 else attention_mask_1.unsqueeze(0)
        attention_mask_2 = attention_mask_2 if attention_mask_2.dim() == 2 else attention_mask_2.unsqueeze(0)

        # Adjust attention masks accordingly
        attention_mask_1_padded = torch.cat(
            [attention_mask_1, torch.zeros(maxlength - attention_mask_1.size(1), dtype=torch.long, device=attention_mask_1.device)],
            dim=1
)[:, :maxlength]
        attention_mask_2_padded = torch.cat(
            [attention_mask_2, torch.zeros(maxlength - attention_mask_2.size(1), dtype=torch.long, device=attention_mask_2.device)],
            dim=1
)[:, :maxlength]
        print(f"input_ids shape: {input_ids.shape}")
        print(f"attention_mask_1 shape: {attention_mask_1.shape}")
        print(f"attention_mask_2 shape: {attention_mask_2.shape}")
        print(f"attention_mask_1_padded shape: {attention_mask_1_padded.shape}")
        print(f"attention_mask_2_padded shape: {attention_mask_2_padded.shape}")
# Create concatenated attention mask
        attention_mask = torch.cat(
            [
                torch.ones((attention_mask_1.size(0), 1), dtype=torch.long, device=attention_mask_1.device),
                attention_mask_1_padded,
                torch.ones((attention_mask_1.size(0), 1), dtype=torch.long, device=attention_mask_1.device),
                attention_mask_2_padded,
                torch.ones((attention_mask_1.size(0), 1), dtype=torch.long, device=attention_mask_1.device)
            ], 
            dim=1
)[:, :maxlength]
        
        # Ensure input_ids and attention_mask are truncated to max_length
        if input_ids.size(1) > maxlength:
            input_ids = input_ids[:, :maxlength]
            attention_mask = attention_mask[:, :maxlength]
        # Forward pass through BERT
        outputs = self.bert(input_ids, attention_mask)
        # Get the pooler output from the [CLS] token
        pooled_output = outputs["pooler_output"]
        # Pass through the classifier
        logits = self.classifier(pooled_output)
        # Apply sigmoid to get the final prediction between 0 and 1
        probabilities = self.sigmoid(logits)
        return probabilities




class SentencePairDataset(Dataset):
    def __init__(self, dataset, args, isRegression=False):
        self.dataset = dataset
        self.p = args
        self.isRegression = isRegression
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", local_files_only=args.local_files_only
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        labels = [x[2] for x in data]
        sent_ids = [x[3] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors="pt", padding=True, truncation=True, max_length=512)
        encoding2 = self.tokenizer(sent2, return_tensors="pt", padding=True, truncation=True, max_length=512)

        token_ids = torch.LongTensor(encoding1["input_ids"])
        attention_mask = torch.LongTensor(encoding1["attention_mask"])
        token_type_ids = torch.LongTensor(encoding1["token_type_ids"])

        token_ids2 = torch.LongTensor(encoding2["input_ids"])
        attention_mask2 = torch.LongTensor(encoding2["attention_mask"])
        token_type_ids2 = torch.LongTensor(encoding2["token_type_ids"])
        if self.isRegression:
            labels = torch.DoubleTensor(labels)
        else:
            labels = torch.LongTensor(labels)

        return (
            token_ids,
            token_type_ids,
            attention_mask,
            token_ids2,
            token_type_ids2,
            attention_mask2,
            labels,
            sent_ids,
        )

    def collate_fn(self, all_data):
        (
            token_ids,
            token_type_ids,
            attention_mask,
            token_ids2,
            token_type_ids2,
            attention_mask2,
            labels,
            sent_ids,
        ) = self.pad_data(all_data)

        batched_data = {
            "token_ids_1": token_ids,
            "token_type_ids_1": token_type_ids,
            "attention_mask_1": attention_mask,
            "token_ids_2": token_ids2,
            "token_type_ids_2": token_type_ids2,
            "attention_mask_2": attention_mask2,
            "labels": labels,
            "sent_ids": sent_ids,
        }

        return batched_data
    

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Apply bias correction
                if group["correct_bias"]:
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1
                else:
                    step_size = group["lr"]

                denom = exp_avg_sq.sqrt().add_(group["eps"])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                if group["weight_decay"] != 0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss




best_dev_acc = 0
best_hyperparams = {}

for alpha in alpha_range:
    for gamma in gamma_range:
        for lambda_reg in lambda_reg_range:
            for lr in learning_rates:
                # Reset model and optimizer
                model = YourModelClass(config)  # Initialize your model
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
                
                print(f"Training with alpha={alpha}, gamma={gamma}, lambda_reg={lambda_reg}, lr={lr}")

                # Your existing training loop goes here
                for epoch in range(num_epochs):
                    # ... (your existing epoch loop)

                    criterion_LabelSmoothing = FocalLoss_LabelSmoothing(alpha=alpha, gamma=gamma, label_smoothing=0.0)

                    # ... (rest of your training code)

                    loss = smoothness_inducing_loss(model, input_ids_1=b_ids_1, attention_mask_1=b_mask_1, 
                            input_ids_2=b_ids_2, attention_mask_2=b_mask_2,
                            labels=b_labels,
                            criterion=criterion_LabelSmoothing,
                            epsilon=1e-10,
                            lambda_reg=lambda_reg
                    )

                    # ... (rest of your epoch loop)

                # After training, evaluate on dev set
                _, dev_acc = model_eval_multitask(
                    sst_dev_dataloader,
                    quora_dev_dataloader,
                    sts_dev_dataloader,
                    etpc_dev_dataloader,
                    model=model,
                    device=device,
                    task=args.task,
                )

                # Update best model if necessary
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    best_hyperparams = {
                        'alpha': alpha,
                        'gamma': gamma,
                        'lambda_reg': lambda_reg,
                        'learning_rate': lr
                    }
                    save_model(model, optimizer, args, config, args.filepath)

print(f"Best dev accuracy: {best_dev_acc}")
print(f"Best hyperparameters: {best_hyperparams}")




def dynamic_masking(self, input_ids):
        """
        Apply dynamic masking to the input IDs.
        
        Args:
            input_ids (torch.Tensor): Input token IDs.
        
        Returns:
            torch.Tensor: Masked input token IDs.
        """
        masked_input_ids = input_ids.clone()
        for i in range(masked_input_ids.size(0)):
            for j in range(masked_input_ids.size(1)):
                if random.random() < self.mask_prob and masked_input_ids[i, j] not in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]:
                    masked_input_ids[i, j] = self.tokenizer.mask_token_id
        return masked_input_ids

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        labels = [x[2] for x in data]
        sent_ids = [x[3] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors="pt", padding=True, truncation=True, max_length=512)
        encoding2 = self.tokenizer(sent2, return_tensors="pt", padding=True, truncation=True, max_length=512)

        token_ids1 = torch.LongTensor(encoding1["input_ids"])
        attention_mask1 = torch.LongTensor(encoding1["attention_mask"])
        token_type_ids1 = torch.LongTensor(encoding1["token_type_ids"])

        token_ids2 = torch.LongTensor(encoding2["input_ids"])
        attention_mask2 = torch.LongTensor(encoding2["attention_mask"])
        token_type_ids2 = torch.LongTensor(encoding2["token_type_ids"])

        # Apply dynamic masking to token IDs
        token_ids1 = self.dynamic_masking(token_ids1)
        token_ids2 = self.dynamic_masking(token_ids2)

        if self.isRegression:
            labels = torch.DoubleTensor(labels)
        else:
            labels = torch.LongTensor(labels)

        return (
            token_ids1,
            token_type_ids1,
            attention_mask1,
            token_ids2,
            token_type_ids2,
            attention_mask2,
            labels,
            sent_ids,
        )

    def collate_fn(self, all_data):
        (
            token_ids1,
            token_type_ids1,
            attention_mask1,
            token_ids2,
            token_type_ids2,
            attention_mask2,
            labels,
            sent_ids,
        ) = self.pad_data(all_data)

        batched_data = {
            "token_ids_1": token_ids1,
            "token_type_ids_1": token_type_ids1,
            "attention_mask_1": attention_mask1,
            "token_ids_2": token_ids2,
            "token_type_ids_2": token_type_ids2,
            "attention_mask_2": attention_mask2,
            "labels": labels,
            "sent_ids": sent_ids,
        }

        return batched_data
