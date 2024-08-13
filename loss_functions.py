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


