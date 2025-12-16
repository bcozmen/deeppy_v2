import torch
import torch.nn as nn


def unique(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)


class HMLC(nn.Module):
    def __init__(self, temperatures = [0.07,0.07], layer_penalty=None, loss_type='hmce'):
        super(HMLC, self).__init__()
        self.temperatures = temperatures
        if not layer_penalty:
            self.layer_penalty = self.pow_2
        else:
            self.layer_penalty = layer_penalty
        self.sup_con_loss = SupConLoss()
        self.loss_type = loss_type
    def pow_2(self, value):
        return torch.pow(2, value)

    def forward(self, features, labels):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        mask = torch.ones(labels.shape).to(device)
        cumulative_loss = torch.tensor(0.0).to(device)
        max_loss_lower_layer = torch.tensor(float('-inf'))

        layer_losses = []
        layer_metrics = []

        for l, temperature in zip(range(1, labels.shape[1]), self.temperatures):
            mask[:, labels.shape[1]-l:] = 0
            layer_labels = labels * mask
            mask_labels = torch.stack([torch.all(torch.eq(layer_labels[i], layer_labels), dim=1)
                                       for i in range(layer_labels.shape[0])]).type(torch.uint8).to(device)

            layer_loss, metrics = self.sup_con_loss(features, mask=mask_labels, temperature=temperature)
            layer_metrics.append(metrics)
            layer_losses.append(layer_loss.item())

            if self.loss_type == 'hmc':
                cumulative_loss += self.layer_penalty(torch.tensor(
                  1/(l)).type(torch.float)) * layer_loss
            elif self.loss_type == 'hce':
                layer_loss = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)
                cumulative_loss += layer_loss
            elif self.loss_type == 'hmce':
                layer_loss = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)
                cumulative_loss += self.layer_penalty(torch.tensor(
                    1/l).type(torch.float)) * layer_loss
            else:
                raise NotImplementedError('Unknown loss')
            _, unique_indices = unique(layer_labels, dim=0)
            max_loss_lower_layer = torch.max(
                max_loss_lower_layer.to(layer_loss.device), layer_loss)
            labels = labels[unique_indices]
            mask = mask[unique_indices]
            features = features[unique_indices]
        
        return cumulative_loss / labels.shape[1], layer_losses, layer_metrics

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.contrast_mode = contrast_mode
    def forward(self, features, labels=None, mask=None, temperature = 0.07):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        # Normalize feature vectors along the embedding dimension.
        # Contrastive losses generally measure cosine similarity; normalizing
        # makes the loss invariant to vector norms and stabilizes training.

        features = nn.functional.normalize(features, dim=2)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast_before_temp = torch.matmul(anchor_feature, contrast_feature.T)
        anchor_dot_contrast = torch.div(anchor_dot_contrast_before_temp, temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        bool_mask = mask.clone().bool()
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob - ensure FP32 for numerical stability with AMP
        logits_fp32 = logits.float()
        logits_mask_fp32 = logits_mask.float()
        exp_logits = torch.exp(logits_fp32) * logits_mask_fp32
        log_prob = logits_fp32 - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # compute mean of log-likelihood over positive
        mask_fp32 = mask.float()
        mean_log_prob_pos = (mask_fp32 * log_prob).sum(1) / torch.clamp(mask_fp32.sum(1), min=1e-6)

        # loss
        loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        metrics = self._compute_metrics(anchor_dot_contrast_before_temp, bool_mask, logits_mask, temperature)
        return loss, metrics

    def _compute_metrics(self, similarity_matrix, pos_mask, logits_mask, temperature):
        """Compute diagnostic metrics for the similarity matrix.
        
        Args:
            similarity_matrix: [N*anchor_count, N*contrast_count] similarity scores (before temp scaling)
            pos_mask: [N*anchor_count, N*contrast_count] mask for positive pairs
            logits_mask: mask excluding self-contrasts
            anchor_count: number of anchor views
            batch_size: batch size
            
        Returns:
            Dictionary with various similarity metrics
        """
        with torch.no_grad():
            # similarity_matrix is the dot product of normalized vectors (cosine similarity in [-1, 1])
            # Do NOT apply temperature scaling to diagnostics like alignment/uniformity —
            # these metrics expect raw cosine similarities. Keep a scaled version for
            # anything that might need it (ranking/order is invariant to positive scaling).

            # Separate positive and negative similarities (use cosine similarities)
            pos_sims = similarity_matrix[pos_mask.bool()]
            neg_sims = similarity_matrix[(~pos_mask.bool()) & (logits_mask.bool())]
            
            metrics = {
                # Positive pair statistics
                'pos_sim_mean': pos_sims.mean().item() if len(pos_sims) > 0 else 0.0,
                'pos_sim_std': pos_sims.std().item() if len(pos_sims) > 0 else 0.0,
                'pos_sim_min': pos_sims.min().item() if len(pos_sims) > 0 else 0.0,
                'pos_sim_max': pos_sims.max().item() if len(pos_sims) > 0 else 0.0,
                
                # Negative pair statistics
                'neg_sim_mean': neg_sims.mean().item() if len(neg_sims) > 0 else 0.0,
                'neg_sim_std': neg_sims.std().item() if len(neg_sims) > 0 else 0.0,
                'neg_sim_min': neg_sims.min().item() if len(neg_sims) > 0 else 0.0,
                'neg_sim_max': neg_sims.max().item() if len(neg_sims) > 0 else 0.0,
                
                # Separation metrics
                'pos_neg_gap': (pos_sims.mean() - neg_sims.mean()).item() if len(pos_sims) > 0 and len(neg_sims) > 0 else 0.0,
                
                # Alignment: measures how well positive pairs align (Wang & Isola, 2020)
                # E[(x - y)^2] for positive pairs, lower is better (we negate for intuition)
                'alignment': -((2 - 2 * pos_sims).mean().item()) if len(pos_sims) > 0 else 0.0,
                
                # Uniformity: measures how uniformly features are distributed (Wang & Isola, 2020)
                # log E[e^(-2||x-y||^2)] for all pairs, lower is better. Use cosine similarities
                # (unscaled) in the formula: ||x-y||^2 = 2 - 2 * cos(x,y).
                'uniformity': (torch.log(
                    torch.exp(-2 * (2 - 2 * similarity_matrix[logits_mask.bool()])).mean()
                ).item() if logits_mask.bool().any() else 0.0),

                # Rank statistics: average rank of positive pairs (ranking invariant to positive scaling)
                #'pos_rank_mean': self._compute_positive_rank(cos_sim, pos_mask, logits_mask).item(),
            }
            
            # Add count information
            metrics['num_positives'] = pos_sims.numel()
            metrics['num_negatives'] = neg_sims.numel()
            
        return metrics
    
    def _compute_positive_rank(self, similarity_matrix, pos_mask, logits_mask):
        """Compute average rank of positive pairs in similarity distribution.
        Lower rank (closer to 1) means positives rank highly, which is desired.
        """
        ranks = []
        for i in range(similarity_matrix.size(0)):
            row_sims = similarity_matrix[i]
            valid_mask = logits_mask[i].bool()
            pos_in_row = pos_mask[i].bool() & valid_mask
            
            if pos_in_row.sum() == 0:
                continue
                
            # Get similarities for valid samples
            valid_sims = row_sims[valid_mask]
            # Sort in descending order (higher similarity = lower rank)
            sorted_indices = torch.argsort(valid_sims, descending=True)
            
            # Find ranks of positive samples (1-indexed)
            pos_indices = torch.where(pos_in_row[valid_mask])[0]
            for pos_idx in pos_indices:
                rank = (sorted_indices == pos_idx).nonzero(as_tuple=True)[0].item() + 1
                ranks.append(rank)
        
        return torch.tensor(ranks).float().mean() if ranks else torch.tensor(0.0)


class NT_Xent(nn.Module):
    def __init__(self, temp):
        super(NT_Xent, self).__init__()
        self.temperature = temp
        self.mask = None

        
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self):
        # create mask for negative samples: main diagonal, +-batch_size off-diagonal are set to 0
        N = 2 * self.batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(self.batch_size):
            mask[i, self.batch_size + i] = 0
            mask[self.batch_size + i, i] = 0
        self.mask = mask

    def forward(self, z_i, z_j):
        """
        z_i, z_j: representations of batch in two different views. shape: batch_size x C
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        # dimension of similarity matrix
        batch_size = z_i.size(0)


        N = 2 * batch_size
        if self.mask is None or self.batch_size != batch_size:
            self.batch_size = batch_size
            self.mask_correlated_samples()

        # concat both representations to easily compute similarity matrix
        z = torch.cat((z_i, z_j), dim=0)
        # compute similarity matrix around dimension 2, which is the representation depth. the unsqueeze ensures the matmul/ outer product
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        # take positive samples
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # We have 2N samples,resulting in: 2xNx1
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # negative samples are singled out with the mask
        negative_samples = sim[self.mask].reshape(N, -1)

        # reformulate everything in terms of CrossEntropyLoss: https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html
        # labels in nominator, logits in denominator
        # positve class: 0 - that's the first component of the logits corresponding to the positive samples
        labels = torch.zeros(N).to(positive_samples.device).long()
        # the logits are NxN (N+1?) predictions for imaginary classes.
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

class NTXentLoss(nn.Module):
    def __init__(self, temp=0.5):
        super(NTXentLoss, self).__init__()
        self.temp = temp

    def forward(self, z_i, z_j):
        """
        z_i: Tensor of shape [N, D] - first view embeddings
        z_j: Tensor of shape [N, D] - second view embeddings
        Returns:
            Scalar NT-Xent loss
        """
        N = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)  # [2N, D]

        # Normalize embeddings
        z = nn.functional.normalize(z, dim=1)

        # Cosine similarity matrix
        sim_matrix = torch.matmul(z, z.T)  # [2N, 2N]
        sim_matrix = sim_matrix / self.temp

        # Mask to remove self-similarity
        mask = (~torch.eye(2 * N, 2 * N, dtype=bool, device=z.device)).float()

        # Numerator: positive pairs (i, j)
        pos_sim = torch.exp(torch.sum(z_i * z_j, dim=-1) / self.temp)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)  # [2N]

        # Denominator: sum over all except self
        denom = torch.sum(torch.exp(sim_matrix) * mask, dim=1)  # [2N]

        loss = -torch.log(pos_sim / denom)
        return loss.mean()



class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        pt = torch.exp(-bce)
        loss = self.alpha * (1-pt)**self.gamma * bce
        return loss.mean()


class JSDivLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.kl_div = torch.nn.KLDivLoss(reduction='batchmean')

    def forward(self, logits1, logits2):
        pass

class WasserSteinLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p, q):
        p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
        q = q / (q.sum(dim=-1, keepdim=True) + 1e-8)

        cdf_p = torch.cumsum(p, dim=-1)
        cdf_q = torch.cumsum(q, dim=-1)

        w1_per_sample = torch.abs(cdf_p - cdf_q)
        return w1_per_sample
