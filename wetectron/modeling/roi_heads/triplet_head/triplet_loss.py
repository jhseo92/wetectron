import torch
import torch.nn as nn
import torch.nn.functional as F
#from pytorch_metric_learning import losses, distances, reducers

class Triplet_Loss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self):
        super(Triplet_Loss, self).__init__()
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(
                                distance_function=nn.CosineSimilarity(dim=1, eps=1e-6), margin=0.5)
        self.triplet_loss_l2 = nn.TripletMarginWithDistanceLoss(
                                distance_function=nn.PairwiseDistance(p=2), margin=0.5)
        self.l2_dist = nn.PairwiseDistance(p=2)
        self.cos_dist =  nn.CosineSimilarity(dim=1, eps=1e-6)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
    def forward(self, anchor, positive, negative):
        output = self.triplet_loss(anchor, positive, negative)
        #output = self.triplet_loss_l2(anchor, positive, negative)
        return output
