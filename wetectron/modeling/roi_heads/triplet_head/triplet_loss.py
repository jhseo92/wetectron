import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses, distances, reducers

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
        self.pair_dist = nn.PairwiseDistance(p=2)
        self.cos_dist =  nn.CosineSimilarity(dim=1, eps=1e-6)
        self.sigmoid = nn.Sigmoid()
        self.margin = 0.3
    def forward(self, anchor, positive, negative):
        #cos = nn.CosineSimilarity(dim=1,eps=1e-6)

        output = self.triplet_loss(anchor, positive, negative)

        #squarred_distance_1 = (anchor - positive).pow(2).sum(1)
        #squarred_distance_2 = (anchor - negative).pow(2).sum(1)

        #triplet_loss = F.relu( self.margin + squarred_distance_1 - squarred_distance_2 ).mean()
        #output_l2 = self.triplet_loss_l2(anchor, positive, negative)
        #import IPython; IPython.embed()

        #pos_dist = (anchor - positive).pow(2).sum(1)
        #neg_dist = (anchor - negative).pow(2).sum(1)
        #prob_dist = torch.cat((pos_dist, neg_dist),0)
        #losses = F.relu(pos_dist - neg_dist + 0.5)

        #p = self.pair_dist(anchor, positive)
        #n = self.pair_dist(anchor, negative)
        #prob = torch.cat((p,n),0)
        #import IPython; IPython.embed()
        return output

