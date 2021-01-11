import torch
import torch.nn as nn
import torch.nn.functional as F


class Triplet_Loss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self):
        super(Triplet_Loss, self).__init__()
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(
                                distance_function=nn.CosineSimilarity(dim=1, eps=1e-6), margin=0.5)
        self.l2_dist = nn.PairwiseDistance(p=2)
        self.triplet_loss_l2 = nn.TripletMarginWithDistanceLoss(
                                distance_function=nn.PairwiseDistance(p=2), margin=0.5)
        self.cos_dist =  nn.CosineSimilarity(dim=1, eps=1e-6)
        self.sigmoid = nn.Sigmoid()

    def forward(self, anchor, positive, negative):
        output = self.triplet_loss(anchor, positive, negative)
        #output_l2 = self.triplet_loss_l2(anchor, positive, negative)

        '''pos_dist = self.l2_dist(anchor, positive)
        neg_dist = self.l2_dist(anchor, negative)
        norm_dist = torch.cat((pos_dist, neg_dist), dim=0)
        norm_dist = F.normalize(norm_dist, p=2, dim=0)
        output_l2 = F.relu(norm_dist[0] - norm_dist[1] + 0.5)
        '''
        #output = self.triplet_loss_l2(anchor, positive, negative)

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
        #return output_l2

    #def get_embedding(self, triplet_feature):
    #    import IPython; IPython.embed()
    #    return embeddings

