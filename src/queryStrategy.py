import random
from sklearn.manifold import TSNE
import numpy as np
from scipy.spatial.distance import cdist
import torch
from sklearn.cluster import KMeans
from torch.nn.functional import normalize



## Random generator for X prime
def random_generator_for_x_prime(x_dim, size):
    sample_indices = random.sample(range(0, x_dim), round(x_dim * size))
    return sorted(sample_indices)




## CoreLog
def bemps_corelog(probs_B_K_C, X):

    ## Pr(y|theta,x)
    pr_YThetaX_X_E_Y = probs_B_K_C
    pr_ThetaL = 1 / pr_YThetaX_X_E_Y.shape[1]

    ## Generate random number of x'
    xp_indices = random_generator_for_x_prime(pr_YThetaX_X_E_Y.shape[0], X)
    pr_YhThetaXp_Xp_E_Yh = pr_YThetaX_X_E_Y[xp_indices, :, :]

    ## Transpose dimension of Pr(y|theta,x), and calculate pr(theta|L,(x,y))
    pr_YThetaX_X_E_Y = pr_ThetaL * pr_YThetaX_X_E_Y
    pr_YThetaX_X_Y_E = torch.transpose(pr_YThetaX_X_E_Y, 1, 2)  ## transpose by dimension E and Y

    sum_pr_YThetaX_X_Y_1 = torch.sum(pr_YThetaX_X_Y_E, dim=-1).unsqueeze(dim=-1)
    pr_ThetaLXY_X_Y_E = pr_YThetaX_X_Y_E / sum_pr_YThetaX_X_Y_1

    ## Calculate pr(y_hat)
    pr_ThetaLXY_X_1_Y_E = pr_ThetaLXY_X_Y_E.unsqueeze(dim=1)
    pr_Yhat_X_Xp_Y_Yh = torch.matmul(pr_ThetaLXY_X_1_Y_E, pr_YhThetaXp_Xp_E_Yh)


    ## Calculate core MSE by using unsqueeze into same dimension for pr(y_hat) and pr(y_hat|theta,x)
    pr_YhThetaXp_1_1_Xp_E_Yh = pr_YhThetaXp_Xp_E_Yh.unsqueeze(dim = 0).unsqueeze(dim = 0)
    pr_YhThetaXp_X_Y_Xp_E_Yh = pr_YhThetaXp_1_1_Xp_E_Yh.repeat(pr_Yhat_X_Xp_Y_Yh.shape[0], pr_Yhat_X_Xp_Y_Yh.shape[2], 1, 1, 1)

    pr_Yhat_1_X_Xp_Y_Yh = pr_Yhat_X_Xp_Y_Yh.unsqueeze(dim = 0)
    pr_Yhat_E_X_Xp_Y_Yh = pr_Yhat_1_X_Xp_Y_Yh.repeat(pr_YhThetaXp_Xp_E_Yh.shape[1],1,1,1,1)
    pr_Yhat_X_Y_Xp_E_Yh = pr_Yhat_E_X_Xp_Y_Yh.transpose(0,3).transpose(0,1)

    core_mse = torch.mul(pr_YhThetaXp_X_Y_Xp_E_Yh, torch.div(pr_YhThetaXp_X_Y_Xp_E_Yh, pr_Yhat_X_Y_Xp_E_Yh))
    core_mse_X_Y = torch.sum(torch.sum(core_mse.sum(dim=-1), dim=-1),dim=-1)

    ## Calculate RR
    pr_YLX_X_Y = torch.sum(pr_YThetaX_X_Y_E, dim=-1)
    rr = torch.sum(torch.mul(pr_YLX_X_Y, core_mse_X_Y), dim=-1) / pr_YhThetaXp_Xp_E_Yh.shape[0]

    return rr



## CoreMSE
def bemps_coremse(probs_B_K_C, X):

    ## Pr(y|theta,x)
    pr_YThetaX_X_E_Y = probs_B_K_C
    pr_ThetaL = 1 / pr_YThetaX_X_E_Y.shape[1]

    ## Generate random number of x'
    xp_indices = random_generator_for_x_prime(pr_YThetaX_X_E_Y.shape[0], X)
    pr_YhThetaXp_Xp_E_Yh = pr_YThetaX_X_E_Y[xp_indices, :, :]

    ## Transpose dimension of Pr(y|theta,x), and calculate pr(theta|L,(x,y))
    pr_YThetaX_X_E_Y = pr_ThetaL * pr_YThetaX_X_E_Y
    pr_YThetaX_X_Y_E = torch.transpose(pr_YThetaX_X_E_Y, 1, 2)  ## transpose by dimension E and Y

    sum_pr_YThetaX_X_Y_1 = torch.sum(pr_YThetaX_X_Y_E, dim=-1).unsqueeze(dim=-1)
    pr_ThetaLXY_X_Y_E = pr_YThetaX_X_Y_E / sum_pr_YThetaX_X_Y_1

    ## Calculate pr(y_hat)
    pr_ThetaLXY_X_1_Y_E = pr_ThetaLXY_X_Y_E.unsqueeze(dim=1)
    pr_Yhat_X_Xp_Y_Yh = torch.matmul(pr_ThetaLXY_X_1_Y_E, pr_YhThetaXp_Xp_E_Yh)

    ## Calculate core MSE by using unsqueeze into same dimension for pr(y_hat) and pr(y_hat|theta,x)
    pr_YhThetaXp_1_1_Xp_E_Yh = pr_YhThetaXp_Xp_E_Yh.unsqueeze(dim = 0).unsqueeze(dim = 0)
    pr_YhThetaXp_X_Y_Xp_E_Yh = pr_YhThetaXp_1_1_Xp_E_Yh.repeat(pr_Yhat_X_Xp_Y_Yh.shape[0], pr_Yhat_X_Xp_Y_Yh.shape[2], 1, 1, 1)

    pr_Yhat_1_X_Xp_Y_Yh = pr_Yhat_X_Xp_Y_Yh.unsqueeze(dim = 0)
    pr_Yhat_E_X_Xp_Y_Yh = pr_Yhat_1_X_Xp_Y_Yh.repeat(pr_YhThetaXp_Xp_E_Yh.shape[1],1,1,1,1)
    pr_Yhat_X_Y_Xp_E_Yh = pr_Yhat_E_X_Xp_Y_Yh.transpose(0,3).transpose(0,1)

    core_mse = (pr_YhThetaXp_X_Y_Xp_E_Yh - pr_Yhat_X_Y_Xp_E_Yh).pow(2)
    core_mse_X_Y = torch.sum(torch.sum(core_mse.sum(dim=-1), dim=-1),dim=-1)

    ## Calculate RR
    pr_YLX_X_Y = torch.sum(pr_YThetaX_X_Y_E, dim=-1)
    rr = torch.sum(torch.mul(pr_YLX_X_Y, core_mse_X_Y), dim=-1)/pr_YhThetaXp_Xp_E_Yh.shape[0]

    return rr





## CoreMSE batch mode
def bemps_coremse_batch(probs_B_K_C, batch_size, X, T):

    ## Pr(y|theta,x)
    pr_YThetaX_X_E_Y = probs_B_K_C
    pr_ThetaL = 1 / pr_YThetaX_X_E_Y.shape[1]

    ## Generate random number of x'
    xp_indices = random_generator_for_x_prime(pr_YThetaX_X_E_Y.shape[0], X)
    pr_YhThetaXp_Xp_E_Yh = pr_YThetaX_X_E_Y[xp_indices, :, :]

    ## Transpose dimension of Pr(y|theta,x), and calculate pr(theta|L,(x,y))
    pr_YThetaX_X_E_Y = pr_ThetaL * pr_YThetaX_X_E_Y
    pr_YThetaX_X_Y_E = torch.transpose(pr_YThetaX_X_E_Y, 1, 2)  ## transpose by dimension E and Y

    sum_pr_YThetaX_X_Y_1 = torch.sum(pr_YThetaX_X_Y_E, dim=-1).unsqueeze(dim=-1)
    pr_ThetaLXY_X_Y_E = pr_YThetaX_X_Y_E / sum_pr_YThetaX_X_Y_1

    ## Calculate pr(y_hat)
    pr_ThetaLXY_X_1_Y_E = pr_ThetaLXY_X_Y_E.unsqueeze(dim=1)
    pr_Yhat_X_Xp_Y_Yh = torch.matmul(pr_ThetaLXY_X_1_Y_E, pr_YhThetaXp_Xp_E_Yh)

    ## Calculate core MSE by using unsqueeze into same dimension for pr(y_hat) and pr(y_hat|theta,x)
    pr_YhThetaXp_1_1_Xp_E_Yh = pr_YhThetaXp_Xp_E_Yh.unsqueeze(dim = 0).unsqueeze(dim = 0)
    pr_YhThetaXp_X_Y_Xp_E_Yh = pr_YhThetaXp_1_1_Xp_E_Yh.repeat(pr_Yhat_X_Xp_Y_Yh.shape[0], pr_Yhat_X_Xp_Y_Yh.shape[2], 1, 1, 1)

    pr_Yhat_1_X_Xp_Y_Yh = pr_Yhat_X_Xp_Y_Yh.unsqueeze(dim = 0)
    pr_Yhat_E_X_Xp_Y_Yh = pr_Yhat_1_X_Xp_Y_Yh.repeat(pr_YhThetaXp_Xp_E_Yh.shape[1],1,1,1,1)
    pr_Yhat_X_Y_Xp_E_Yh = pr_Yhat_E_X_Xp_Y_Yh.transpose(0,3).transpose(0,1)

    core_mse = (pr_YhThetaXp_X_Y_Xp_E_Yh - pr_Yhat_X_Y_Xp_E_Yh).pow(2)
    core_mse_X_Y_Xp = torch.sum(core_mse.sum(dim=-1), dim=-1)
    core_mse_X_Xp_Y = torch.transpose(core_mse_X_Y_Xp, 1, 2)
    core_mse_Xp_X_Y = torch.transpose(core_mse_X_Xp_Y, 0, 1)

    ## Calculate RR
    pr_YLX_X_Y = torch.sum(pr_YThetaX_X_Y_E, dim=-1)

    rr_Xp_X_Y = pr_YLX_X_Y.unsqueeze(0) * core_mse_Xp_X_Y

    rr_Xp_X = torch.sum(rr_Xp_X_Y, dim=-1)
    rr_X_Xp = torch.transpose(rr_Xp_X, 0, 1)

    rr = clustering(rr_X_Xp, probs_B_K_C, T, batch_size)

    return rr


## CoreMSE top rank mode
def bemps_coremse_batch_topk(probs_B_K_C, batch_size, X):
    ## Pr(y|theta,x)
    pr_YThetaX_X_E_Y = probs_B_K_C
    pr_ThetaL = 1 / pr_YThetaX_X_E_Y.shape[1]

    ## Generate random number of x'
    xp_indices = random_generator_for_x_prime(pr_YThetaX_X_E_Y.shape[0], X)
    pr_YhThetaXp_Xp_E_Yh = pr_YThetaX_X_E_Y[xp_indices, :, :]

    ## Transpose dimension of Pr(y|theta,x), and calculate pr(theta|L,(x,y))
    pr_YThetaX_X_E_Y = pr_ThetaL * pr_YThetaX_X_E_Y
    pr_YThetaX_X_Y_E = torch.transpose(pr_YThetaX_X_E_Y, 1, 2)  ## transpose by dimension E and Y

    sum_pr_YThetaX_X_Y_1 = torch.sum(pr_YThetaX_X_Y_E, dim=-1).unsqueeze(dim=-1)
    pr_ThetaLXY_X_Y_E = pr_YThetaX_X_Y_E / sum_pr_YThetaX_X_Y_1

    ## Calculate pr(y_hat)
    pr_ThetaLXY_X_1_Y_E = pr_ThetaLXY_X_Y_E.unsqueeze(dim=1)
    pr_Yhat_X_Xp_Y_Yh = torch.matmul(pr_ThetaLXY_X_1_Y_E, pr_YhThetaXp_Xp_E_Yh)

    ## Calculate core MSE by using unsqueeze into same dimension for pr(y_hat) and pr(y_hat|theta,x)
    pr_YhThetaXp_1_1_Xp_E_Yh = pr_YhThetaXp_Xp_E_Yh.unsqueeze(dim=0).unsqueeze(dim=0)
    pr_YhThetaXp_X_Y_Xp_E_Yh = pr_YhThetaXp_1_1_Xp_E_Yh.repeat(pr_Yhat_X_Xp_Y_Yh.shape[0], pr_Yhat_X_Xp_Y_Yh.shape[2],
                                                               1, 1, 1)

    pr_Yhat_1_X_Xp_Y_Yh = pr_Yhat_X_Xp_Y_Yh.unsqueeze(dim=0)
    pr_Yhat_E_X_Xp_Y_Yh = pr_Yhat_1_X_Xp_Y_Yh.repeat(pr_YhThetaXp_Xp_E_Yh.shape[1], 1, 1, 1, 1)
    pr_Yhat_X_Y_Xp_E_Yh = pr_Yhat_E_X_Xp_Y_Yh.transpose(0, 3).transpose(0, 1)

    core_mse = (pr_YhThetaXp_X_Y_Xp_E_Yh - pr_Yhat_X_Y_Xp_E_Yh).pow(2)
    core_mse_X_Y = torch.sum(torch.sum(core_mse.sum(dim=-1), dim=-1), dim=-1)

    ## Calculate RR
    pr_YLX_X_Y = torch.sum(pr_YThetaX_X_Y_E, dim=-1)
    rr = torch.sum(torch.mul(pr_YLX_X_Y, core_mse_X_Y), dim=-1) / pr_YhThetaXp_Xp_E_Yh.shape[0]


    return  rr.topk(batch_size).indices.numpy()



## CoreLog top rank mode
def bemps_corelog_batch_topk(probs_B_K_C, batch_size, X):

    ## Pr(y|theta,x)
    pr_YThetaX_X_E_Y = probs_B_K_C
    pr_ThetaL = 1 / pr_YThetaX_X_E_Y.shape[1]

    ## Generate random number of x'
    xp_indices = random_generator_for_x_prime(pr_YThetaX_X_E_Y.shape[0], X)
    pr_YhThetaXp_Xp_E_Yh = pr_YThetaX_X_E_Y[xp_indices, :, :]

    ## Transpose dimension of Pr(y|theta,x), and calculate pr(theta|L,(x,y))
    pr_YThetaX_X_E_Y = pr_ThetaL * pr_YThetaX_X_E_Y
    pr_YThetaX_X_Y_E = torch.transpose(pr_YThetaX_X_E_Y, 1, 2)  ## transpose by dimension E and Y

    sum_pr_YThetaX_X_Y_1 = torch.sum(pr_YThetaX_X_Y_E, dim=-1).unsqueeze(dim=-1)
    pr_ThetaLXY_X_Y_E = pr_YThetaX_X_Y_E / sum_pr_YThetaX_X_Y_1

    ## Calculate pr(y_hat)
    pr_ThetaLXY_X_1_Y_E = pr_ThetaLXY_X_Y_E.unsqueeze(dim=1)
    pr_Yhat_X_Xp_Y_Yh = torch.matmul(pr_ThetaLXY_X_1_Y_E, pr_YhThetaXp_Xp_E_Yh)


    ## Calculate core MSE by using unsqueeze into same dimension for pr(y_hat) and pr(y_hat|theta,x)
    pr_YhThetaXp_1_1_Xp_E_Yh = pr_YhThetaXp_Xp_E_Yh.unsqueeze(dim = 0).unsqueeze(dim = 0)
    pr_YhThetaXp_X_Y_Xp_E_Yh = pr_YhThetaXp_1_1_Xp_E_Yh.repeat(pr_Yhat_X_Xp_Y_Yh.shape[0], pr_Yhat_X_Xp_Y_Yh.shape[2], 1, 1, 1)

    pr_Yhat_1_X_Xp_Y_Yh = pr_Yhat_X_Xp_Y_Yh.unsqueeze(dim = 0)
    pr_Yhat_E_X_Xp_Y_Yh = pr_Yhat_1_X_Xp_Y_Yh.repeat(pr_YhThetaXp_Xp_E_Yh.shape[1],1,1,1,1)
    pr_Yhat_X_Y_Xp_E_Yh = pr_Yhat_E_X_Xp_Y_Yh.transpose(0,3).transpose(0,1)

    core_mse = torch.mul(pr_YhThetaXp_X_Y_Xp_E_Yh, torch.div(pr_YhThetaXp_X_Y_Xp_E_Yh, pr_Yhat_X_Y_Xp_E_Yh))
    core_mse_X_Y = torch.sum(torch.sum(core_mse.sum(dim=-1), dim=-1),dim=-1)

    ## Calculate RR
    pr_YLX_X_Y = torch.sum(pr_YThetaX_X_Y_E, dim=-1)
    rr = torch.sum(torch.mul(pr_YLX_X_Y, core_mse_X_Y), dim=-1) / pr_YhThetaXp_Xp_E_Yh.shape[0]

    return  rr.topk(batch_size).indices.numpy()




## CoreLog batch mode
def bemps_corelog_batch(probs_B_K_C, batch_size, X, T):

    ## Pr(y|theta,x)
    pr_YThetaX_X_E_Y = probs_B_K_C
    pr_ThetaL = 1 / pr_YThetaX_X_E_Y.shape[1]

    ## Generate random number of x'
    xp_indices = random_generator_for_x_prime(pr_YThetaX_X_E_Y.shape[0], X)
    pr_YhThetaXp_Xp_E_Yh = pr_YThetaX_X_E_Y[xp_indices, :, :]

    ## Transpose dimension of Pr(y|theta,x), and calculate pr(theta|L,(x,y))
    pr_YThetaX_X_E_Y = pr_ThetaL * pr_YThetaX_X_E_Y
    pr_YThetaX_X_Y_E = torch.transpose(pr_YThetaX_X_E_Y, 1, 2)  ## transpose by dimension E and Y

    sum_pr_YThetaX_X_Y_1 = torch.sum(pr_YThetaX_X_Y_E, dim=-1).unsqueeze(dim=-1)
    pr_ThetaLXY_X_Y_E = pr_YThetaX_X_Y_E / sum_pr_YThetaX_X_Y_1

    ## Calculate pr(y_hat)
    pr_ThetaLXY_X_1_Y_E = pr_ThetaLXY_X_Y_E.unsqueeze(dim=1)
    pr_Yhat_X_Xp_Y_Yh = torch.matmul(pr_ThetaLXY_X_1_Y_E, pr_YhThetaXp_Xp_E_Yh)


    ## Calculate core MSE by using unsqueeze into same dimension for pr(y_hat) and pr(y_hat|theta,x)
    pr_YhThetaXp_1_1_Xp_E_Yh = pr_YhThetaXp_Xp_E_Yh.unsqueeze(dim = 0).unsqueeze(dim = 0)
    pr_YhThetaXp_X_Y_Xp_E_Yh = pr_YhThetaXp_1_1_Xp_E_Yh.repeat(pr_Yhat_X_Xp_Y_Yh.shape[0], pr_Yhat_X_Xp_Y_Yh.shape[2], 1, 1, 1)

    pr_Yhat_1_X_Xp_Y_Yh = pr_Yhat_X_Xp_Y_Yh.unsqueeze(dim = 0)
    pr_Yhat_E_X_Xp_Y_Yh = pr_Yhat_1_X_Xp_Y_Yh.repeat(pr_YhThetaXp_Xp_E_Yh.shape[1],1,1,1,1)
    pr_Yhat_X_Y_Xp_E_Yh = pr_Yhat_E_X_Xp_Y_Yh.transpose(0,3).transpose(0,1)

    core_mse = torch.mul(pr_YhThetaXp_X_Y_Xp_E_Yh, torch.div(pr_YhThetaXp_X_Y_Xp_E_Yh, pr_Yhat_X_Y_Xp_E_Yh))
    core_mse_X_Y_Xp = torch.sum(core_mse.sum(dim=-1), dim=-1)
    core_mse_X_Xp_Y = torch.transpose(core_mse_X_Y_Xp, 1, 2)
    core_mse_Xp_X_Y = torch.transpose(core_mse_X_Xp_Y, 0, 1)

    ## Calculate RR
    pr_YLX_X_Y = torch.sum(pr_YThetaX_X_Y_E, dim=-1)
    rr_Xp_X_Y = pr_YLX_X_Y.unsqueeze(0) * core_mse_Xp_X_Y
    rr_Xp_X = torch.sum(rr_Xp_X_Y, dim=-1)
    rr_X_Xp = torch.transpose(rr_Xp_X, 0, 1)

    rr = clustering(rr_X_Xp, probs_B_K_C, T, batch_size)

    return rr



## cluster methods
def clustering(rr_X_Xp, probs_B_K_C, T, batch_size):
    rr_X = torch.sum(rr_X_Xp, dim=-1)
    rr_topk_X = torch.topk(rr_X, round(probs_B_K_C.shape[0] * T))
    rr_topk_X_indices = rr_topk_X.indices.cpu().detach().numpy()
    rr_X_Xp = rr_X_Xp[rr_topk_X_indices]

    rr_X_Xp = normalize(rr_X_Xp)
    # rr_X_Xp = convert_embedding_by_tsne(rr_X_Xp)

    rr = kmeans(rr_X_Xp, batch_size)
    rr = [rr_topk_X_indices[x] for x in rr]

    return rr


## sub fuction for kmeans ++
def closest_center_dist(rr, centers):
    dist = torch.cdist(rr, rr[centers])
    cd = dist.min(axis=1).values
    return cd



## kmeans
def kmeans(rr, k):

    kmeans = KMeans(n_clusters=k, n_jobs=-1).fit(rr)
    centers = kmeans.cluster_centers_
    # find the nearest point to centers
    centroids = cdist(centers, rr).argmin(axis=1)
    centroids_set = np.unique(centroids)
    m = k - len(centroids_set)
    if m > 0:
        pool = np.delete(np.arange(len(X)), centroids_set)
        p = np.random.choice(len(pool), m)
        centroids = np.concatenate((centroids_set, pool[p]), axis = None)
    return centroids



## create tsne feature space
def convert_embedding_by_tsne(X):
    tsne = TSNE(n_components=3, random_state=100)
    X = tsne.fit_transform(X)
    return X


## random a single index
def random_queries(len_samples):
    rand_index = random.sample(range(len_samples), 1)
    return rand_index[0]

## random a set of indices
def random_queries_batch(len_samples, batch_size):
    rand_index = random.sample(range(len_samples), batch_size)
    return rand_index

## mean of the probability
def prob_mean(probs_B_K_C, dim: int, keepdim: bool = False):
    return torch.mean(probs_B_K_C, dim=dim, keepdim=keepdim)

## entropy
def entropy(probs_B_K_C, dim: int, keepdim: bool = False):
    return -torch.sum((torch.log(probs_B_K_C) * probs_B_K_C).double(), dim=dim, keepdim=keepdim)

## max entropy
def max_entropy_acquisition_function(probs_B_K_C):
    return entropy(prob_mean(probs_B_K_C, dim=1, keepdim=False), dim=-1)



## test function
def test_function():
    ## generate data model classes
    torch.manual_seed(1)
    probs_matrix = torch.rand((100, 5, 2))
    probs_matrix = torch.softmax(probs_matrix, dim=2)
    probs_matrix = torch.FloatTensor(probs_matrix)



def main():
    test_function()


if __name__ == "__main__":
    main()