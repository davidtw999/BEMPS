import random
from sklearn.manifold import TSNE
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import torch
from sklearn.cluster import KMeans
from torch.nn.functional import normalize



## Random generator for X prime
def random_generator_for_x_prime(x_dim, size):
    sample_indices = random.sample(range(0, x_dim), round(x_dim * size))
    return sorted(sample_indices)



## BALD
def bald(probs_B_K_C, k):
    ens_Prob_y_L_x = torch.mean(probs_B_K_C, dim=1, keepdim=True)
    ens_Sum_KL_k_c = torch.sum(probs_B_K_C * torch.log(probs_B_K_C / ens_Prob_y_L_x), dim=1)
    ens_Sum_KL_k = torch.sum(ens_Sum_KL_k_c, dim=1)
    bald_score = ens_Sum_KL_k / k
    return bald_score


## sub function for MOCU and WMOCU
def estimate_scores(pr_ThetaLXY_X_Y_E, c , cur_flag, X):

    xp_indices = random_generator_for_x_prime(pr_ThetaLXY_X_Y_E.shape[0], X)
    pr_YpThetaXp_Xp_Yp_E = pr_ThetaLXY_X_Y_E[xp_indices, :, :]
    pr_YpThetaXp_Xp_E_Yp = pr_YpThetaXp_Xp_Yp_E.transpose(1,2)

    if cur_flag:
        pr_ThetaD = 1/pr_YpThetaXp_Xp_E_Yp.shape[1]
        max_pr_YpTXp_Xp_E = torch.max(pr_YpThetaXp_Xp_E_Yp, dim=-1).values
        bayesian_error = torch.sum(pr_ThetaD * (1 - max_pr_YpTXp_Xp_E), dim=-1)

        pr_YpDXp_Xp_Yp = pr_ThetaD * torch.sum(pr_YpThetaXp_Xp_E_Yp, dim=1)
        max_pr_YpTXp_Xp = torch.max(pr_YpDXp_Xp_Yp, dim=-1).values
        obc_error = 1 - max_pr_YpTXp_Xp

        K = obc_error - bayesian_error
        wmocu = torch.sum(1/pr_YpThetaXp_Xp_E_Yp.shape[0] * ((1 - c * K) * K), dim = 0)


    else:
        pr_ThetaDXY_X_Y_E = pr_ThetaLXY_X_Y_E
        max_pr_YpTXp_Xp_E = torch.max(pr_YpThetaXp_Xp_E_Yp, dim=-1).values
        min_pr_YpTXp_Xp_E = 1 - max_pr_YpTXp_Xp_E
        min_pr_YpTXp_E_Xp = min_pr_YpTXp_Xp_E.transpose(0,1)
        bayesian_error_X_Y_Xp = torch.matmul(pr_ThetaDXY_X_Y_E, min_pr_YpTXp_E_Xp)

        pr_ThetaDXY_X_1_Y_E = pr_ThetaDXY_X_Y_E.unsqueeze(dim=1)
        pr_YpDXp_X_Xp_Y_Yp = torch.matmul(pr_ThetaDXY_X_1_Y_E, pr_YpThetaXp_Xp_E_Yp)

        obc_error_X_Xp_Y = 1 - torch.max(pr_YpDXp_X_Xp_Y_Yp, dim=-1).values
        obc_error_X_Y_Xp = obc_error_X_Xp_Y.transpose(1,2)

        K = obc_error_X_Y_Xp - bayesian_error_X_Y_Xp
        pr_1CK = 1 - c * K
        pr_Xp1CKK_X_Y_Xp = 1/pr_YpThetaXp_Xp_E_Yp.shape[0] * (pr_1CK * K)

        wmocu = torch.sum(pr_Xp1CKK_X_Y_Xp, dim=-1)


    return wmocu


## MOCU and WMOCU
def mocu_wmocu(probs_B_K_C, c, X):

    pr_YThetaX_X_E_Y = probs_B_K_C

    pr_ThetaD = 1 / pr_YThetaX_X_E_Y.shape[1]
    pr_YThetaX_X_E_Y = pr_ThetaD * pr_YThetaX_X_E_Y
    pr_YThetaX_X_Y_E = torch.transpose(pr_YThetaX_X_E_Y, 1, 2)  ## transpose by dimension E and Y

    pr_YDX_X_Y = torch.sum(pr_YThetaX_X_Y_E, dim=-1)
    sum_pr_YThetaX_X_Y_1 = torch.sum(pr_YThetaX_X_Y_E, dim=-1).unsqueeze(dim=-1)

    pr_ThetaLXY_X_Y_E = pr_YThetaX_X_Y_E / sum_pr_YThetaX_X_Y_1

    wmocu_cur = estimate_scores(pr_ThetaLXY_X_Y_E, c, 1, X)
    wmocu_temp_X_Y = estimate_scores(pr_ThetaLXY_X_Y_E, c, 0, X)
    wmocu_next = torch.sum(torch.mul(pr_YDX_X_Y, wmocu_temp_X_Y), dim=-1)


    return wmocu_cur - wmocu_next



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



## sub function for MOCU and WMOCU batch mode
def estimate_score_batch(pr_ThetaLXY_X_Y_E, c , cur_flag, X):

    xp_indices = random_generator_for_x_prime(pr_ThetaLXY_X_Y_E.shape[0], X)
    pr_YpThetaXp_Xp_Yp_E = pr_ThetaLXY_X_Y_E[xp_indices, :, :]
    pr_YpThetaXp_Xp_E_Yp = pr_YpThetaXp_Xp_Yp_E.transpose(1,2)

    if cur_flag:
        pr_ThetaD = 1/pr_YpThetaXp_Xp_E_Yp.shape[1]
        max_pr_YpTXp_Xp_E = torch.max(pr_YpThetaXp_Xp_E_Yp, dim=-1).values
        bayesian_error = torch.sum(pr_ThetaD * (1 - max_pr_YpTXp_Xp_E), dim=-1)

        pr_YpDXp_Xp_Yp = pr_ThetaD * torch.sum(pr_YpThetaXp_Xp_E_Yp, dim=1)
        max_pr_YpTXp_Xp = torch.max(pr_YpDXp_Xp_Yp, dim=-1).values
        obc_error = 1 - max_pr_YpTXp_Xp

        K = obc_error - bayesian_error
        wmocu = torch.sum(1/pr_YpThetaXp_Xp_E_Yp.shape[0] * ((1 - c * K) * K), dim = 0)


    else:
        pr_ThetaDXY_X_Y_E = pr_ThetaLXY_X_Y_E
        max_pr_YpTXp_Xp_E = torch.max(pr_YpThetaXp_Xp_E_Yp, dim=-1).values
        min_pr_YpTXp_Xp_E = 1 - max_pr_YpTXp_Xp_E
        min_pr_YpTXp_E_Xp = min_pr_YpTXp_Xp_E.transpose(0,1)
        bayesian_error_X_Y_Xp = torch.matmul(pr_ThetaDXY_X_Y_E, min_pr_YpTXp_E_Xp)

        pr_ThetaDXY_X_1_Y_E = pr_ThetaDXY_X_Y_E.unsqueeze(dim=1)
        pr_YpDXp_X_Xp_Y_Yp = torch.matmul(pr_ThetaDXY_X_1_Y_E, pr_YpThetaXp_Xp_E_Yp)

        obc_error_X_Xp_Y = 1 - torch.max(pr_YpDXp_X_Xp_Y_Yp, dim=-1).values
        obc_error_X_Y_Xp = obc_error_X_Xp_Y.transpose(1,2)

        K = obc_error_X_Y_Xp - bayesian_error_X_Y_Xp
        pr_1CK = 1 - c * K
        pr_Xp1CKK_X_Y_Xp = 1/pr_YpThetaXp_Xp_E_Yp.shape[0] * (pr_1CK * K)

        pr_Xp1CKK_X_Xp_Y = torch.transpose(pr_Xp1CKK_X_Y_Xp, 1, 2)
        wmocu = torch.transpose(pr_Xp1CKK_X_Xp_Y, 0, 1)


    return wmocu


## MOCU and WMOCU batch mode
def mocu_wmocu_batch(probs_B_K_C, c, X, batch_size, T):

    pr_YThetaX_X_E_Y = probs_B_K_C

    pr_ThetaD = 1 / pr_YThetaX_X_E_Y.shape[1]
    pr_YThetaX_X_E_Y = pr_ThetaD * pr_YThetaX_X_E_Y
    pr_YThetaX_X_Y_E = torch.transpose(pr_YThetaX_X_E_Y, 1, 2)  ## transpose by dimension E and Y

    pr_YDX_X_Y = torch.sum(pr_YThetaX_X_Y_E, dim=-1)
    sum_pr_YThetaX_X_Y_1 = torch.sum(pr_YThetaX_X_Y_E, dim=-1).unsqueeze(dim=-1)

    pr_ThetaLXY_X_Y_E = pr_YThetaX_X_Y_E / sum_pr_YThetaX_X_Y_1

    wmocu_cur = estimate_score_batch(pr_ThetaLXY_X_Y_E, c, 1, X)
    wmocu_temp_Xp_X_Y = estimate_score_batch(pr_ThetaLXY_X_Y_E, c, 0, X)

    ## change here
    wmocu_next_Xp_X_Y = pr_YDX_X_Y.unsqueeze(0) * wmocu_temp_Xp_X_Y
    wmocu_next_Xp_X = torch.sum(wmocu_next_Xp_X_Y, dim=-1)
    wmocu_next_X_Xp = torch.transpose(wmocu_next_Xp_X, 0, 1)

    rr_X_Xp = wmocu_cur - wmocu_next_X_Xp

    rr = clustering(rr_X_Xp, probs_B_K_C, T, batch_size)

    return rr


## pretrained LM clustering
def badge_clustering(probs_B_K_C, batch_size):
    vectors = normalize(probs_B_K_C)
    rr = kmeans_pp(vectors, batch_size, [])

    return rr



## pretrained LM clustering
def lm_clustering(probs_B_K_C, batch_size):
    pr_YThetaX_X_E_Y = probs_B_K_C

    rr_X_Y = torch.sum(pr_YThetaX_X_E_Y, dim=1) / pr_YThetaX_X_E_Y.shape[1]
    rr_X_Y = normalize(rr_X_Y)
    rr = kmeans(rr_X_Y, batch_size)

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
    # rr = kmeans_pp(rr_X_Xp, batch_size, [])
    rr = [rr_topk_X_indices[x] for x in rr]

    return rr


## sub fuction for kmeans ++
def closest_center_dist(rr, centers):
    dist = torch.cdist(rr, rr[centers])
    cd = dist.min(axis=1).values
    return cd


## kmeans ++
def kmeans_pp(rr, k, centers):

    if len(centers) == 0:
        c1 = np.random.choice(rr.size(0))
        centers.append(c1)
        k -= 1

    for i in tqdm(range(k)):
        dist = closest_center_dist(rr, centers) ** 2
        prob = (dist / dist.sum()).cpu().detach().numpy()
        ci = np.random.choice(rr.size(0), p=prob)
        centers.append(ci)
    return centers


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

    ## testing the function


def main():
    test_function()


if __name__ == "__main__":
    main()