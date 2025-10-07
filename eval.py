import time
import pandas as pd
import numpy as np
import torch

def l1_distance_with_cont_feat(series1, series2, cont_feat):
    """
    Compute L1 distance using cont_feat to identify continuous vs categorical features.
    
    Parameters:
    - series1, series2: pandas Series to compare
    - cont_feat: list/array of column names that are continuous features
    """
    distance = 0
    
    for col in series1.index:
        val1, val2 = series1[col], series2[col]
        
        if col in cont_feat:
            # Continuous feature: use absolute difference
            distance += abs(val1 - val2)
        else:
            # Categorical feature: 1 if different, 0 if same
            distance += 0 if val1 == val2 else 1
    
    return distance


def compute_dist(x1, x2, dice_inst):
    """Compute weighted distance between two vectors."""
    return torch.sum(torch.mul((torch.abs(x1 - x2)), dice_inst.feature_weights_list), dim=0)



def dpp_style(cfs, dice_inst):
    num_cfs = len(cfs)
    """Computes the DPP of a matrix."""
    det_entries = torch.ones((num_cfs, num_cfs))
    for i in range(num_cfs):
        for j in range(num_cfs):
            det_entries[(i, j)] = 1.0 / (
                    1.0 + compute_dist(torch.tensor(cfs.iloc[i].values), torch.tensor(cfs.iloc[j].values),
                                       dice_inst))
            if i == j:
                det_entries[(i, j)] += 0.0001
    diversity_loss = torch.det(det_entries)
    return diversity_loss


def dpp_style_numpy(cfs, dice_inst):
    num_cfs = len(cfs)
    """Computes the DPP of a matrix."""
    det_entries = torch.ones((num_cfs, num_cfs))
    for i in range(num_cfs):
        for j in range(num_cfs):
            det_entries[(i, j)] = 1.0 / (
                    1.0 + compute_dist(torch.tensor(cfs[i]), torch.tensor(cfs[j]),
                                       dice_inst))
            if i == j:
                det_entries[(i, j)] += 0.0001
    diversity_loss = torch.det(det_entries)
    return diversity_loss




def pairwise_diversity_sum(cfs, dice_inst):
    """Computes the average pairwise distance diversity metric."""
    num_cfs = len(cfs)
    if num_cfs <= 1:
        return torch.tensor(0.0)
    
    distance_sum = torch.tensor(0.0)
    pair_count = 0
    
    for i in range(num_cfs):
        for j in range(i + 1, num_cfs):  # Only compute upper triangle to avoid duplicates
            distance_sum += compute_dist(
                torch.tensor(cfs.iloc[i].values), 
                torch.tensor(cfs.iloc[j].values),
                dice_inst
            )
            pair_count += 1
    
    diversity_loss = distance_sum / pair_count  # Average of all pairwise distances
    return diversity_loss


def minimal_distance_diversity(cfs, dice_inst):
    """Computes the minimal distance between any pair as diversity metric."""
    num_cfs = len(cfs)
    if num_cfs <= 1:
        return torch.tensor(0.0)
    
    min_distance = torch.tensor(float('inf'))
    
    for i in range(num_cfs):
        for j in range(i + 1, num_cfs):  # Only compute upper triangle to avoid duplicates
            curr_distance = compute_dist(
                torch.tensor(cfs.iloc[i].values), 
                torch.tensor(cfs.iloc[j].values),
                dice_inst
            )
            min_distance = torch.min(min_distance, curr_distance)
    
    diversity_loss = min_distance
    return diversity_loss

def n_cfs_score(comb_cfs, origin_instance,projection_config):
    transformer = projection_config.transformer
    exp_random = projection_config.exp_random
    dpp_score = dpp_style(transformer.transform(comb_cfs), exp_random).item()
    pwise_div = pairwise_diversity_sum(transformer.transform(comb_cfs), exp_random).item()
    min_dist_div = minimal_distance_diversity(transformer.transform(comb_cfs), exp_random).item()
    prox_distances = []
    l0_distances = []
    l1_distances = []
    
    for index, row in comb_cfs.iterrows():
        proximity_dist = compute_dist(
            torch.tensor(transformer.transform(origin_instance.to_frame().T).values).flatten(),
            torch.tensor(transformer.transform(row.to_frame().T).values).flatten(), exp_random)
        prox_distances.append(proximity_dist)

        l0_distance = (origin_instance != row).sum()
        l1_distance = l1_distance_with_cont_feat(origin_instance, row, cont_feat=projection_config.args.cont_feat)
        
        l0_distances.append(l0_distance)
        l1_distances.append(l1_distance)
    proximity_distance = np.mean(prox_distances)
    l0_distance = np.mean(l0_distances)
    l1_distance = np.mean(l1_distances)
    return dpp_score, proximity_distance, l0_distance, l1_distance, pwise_div, min_dist_div

def best_from_dataset(dataset, origin_instance, prev,projection_config, timeout=None):
    transformer = projection_config.transformer
    
    args = projection_config.args
    cfs_pool = dataset.copy()
    curr_best = prev.copy()
    dic = {}
    start_time = time.time()
    
    origin_transformed = transformer.transform(origin_instance.to_frame().T)
    cfs_pool_transformed = transformer.transform(cfs_pool).values

    is_empty_start = curr_best.empty
    if not is_empty_start:
        curr_best_transformed = transformer.transform(curr_best).values

    for i in range(len(cfs_pool)):
        # Check if timeout is specified and if we've exceeded it
        if timeout is not None and (time.time() - start_time) > timeout:
            print(f"Timeout reached after {time.time() - start_time:.2f} seconds")
            break
        cf = cfs_pool.iloc[i]
        cf_index = cfs_pool.index[i]

        if is_empty_start:
            comb_transformed = cfs_pool_transformed[i:i+1]
            
        else:
            comb_transformed = np.vstack([curr_best_transformed, cfs_pool_transformed[i:i+1]])

        dpp_score, proximity_distance = n_cfs_score_transformed(comb_transformed, origin_transformed, projection_config)
        dic[cf_index] = (dpp_score, proximity_distance, dpp_score - args.delta/100 * proximity_distance)
    
    # If we have any results in dic, process them
    if dic:
        best_index = max(dic, key=lambda x: dic[x][-1])
        best_cf = cfs_pool.loc[best_index]
        cfs_pool = cfs_pool.drop(best_index, axis=0)
        curr_best = pd.concat([curr_best, pd.DataFrame([best_cf])],axis=0)
    
    return curr_best

def n_cfs_score_transformed(comb_transformed, origin_transformed, projection_config):
    exp_random = projection_config.exp_random
    
    if len(comb_transformed) == 1:
        dpp_score = 1.0
        proximity_dist = compute_dist(
            torch.tensor(origin_transformed.values).flatten(),
            torch.tensor(comb_transformed[0]).flatten(),
            exp_random
        ).item()
        proximity_distance = proximity_dist / len(exp_random.minx[0])
        return dpp_score, proximity_distance
    
    # Multiple counterfactuals - data already transformed
    dpp_score = dpp_style_numpy(comb_transformed, exp_random).item()
    
    prox_distances = []
    for i in range(len(comb_transformed)):
        proximity_dist = compute_dist(
            torch.tensor(origin_transformed.values).flatten(),
            torch.tensor(comb_transformed[i]).flatten(),
            exp_random
        ).item()
        proximity_dist /= len(exp_random.minx[0])
        prox_distances.append(proximity_dist)
    
    proximity_distance = np.mean(prox_distances)
    return dpp_score, proximity_distance

# def n_cfs_score(comb_cfs, origin_instance,projection_config, origin_transformed=None):
#     transformer, exp_random = projection_config.transformer, projection_config.exp_random

#     if origin_transformed is None:
#         origin_transformed = transformer.transform(origin_instance.to_frame().T)

#     if len(comb_cfs) == 1:
#         dpp_score = 1.0
#         # Transform both just once
#         cf_transformed = transformer.transform(comb_cfs)
        
#         # Calculate proximity directly without loop
#         proximity_dist = compute_dist(
#             torch.tensor(origin_transformed.values).flatten(),
#             torch.tensor(cf_transformed.values).flatten(),
#             exp_random
#         )
#         proximity_distance = proximity_dist / len(exp_random.minx[0])
        
#         return dpp_score, proximity_distance

#     else:
#         dpp_score = dpp_style(transformer.transform(comb_cfs), exp_random).item()
#     prox_distances = []
#     for index, row in comb_cfs.iterrows():
#         proximity_dist = compute_dist(
#             torch.tensor(origin_transformed.values).flatten(),
#             torch.tensor(transformer.transform(row.to_frame().T).values).flatten(), exp_random)
#         proximity_dist /= len(exp_random.minx[0])
#         prox_distances.append(proximity_dist)
#     proximity_distance = np.mean(prox_distances)
#     return dpp_score, proximity_distance

def cons_score(cf,projection_config):
    cons_function,unary_cons_lst = projection_config.cons_function, projection_config.unary_cons_lst
    bin_cons = projection_config.bin_cons
    d = projection_config.d
    un_cons_brk = 0
    cons_brk = 0
    brk_rows = None
    # for i in range(len(constraints)):
    for i in bin_cons:
        # brk_per = d.data_df.apply(cons_function(cf, i), axis=1).sum() / len(d.data_df)
        if brk_rows is None:
            brk_rows = cons_function(d.data_df, cf, i)
        else:
            brk_rows = brk_rows | cons_function(d.data_df, cf, i)
        brk_per = (~cons_function(d.data_df, cf, i)).sum() / len(d.data_df)
        if brk_per < 1.0:
            cons_brk += 1
            # print(f'cons {i} is broken')
    for i in unary_cons_lst:
        brk_per = (~cons_function(d.data_df, cf, i)).sum() / len(d.data_df)
        if brk_per < 1.0:
            un_cons_brk += 1
            cons_brk += 1
    return un_cons_brk,brk_rows.sum(),cons_brk
    pass

# def metrics(origin_instance, cfs, dice_inst):
#     cons_total = 0
#     prox_total = 0
#     sparse_total = 0
#     brk_rows_total = 0
#     cfs_break = 0
#     for i, cf in cfs.iterrows():
#         print(f'cf number {i}')
#         constraints_score = cons_score(cf)
#         cons_total += constraints_score[0]
#         brk_rows_total += constraints_score[1]
#         prox_total += compute_dist(
#             torch.tensor(transformer.transform(origin_instance.to_frame().T).values).flatten(),
#             torch.tensor(transformer.transform(cf.to_frame().T).values).flatten(),
#             dice_inst)
#         sparse_total += (cf != origin_instance).sum()
#         if constraints_score[0] != 0 or constraints_score[1] != 0:
#             cfs_break +=1
#     print(f'Number of Cfs broke a constraint: {cfs_break}')
#     diversity_total = dpp_style(transformer.transform(cfs), dice_inst)
#     cons_total /= len(cfs)
#     brk_rows_total /= len(cfs)
#     prox_total /= len(cfs)
#     sparse_total /= len(cfs)
#     unique_dice_rows = cfs.drop_duplicates().shape[0]
#     total_dice_rows = cfs.shape[0]
#     print(f'DICE CFs - Total rows: {total_dice_rows}, Unique rows: {unique_dice_rows}, Duplicates: {total_dice_rows - unique_dice_rows}')
#     return cons_total,brk_rows_total, prox_total.item(), sparse_total, diversity_total.item(),cfs_break,unique_dice_rows

