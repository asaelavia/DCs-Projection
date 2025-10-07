import dice_ml
from perturb import *
from eval import *
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
from class_models import Mlp, pretrain
from dice_ml.utils.helpers import DataTransfomer
from torch.utils.data import TensorDataset, DataLoader


if __name__ == '__main__':

    args = parse_arguments()

    
    if not os.path.exists(f'data/{args.exp_name}'):
        os.makedirs(f'data/{args.exp_name}')
    with open(f'data/{args.exp_name}/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    constraints, dic_cols, cons_function, cons_feat, unary_cons_lst, unary_cons_lst_single, bin_cons = load_constraints(args.constraints_path)
    
    
    dice_runtimes = []
    # project_runtimes['project_constraints_exact_fast'] = []
    # project_runtimes['project_constraints_exact_fast_long'] = []
    df = pd.read_csv(args.dataset_path)
    # df = df.iloc[:len(df)//50,:]
    for col in df.columns:
        if col in args.cont_feat + ['label']:
            continue
        df[col] = df[col].astype('object')
    y = df['label']
    train_dataset, test_dataset, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0, stratify=y)
    x_train = train_dataset.drop('label', axis=1, errors='ignore')
    x_test = test_dataset.drop('label', axis=1, errors='ignore')
    categorical_column_names = [col for col in x_train.columns if col not in args.cont_feat]
    norm_params, normalized_medians = create_normalization_params(x_train,args.cont_feat,categorical_column_names)
    # Initialize coefs_dic and intercept
    coefs_dic = None
    intercept = None
    category_mappings = {}
    
    
    if args.load_test:
        with open(f'data/{args.exp_name}/rows_to_run_{args.exp_name}.pkl', 'rb') as f:
            x_test = pickle.load(f)
    else:
        with open(f'data/{args.exp_name}/rows_to_run_{args.exp_name}.pkl','wb') as f:
            pickle.dump(x_test[y_test == 0][:10],f)
    # exit(0)
    
    d = dice_ml.Data(dataframe=df, continuous_features=args.cont_feat, outcome_name='label')
    with open(f'data/{args.exp_name}/transformer_{args.exp_name}.pkl', 'wb') as f:
        pickle.dump(d, f)
    if args.load_transformer:
        with open(f'data/{args.exp_name}/transformer_{args.exp_name}.pkl', 'rb') as f:
            d = pickle.load(f)
    transformer = DataTransfomer('ohe-min-max')
    transformer.feed_data_params(d)
    transformer.initialize_transform_func()
    
    X_train = transformer.transform(x_train)
    model = Mlp(X_train.shape[1], [100, 1])
    model.train()
  
    
    
    train_loader = DataLoader(TensorDataset(torch.Tensor(X_train.values), torch.Tensor(y_train.values.astype('int'))),
                              64, shuffle=True)
    print('Train started')
    if args.load_model:
        model.load_state_dict(torch.load(f'data/{args.exp_name}/ml_model_state_dict.pth'))
    else:
        model = pretrain(model, 'cpu', train_loader, lr=1e-4, epochs=args.epochs)
        torch.save(model.state_dict(), f'data/{args.exp_name}/ml_model_state_dict.pth')
    
    # Initialize explainer as None
    explainer = None
    
    m = dice_ml.Model(model=model, backend='PYT', func="ohe-min-max")
    exp_random = dice_ml.Dice(d, m, method="gradient", constraints=False)

    model_labels = model(torch.Tensor(transformer.transform(x_test).values)).round().detach().numpy()
    x_test = x_test[model_labels == 0]
    features_to_vary = list(df.columns)
    for feat in args.fixed_feat + ['label']:
        features_to_vary.remove(feat)
    print('Features to vary:', features_to_vary)
    exp_random.generate_counterfactuals(x_test[0:1], total_CFs=2, desired_class=1,
                                                                            verbose=False,
                                                                            features_to_vary=features_to_vary, min_iter=500,max_iter=600,
                                                                            learning_rate=6e-1, proximity_weight=args.delta/100)

    
    # bfs_runtimes = {'fast': {},'solver': {}, 'corr': {},'dom': {}, 'long': {}}
    bfs_runtimes = {'solver': {}}
    solver_kwargs = {
        'constraints': constraints, 
        'dic_cols': dic_cols, 
        'cons_function': cons_function,
        'cons_feat': cons_feat, 
        'categorical_column_names': categorical_column_names,
        'coefs_dic': coefs_dic,
        'intercept': intercept,
        'unary_cons_lst': unary_cons_lst,
        'unary_cons_lst_single': unary_cons_lst_single,
        'bin_cons': bin_cons,
    }
    project_runtimes = {solver: [] for solver in ['exhaustive', 'solver', 'solver_linear','best_in_dataset']}
    projection_metrics = {solver: [] for solver in ['exhaustive', 'solver', 'solver_linear','dice_linear','dice','best_in_dataset']}
    perturb_metrics = {solver: [] for solver in ['exhaustive', 'solver', 'solver_linear','dice_linear','dice','best_in_dataset']}
    projection_config = ProjectionConfig(
        args=args,
        df=df,
        d=d,
        exp_random=exp_random,
        exp_random_lin= None,
        transformer=transformer,
        category_mappings=category_mappings,
        mode='solver',projection_func=project_solver,
        project_runtimes=project_runtimes,
        projection_metrics=projection_metrics,
        norm_params=norm_params,
        normalized_medians=normalized_medians,
        model=model,
        **solver_kwargs
    )
    

    
    prev={}
    for k in range(args.k_lower,args.k_upper):
        
        print(f'K = {k}')
        dice_runtimes = {}
        for i in range(args.num_samples):
            print(f'Sample: {i}')
            # dice_exp_random = None
            dice_cfs = None
            # query_instances = x_test[y_test == 0][i:i + 1]
            query_instances = x_test[i:i + 1]
            st = time.time()
            dice_exp_random = exp_random.generate_counterfactuals(query_instances, total_CFs=k, desired_class=1,
                                                                verbose=True,
                                                                features_to_vary=features_to_vary, max_iter=1000,
                                                                learning_rate=6e-1, proximity_weight=args.delta/100)
            et = time.time()
            dice_cfs_orig = dice_exp_random.cf_examples_list[0].final_cfs_df_sparse
            dice_dpp, dice_distance , l0_distance, l1_distance, pwise_div, min_dist_div = n_cfs_score(dice_cfs_orig.drop('label',axis=1, errors='ignore'), query_instances.iloc[0], projection_config)
            uncons,rows,cons_all = 0,0,0
            for _, cf in dice_cfs_orig.drop('label',axis=1, errors='ignore').iterrows():
                cons_res = cons_score(cf,projection_config)
                uncons += cons_res[0]
                rows += cons_res[1]
                cons_all += cons_res[2]
            perturb_metrics['dice'].append((dice_distance, dice_dpp, uncons/len(dice_cfs_orig), rows/len(dice_cfs_orig), cons_all/len(dice_cfs_orig), l0_distance, l1_distance, pwise_div, min_dist_div))
            dice_cfs_orig.to_csv(f'data/{args.exp_name}/dice_cfs_sample{i}_k{k}.csv', index=False)

            dice_cfs = [dice_exp_random.cf_examples_list[i].final_cfs_df_sparse for i in range(len(dice_exp_random.cf_examples_list))]
            elapsed_time = et - st
            dice_runtimes[k] = dice_runtimes.get(k, []) + [elapsed_time]
                    
                

            mode = args.projection_mode
            proj_params = (False, False)
            projection_config.mode = mode
            
            if mode == 'best_in_dataset':    
                if len(prev.get(i,x_train[1:0])) < k:
                    if k == args.k_lower:  # First k (k=2)
                        st = time.time()
                        prev[i] = x_train[1:0]  # Initialize empty DataFrame
                        df_cand = x_train
                        masks = []
                        for feature in args.fixed_feat:
                            masks.append(df_cand[feature] == query_instances.iloc[0][feature])
                        
                        # Combine all masks using logical AND
                        final_mask = masks[0]
                        for mask in masks[1:]:
                            final_mask = final_mask & mask
                        
                        # Return the filtered DataFrame
                        df_cand = df_cand[final_mask]
                        if len(df_cand) == 0:
                            print('Failed, no matches in dataset')
                        else:
                            print(f'Candidates in dataset: {len(df_cand)}')
                            # Run twice for k=2 to get initial 2 counterfactuals
                            # best_dataset = best_from_dataset(df_cand, query_instances.iloc[0], prev[i])
                            st = time.time()
                            for _ in range(args.k_lower):
                                if args.timeout is not None:
                                    timeout = args.timeout / args.k_lower
                                else:
                                    timeout = None
                                row = dice_cfs[0].iloc[0].drop('label',errors='ignore')
                                best_dataset = best_from_dataset(df_cand, row, prev[i],projection_config, timeout=timeout)

                                prev[i] = best_dataset
                            et = time.time()
                            elapsed_time = et - st
                            project_runtimes[mode].append(elapsed_time)
                            if len(best_dataset) != 0:
                                proj_row = best_dataset.iloc[0]
                                # Compute distance metrics
                                if not proj_row.equals(row):
                                    row = row.drop('label',errors='ignore')
                                    proj_row = proj_row.drop('label',errors='ignore')
                                    dist = compute_dist(
                                        torch.tensor(transformer.transform(
                                            proj_row.to_frame().T).values).flatten(),
                                        torch.tensor(transformer.transform(
                                            row.to_frame().T).values).flatten(), 
                                        exp_random
                                    )
                                    
                                    l0_distance = (proj_row != row).sum()
                                    l1_distance = l1_distance_with_cont_feat(proj_row, row, cont_feat=projection_config.args.cont_feat)
                                    projection_metrics[mode].append((dist, l0_distance, l1_distance))
                                else:
                                    dist = 0.0
                                    l0_distance = 0
                                    l1_distance = 0
                                    print('Row is equal to original row')
                            # projection metrics TODO
                            print(f'Best in dataset: {best_dataset}')
                            if len(best_dataset) > 0:
                                best_dataset.to_csv(f'data/{args.exp_name}/best_in_dataset_projection_sample{i}_k{k}.csv', index=False)

            if mode == 'exhaustive':
                # Prepare kwargs for fast projection
                projection_config.projection_func = project_constraints_exact_fast_long
                # projection_config.projection_func = project_constraints_exact_fast
                projected_cfs = project_counterfactuals(dice_cfs, projection_config)
                print(projected_cfs)
                if projected_cfs is not None and len(projected_cfs) > 0:
                    projected_cfs[0].to_csv(f'data/{args.exp_name}/exhaustive_projection_sample{i}_k{k}.csv', index=False)
                
                # algorithm_cfs_fast = bfs_counterfactuals(dice_cfs, projection_config,k, **solver_kwargs)

                
            if mode == 'solver':
                # Prepare kwargs for solver projection
                
                    
                projected_cfs = project_counterfactuals(dice_cfs, projection_config)
                if projected_cfs is not None and len(projected_cfs) > 0:
                    projected_cfs[0].to_csv(f'data/{args.exp_name}/solver_projection_sample{i}_k{k}.csv', index=False)
                if args.fixed_flag:
                    reset_solver_cache()

        # for mode in ['fast', 'dom']:
        for mode in ['solver','exhaustive', 'best_in_dataset']:
            if project_runtimes[mode]:
                if mode == 'solver':
                    max_index = project_runtimes[mode].index(max(project_runtimes[mode]))
                    project_runtimes[mode].pop(max_index)
                
                print(f'Number of results for {mode}: {len(projection_metrics[mode])}')

                print(f'{mode} Mean projection runtimes: {np.mean(project_runtimes[mode])}')
                print(f'{mode} STD projection runtimes: {np.std(project_runtimes[mode])}')
                print(f'{mode} Max projection runtimes: {np.max(project_runtimes[mode])}')
                print(f'{mode} Min projection runtimes: {np.min(project_runtimes[mode])}')

                mean_distance = np.mean([metric[0] for metric in projection_metrics[mode]])
                std_distance = np.std([metric[0] for metric in projection_metrics[mode]])
                max_distance = np.max([metric[0] for metric in projection_metrics[mode]])
                min_distance = np.min([metric[0] for metric in projection_metrics[mode]])
                print(f'{mode} Mean MAD projection distance: {mean_distance}')
                print(f'{mode} Mean MAD STD distance: {std_distance}')
                print(f'{mode} Max MAD projection distance: {max_distance}')
                print(f'{mode} Min MAD projection distance: {min_distance}')
                
                mean_l0_distance = np.mean([metric[1] for metric in projection_metrics[mode]])
                std_l0_distance = np.std([metric[1] for metric in projection_metrics[mode]])
                max_l0_distance = np.max([metric[1] for metric in projection_metrics[mode]])
                min_l0_distance = np.min([metric[1] for metric in projection_metrics[mode]])
                print(f'{mode} Mean L0 projection distance: {mean_l0_distance}')
                print(f'{mode} Mean L0 STD distance: {std_l0_distance}')
                print(f'{mode} Max L0 projection distance: {max_l0_distance}')
                print(f'{mode} Min L0 projection distance: {min_l0_distance}')

                mean_l1_distance = np.mean([metric[2] for metric in projection_metrics[mode]])
                std_l1_distance = np.std([metric[2] for metric in projection_metrics[mode]])
                max_l1_distance = np.max([metric[2] for metric in projection_metrics[mode]])
                min_l1_distance = np.min([metric[2] for metric in projection_metrics[mode]])
                print(f'{mode} Mean L1 projection distance: {mean_l1_distance}')
                print(f'{mode} Mean L1 STD distance: {std_l1_distance}')
                print(f'{mode} Max L1 projection distance: {max_l1_distance}')
                print(f'{mode} Min L1 projection distance: {min_l1_distance}')

    
        print(f'Mean DICE runtimes: {np.mean(dice_runtimes[k])}')
        print(f'STD DICE runtimes: {np.std(dice_runtimes[k])}')
        print(f'Max DICE runtimes: {np.max(dice_runtimes[k])}')
        print(f'Min DICE runtimes: {np.min(dice_runtimes[k])}')
    