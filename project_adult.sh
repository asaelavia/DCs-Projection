#!/bin/bash
python -u projection_test.py \
    --cont_feat age education_num hours_per_week \
    --fixed_feat age race sex \
    --dataset_path data/datasets/adult.csv \
    --constraints_path data/constraints/adult_dcs.txt \
    --k_lower 1 \
    --k_upper 2 \
    --model_state_dict_path adult_hard.pth \
    --dataset adult_solver_linear \
    --mode hard \
    --exp_name adult_project_best_in_dataset \
    --solver_timeout 1000 \
    --timeout 5000 \
    --projection_mode best_in_dataset \
    > adult_project.out
    # > adult_projection_only_to_1000_best_in_compare_test.out


# python -u projection_test.py \
#     --cont_feat age education_num hours_per_week \
#     --fixed_feat age race sex  \
#     --dataset_path data/adult_clean.csv \
#     --constraints_path data/adult_good_adcs_test.txt \
#     --k_lower 1 \
#     --k_upper 2 \
#     --model_state_dict_path adult_hard.pth \
#     --dataset adult_solver_linear \
#     --mode hard \
#     --load_model \
#     --dice_path adult_dice \
#     --load_transformer \
#     --solver_timeout 5000 \
#     > adult_projection_only_to_5000.out


# python -u projection_test.py \
#     --cont_feat age education_num hours_per_week \
#     --fixed_feat age race sex \
#     --dataset_path data/adult_clean.csv \
#     --constraints_path data/adult_good_adcs_test.txt \
#     --k_lower 1 \
#     --k_upper 2 \
#     --model_state_dict_path adult_hard.pth \
#     --dataset adult_solver_linear \
#     --mode hard \
#     --load_model \
#     --dice_path adult_dice \
#     --load_transformer \
#     --fixed_flag \
#     --solver_timeout 1000 \
#     > adult_projection_only_fixed_to_1000.out

# python -u projection_test.py \
#     --cont_feat age education_num hours_per_week \
#     --fixed_feat age race sex \
#     --dataset_path data/adult_clean.csv \
#     --constraints_path data/adult_good_adcs_test.txt \
#     --k_lower 1 \
#     --k_upper 2 \
#     --model_state_dict_path adult_hard.pth \
#     --dataset adult_solver_linear \
#     --mode hard \
#     --load_model \
#     --dice_path adult_dice \
#     --load_transformer \
#     --fixed_flag \
#     --solver_timeout 5000 \
#     > adult_projection_only_fixed_to_5000.out

# python -u projection_test.py \
#     --cont_feat age education_num hours_per_week \
#     --fixed_feat age race sex \
#     --dataset_path data/adult_clean.csv \
#     --constraints_path data/adult_good_adcs_test.txt \
#     --k_lower 5 \
#     --k_upper 6 \
#     --model_state_dict_path adult_hard.pth \
#     --dataset adult_solver_linear \
#     --mode hard \
#     --load_model \
#     --dice_path adult_dice \
#     --load_transformer \
#     --linear_model \
#     --solver_timeout 500000 \
#     --preload \
#     --delta 50 \
#     --num_samples 11 \
#     > adult_projection_linear_to_500000.out



# python -u projection_test.py \
#     --cont_feat age education_num hours_per_week \
#     --fixed_feat age race sex \
#     --dataset_path data/adult_clean.csv \
#     --constraints_path data/adult_good_adcs_test.txt \
#     --k_lower 5 \
#     --k_upper 6 \
#     --model_state_dict_path adult_hard.pth \
#     --dataset adult_solver_linear \
#     --mode hard \
#     --load_model \
#     --dice_path adult_dice \
#     --load_transformer \
#     --linear_model \
#     --preload \
#     --solver_timeout 10000 \
#     > adult_projection_linear_to_10000.out

# python -u projection_test.py \
#     --cont_feat age education_num hours_per_week \
#     --fixed_feat age race sex \
#     --dataset_path data/adult_clean.csv \
#     --constraints_path data/adult_good_adcs_test.txt \
#     --k_lower 5 \
#     --k_upper 6 \
#     --model_state_dict_path adult_hard.pth \
#     --dataset adult_solver_linear \
#     --mode hard \
#     --load_model \
#     --dice_path adult_dice \
#     --load_transformer \
#     --linear_model \
#     --delta 5 \
#     --solver_timeout 20000 \
#     --blackbox \
#     > adult_projection_linear_to_20000_harmonic_diversity_test.out
    # --preload \



# python -u projection_test.py \
#     --cont_feat age education_num hours_per_week \
#     --fixed_feat age race sex \
#     --dataset_path data/adult_clean.csv \
#     --constraints_path data/adult_good_adcs_test.txt \
#     --k_lower 1 \
#     --k_upper 2 \
#     --model_state_dict_path adult_hard.pth \
#     --dataset adult_solver_linear \
#     --mode hard \
#     --load_model \
#     --dice_path adult_dice \
#     --load_transformer \
#     --solver_timeout 1000 \
#     --timeout 200000 \
#     > adult_best_in_dataset.out