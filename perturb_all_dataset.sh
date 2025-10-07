#!/bin/bash


python -u perturb_test.py \
    --cont_feat age education_num hours_per_week \
    --fixed_feat age race sex \
    --dataset_path data/datasets/adult.csv \
    --constraints_path data/constraints/adult_dcs.txt \
    --k_lower 5 \
    --k_upper 6 \
    --model_state_dict_path adult_hard.pth \
    --dataset adult_solver_linear \
    --mode hard \
    --exp_name adult_pandp \
    --solver_timeout 500000 \
    --delta 50 \
    --num_samples 3 \
    --timeout 200000\
    > adult_pandp.out


python -u perturb_test.py \
    --cont_feat age education_num hours_per_week \
    --fixed_feat age race sex \
    --dataset_path data/datasets/adult.csv \
    --constraints_path data/constraints/adult_dcs.txt \
    --k_lower 5 \
    --k_upper 6 \
    --model_state_dict_path adult_hard.pth \
    --dataset adult_solver_linear \
    --mode hard \
    --exp_name adult_pandp_fixed \
    --solver_timeout 500000 \
    --delta 50 \
    --num_samples 3 \
    --timeout 200000\
    --fixed_flag \
    > adult_pandp_fixed_flag.out

python -u perturb_test.py \
    --cont_feat age education_num hours_per_week \
    --fixed_feat age race sex \
    --dataset_path data/datasets/adult.csv \
    --constraints_path data/constraints/adult_dcs.txt \
    --k_lower 5 \
    --k_upper 6 \
    --model_state_dict_path adult_hard.pth \
    --dataset adult_solver_linear \
    --mode hard \
    --exp_name adult_pandp_linear \
    --solver_timeout 500000 \
    --delta 50 \
    --num_samples 3 \
    --timeout 200000\
    --linear_pandp \
    > adult_pandplinear.out


python -u perturb_test.py \
    --cont_feat age education_num hours_per_week \
    --fixed_feat age race sex \
    --dataset_path data/datasets/adult.csv \
    --constraints_path data/constraints/adult_dcs.txt \
    --k_lower 5 \
    --k_upper 6 \
    --model_state_dict_path adult_hard.pth \
    --dataset adult_solver_linear \
    --mode hard \
    --exp_name adult_pandp_linear_fixed \
    --solver_timeout 500000 \
    --delta 50 \
    --num_samples 3 \
    --timeout 200000\
    --fixed_flag \
    --linear_pandp \
    > adult_pandplinear_fixed_flag.out

python -u perturb_test.py \
    --cont_feat age education_num hours_per_week \
    --fixed_feat age race sex \
    --dataset_path data/datasets/adult.csv \
    --constraints_path data/constraints/adult_dcs.txt \
    --k_lower 5 \
    --k_upper 6 \
    --model_state_dict_path adult_hard.pth \
    --dataset adult_solver_linear \
    --mode hard \
    --exp_name adult_linear \
    --solver_timeout 500000 \
    --delta 50 \
    --num_samples 3 \
    --timeout 200000\
    --linear_model \
    > adult_linear.out


python -u perturb_test.py \
    --cont_feat age education_num hours_per_week \
    --fixed_feat age race sex \
    --dataset_path data/datasets/adult.csv \
    --constraints_path data/constraints/adult_dcs.txt \
    --k_lower 5 \
    --k_upper 6 \
    --model_state_dict_path adult_hard.pth \
    --dataset adult_solver_linear \
    --mode hard \
    --exp_name adult_linear_fixed \
    --solver_timeout 500000 \
    --delta 50 \
    --num_samples 3 \
    --timeout 200000\
    --fixed_flag \
    --linear_model \
    > adult_linear_fixed_flag.out

# python -u perturb_test.py \
#     --fixed_feat Genderstr \
#     --cont_feat Salaryint SingleExempint MarriedExempint ChildExempint \
#     --dataset_path data/tax_no_city.csv \
#     --constraints_path data/tax_adcs_city.txt \
#     --k_lower 5 \
#     --k_upper 6 \
#     --model_state_dict_path model_state_dict_tax.pth \
#     --dataset tax_solver_linear \
#     --mode hard \
#     --load_model \
#     --exp_name tax_dice \
#     --solver_timeout 10000 \
#     --num_samples 11 \
#     --timeout 200000\
#     --linear_pandp \
#     > tax_pandplinear.out

# python -u perturb_test.py \
#     --fixed_feat Genderstr \
#     --cont_feat Salaryint SingleExempint MarriedExempint ChildExempint \
#     --dataset_path data/tax_no_city.csv \
#     --constraints_path data/tax_adcs_city.txt \
#     --k_lower 5 \
#     --k_upper 6 \
#     --model_state_dict_path model_state_dict_tax.pth \
#     --dataset tax_solver_linear \
#     --mode hard \
#     --load_model \
#     --exp_name tax_dice \
#     --solver_timeout 10000 \
#     --timeout 200000\
#     --num_samples 11 \
#     --fixed_flag \
#     --linear_pandp \
#     > tax_pandplinear_fixed_flag.out


# python -u perturb_test.py \
#     --cont_feat age wage_per_hour weeks_worked_in_year capital_gains capital_losses num_person_Worked_employer dividend_from_Stocks \
#     --fixed_feat age hispanic_origin sex race marital_status country_self country_mother citizenship \
#     --dataset_path data/census_clean.csv \
#     --constraints_path data/census_top_1000_adcs_200.txt \
#     --k_lower 5 \
#     --k_upper 6 \
#     --num_samples 11 \
#     --model_state_dict_path model_state_dict_census.pth \
#     --dataset census_solver_linear \
#     --mode hard \
#     --epochs 2 \
#     --exp_name census_dice \
#     --solver_timeout 300000 \
#     --timeout 200000 \
#     --preload \
#     --linear_pandp \
#     > census_pandplinear.out

# python -u perturb_test.py \
#     --cont_feat age wage_per_hour weeks_worked_in_year capital_gains capital_losses num_person_Worked_employer dividend_from_Stocks \
#     --fixed_feat age hispanic_origin sex race marital_status country_self country_mother citizenship \
#     --dataset_path data/census_clean.csv \
#     --constraints_path data/census_top_1000_adcs_200.txt \
#     --k_lower 5 \
#     --k_upper 6 \
#     --num_samples 11 \
#     --model_state_dict_path model_state_dict_census.pth \
#     --dataset census_solver_linear \
#     --mode hard \
#     --epochs 2 \
#     --exp_name census_dice \
#     --solver_timeout 300000 \
#     --timeout 200000 \
#     --fixed_flag \
#     --preload \
#     --linear_pandp \
#     > census_pandplinear_fixed_flag.out

# python -u perturb_test.py \
#     --cont_feat beds bath propertysqft \
#     --fixed_feat type locality sublocality \
#     --dataset_path data/nyhouse.csv \
#     --constraints_path data/ny_good_adcs.txt \
#     --k_lower 5 \
#     --k_upper 6 \
#     --model_state_dict_path model_state_dict_ny_modes.pth \
#     --dataset ny_solver_linear \
#     --mode hard \
#     --load_model \
#     --exp_name ny_dice \
#     --load_transformer \
#     --num_samples 11 \
#     --timeout 200000 \
#     --solver_timeout 1000 \
#     --linear_pandp \
#     > ny_pandplinear.out

# python -u perturb_test.py \
#     --cont_feat beds bath propertysqft \
#     --fixed_feat type locality sublocality \
#     --dataset_path data/nyhouse.csv \
#     --constraints_path data/ny_good_adcs.txt \
#     --k_lower 5 \
#     --k_upper 6 \
#     --model_state_dict_path model_state_dict_ny_modes.pth \
#     --dataset ny_solver_linear \
#     --mode hard \
#     --load_model \
#     --exp_name ny_dice \
#     --load_transformer \
#     --num_samples 11 \
#     --timeout 200000 \
#     --solver_timeout 1000 \
#     --fixed_flag \
#     --linear_pandp \
#     > ny_pandplinear_fixed_flag.out