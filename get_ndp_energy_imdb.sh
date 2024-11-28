#!/bin/bash

base_dir='output/examples/benchmarks'
ndp_config='m2ndp'

python energy/ndp_energy_calculate.py --config energy/ndp_energy_config_64B.yaml --input ${base_dir}/imdb_gt_lt_FP32/${ndp_config}/energy.out --output ${base_dir}/imdb_gt_lt_FP32/${ndp_config}/energy_breakdown.txt

python energy/ndp_energy_calculate.py --config energy/ndp_energy_config_64B.yaml --input ${base_dir}/imdb_gteq_lt_INT64/${ndp_config}/energy.out --output ${base_dir}/imdb_gteq_lt_INT64/${ndp_config}/energy_breakdown.txt

python energy/ndp_energy_calculate.py --config energy/ndp_energy_config_64B.yaml --input ${base_dir}/imdb_lt_INT64/${ndp_config}/energy.out --output ${base_dir}/imdb_lt_INT64/${ndp_config}/energy_breakdown.txt

python energy/ndp_energy_calculate.py --config energy/ndp_energy_config_64B.yaml --input ${base_dir}/imdb_three_col_AND/${ndp_config}/energy.out --output ${base_dir}/imdb_three_col_AND/${ndp_config}/energy_breakdown.txt

python energy/ndp_energy_calculate.py --config energy/ndp_energy_config_64B.yaml --input ${base_dir}/imdb_two_col_AND/${ndp_config}/energy.out --output ${base_dir}/imdb_two_col_AND/${ndp_config}/energy_breakdown.txt