#!/usr/bin/bash

# python generate_maze2d_datasets.py --fname pointmass_original
# python generate_maze2d_datasets.py --fname pointmass_obscure_1 --obscure_mode center
# python generate_maze2d_datasets.py --fname pointmass_obscure_3 --obscure_mode 3
python generate_maze2d_datasets.py --fname pointmass_fpv --env_name maze2d-theta-umaze-v0 \
 --camera fpv --policy theta_10m_20ep_global_std-1.5

# python generate_maze2d_datasets.py --fname pointmass_theta_top --env_name maze2d-theta-umaze-v0 \
#  --policy theta_5m_20ep_global --visible_target
