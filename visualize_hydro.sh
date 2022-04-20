#!/bin/sh

# init time
init='2021072103'
tag='check_model'

# set directory
remote_dir=/g3/wanghao/kezuo/xhj/GRAPES_GFS3.2/GRAPES_GFS3.2_fix_autobc_cyclerun/DATABAK/FCST_results/
local_dir=/mnt/d/GRAPES/check_model/${init}/
visualize_dir=/home/xhj/wkspcs/GRAPES_GFS/warm_start_2022/

# create local directory
if [ ! -d ${local_dir} ]; then
	mkdir ${local_dir}
fi

# download data
scp -i  ~/.ssh/id_rsa  -r wanghao@10.40.140.18:${remote_dir}/model.ctl_${init}_*  ${local_dir}
scp -i  ~/.ssh/id_rsa  -r wanghao@10.40.140.18:${remote_dir}/modelvar${init}_*  ${local_dir}

# convert data
cd ${visualize_dir}
# source activate rdop
python test_convert_to_nc.py "${local_dir}/model.ctl_${init}_*"

# plot data
python plot_hydro.py ${local_dir} ${tag}
mv ${tag}_*.png ${local_dir}

