#!/bin/sh

# alias here
alias untarall='for f in *.tar; do tar xvf "$f"; done'

# dataTag
dataTag='202107'
cold_data='2021071003'
# dataTag='201906'
# cold_data='2019060103'

tar_file='test.tar'

# set directory
branch_dir="/g3/wanghao/kezuo/xhj/GRAPES_GFS3.2/GRAPES_GFS3.2_fix_autobc_cyclerun/"
# branch_dir="/g3/wanghao/kezuo/xhj/GRAPES_GFS3.2/GRAPES_GFS3.2_fix_autobc_new/"

remote_dir="${branch_dir}/DATABAK/4DVAR_results/"
local_dir="./${dataTag}/"

# login using and tar data
ssh -i ~/.ssh/id_rsa wanghao@10.40.140.18 "cd ${remote_dir}; tar -cvf ${tar_file} ./*/checkxb*_*.tar"

# create local directory
if [ ! -d ${local_dir} ]; then
	mkdir ${local_dir}
fi

# download data
scp -i  ~/.ssh/id_rsa -r wanghao@10.40.140.18:${remote_dir}/${tar_file} ${local_dir}

# untar-1
cd ${local_dir}
tar -xvf ${tar_file}

# untar-2
for f in ./??????????; do cd "$f" && untarall && cd ../; done

# remove coldstart data
rm ./${cold_data}/*
rmdir ./${cold_data}

# clean tar files
rm ${tar_file}
for f in ./??????????; do cd "$f" && rm checkxb*_*.tar && cd ../; done

exit

