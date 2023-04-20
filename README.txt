This file gives an overview of how we acquired our results. This file contains the folowing:

1. An instalation guide for heat. In which we explain what steps are taken to install the HEAT module.
2. What map structures we use. It is recomended that you use the same structure.
3. How to reproduce the results from heat.
4. How preprocess new data from RPLAN.
5. How to create the results from RPLAN.

After these steps. There are two files left:
Image_processing.ipynb <-- this produces the pictures from our blog
Create_annot_files.ipynb <-- this contains some experiments into annotation files.

###################################
#--- 1. Instalation guide HEAT ---#
###################################

To install heat walk though the steps in heat_install.txt

###########################
#--- 2.  Map structure ---#
###########################

|- HEAT_Reproduction (<-- This is our git folder)
|- heat (<-- this is the heat folder only containing data, checkpoints, results and evaluation)
  |- checkpoints (<-- this is the checkpoint folder that can be downloaded from the origional heat git repo)
  |- results
       |- input_output  (<-- empty folder)
       |- output_gt  (<-- empty folder)
  |- s3d_floorplan_eval (<-- empty folder) 
  |- data
       |- s3d_floorplan (<-- this is the original floorplan data from heat, contains some txt files and folders)
	 |    |- annot (<-- should be filled)
       |    |- density (<-- should be filled)
       |    |- normals (<-- should be filled)
	 |    |- max (<-- this folder you have to create)
	 |   	|- *various .txt files
       |- RPLAN_small (<-- create this)
		|- annot (<-- empty)
		|- density (<-- empty)
		|- geometry (<-- empty)
		|- geometry_casper (<-- fill with geometry.pickle files)
		|- max (<-- empty)
		|- normals (<-- empty)
		|- original (<-- empty)
		|- original_casper (<-- fill with original RPLAN png images)
		|- *sample.npy (<-- copy this from our git repo)
		|- *test_list (<-- copy this from our git repo)

###########################################
#--- 3. Reproduction results from heat ---#
###########################################
make sure you are in the HEAT_Reproduction folder
Run the following code:

python infer.py --checkpoint_path ../heat/checkpoints/ckpts_heat_s3d_256/checkpoint.pth  --dataset s3d_floorplan --image_size 256 --viz_base ../heat/results/viz_heat_s3d_256 --save_base ../heat/results/npy_heat_s3d_256

cd s3d_floorplan_eval
python visualize_npy.py (<-- edit the file location in this function to go to the heat data instead of RPLAN_small)

###########################################
#--- 4. Preprocess RPLAN data          ---#
###########################################

First, copy the geometry.pickle files to the heat\data\RPLAN_small\geometry_casper folder
and copy the original png images to the heat\data\RPLAN_small\original_casper folder

Second, run RPLAN_Preprocessing.ipnyb notebook. This takes all the valid combinations between geometry and original files, gives them the correct numbers, and then creates density, normal, max and annotaion. 

###########################################
#--- 5. Create results for RPLAN data ---#
###########################################
Make sure you are in the HEAT_Reproduction folder
Run the following code:

python infer.py --checkpoint_path ../heat/checkpoints/ckpts_heat_s3d_256/checkpoint.pth  --dataset RPLAN_small --image_size 256 --viz_base ../heat/results/viz_heat_RPLAN_small_256 --save_base ../heat/results/npy_heat_RPLAN_small_256

cd s3d_floorplan_eval
python visualize_npy.py

Now run the Qualitive_evaluation.ipnyb notebook to produce the Qualitive evaluation results.



