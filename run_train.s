#!/bin/bash                                                                                                                                                                                                                                  
#                                                                                                                                                                                                                                         
#SBATCH --job-name=train_mlp                                                                                                                                                                                                                     
#SBATCH --nodes=1                                                                                                                                                                                                                           
#SBATCH --cpus-per-task=1                                                                                                                                                                                                                    
#SBATCH --time=50:59:00                                                                                                                                                                                                                      
#SBATCH --mem=50GB                                                                                                                                                                                                                            
#SBATCH --gres=gpu:1

module pytorch/python3.5/0.2.0_3

python train.py --lr_anneal 1 > 1.out
#python train.py --lr_anneal 10 > 2.out
#python train.py --lr_anneal 50 > 3.out
