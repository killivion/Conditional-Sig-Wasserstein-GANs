This repository is mainly based on the following to repositorys and their respective underlying papers:

https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/td3
[#papers](https://arxiv.org/abs/1802.09477)

https://github.com/SigCGANs/Conditional-Sig-Wasserstein-GANs
[#papers](https://onlinelibrary.wiley.com/doi/full/10.1111/mafi.12423)


The requirements files are based on Conda for local use in an CPU environemnt (it has to be noted that SigCWGAN cannot be fully utilized in this frame though, SigCWGAN application in the LRZ AI cluster 'ColabRequirements', and TD3 application in the LRZ AI cluster 'LRZ_requirements'.

The Jupyter Notebook code contains sample code to train and generate data under adjustable arguments. 
The training files train.py and TD3/td3_train.py contain default values and present all adjustable parameters.




## Training SigCWGAN:
python train.py -use_cuda -total_steps 1000

## Evaluation SigCWGAN:
python evaluate.py -use_cuda

The trained generator and the numerical results will be saved in the 'numerical_results' folder during training process. Running evaluate.py will produce the 'summary.csv' files.



## Training TD3:
python TD3/td3_train.py -mode train -total_timesteps 10000

## Evaluation via Tensorboard
tensorboard --logdir ./TD3/logs

The trained actor will be saved to 'agent', tensorboard logs are saved to 'logs'


