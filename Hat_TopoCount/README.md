# Crowd Hat + Localization-Based Method TopoCount

####Crowd Hat module is a plug-and-play crowd-analysis enhancement network proposed in our paper [Boosting Detection in Crowd Analysis via Underutilized Output Features, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Boosting_Detection_in_Crowd_Analysis_via_Underutilized_Output_Features_CVPR_2023_paper.pdf)
####This module is though designed for detection-based methods, we found that it can work well with localization-based method and density-based method. 
####Here is the pytorch implementation of Crowd Hat + TopoCount, which is a localization-based method proposed in [**Shahira Abousamra, Minh Hoai, Dimitris Samaras, Chao Chen, Localization in the Crowd with Topological Constraints, AAAI 2021.**](https://arxiv.org/pdf/2012.12482.pdf)

## Preparation
1. Download the checkpoint of TopoCount into 'checkpoints/', which is available at [Google Drive](https://drive.google.com/drive/folders/1qhg3ITOY_qEaNLDfgCP-LOE0Xj1ZH0P7?usp=sharing) 
2. Download NWPU-Crowd dataset from [NWPU-Crowd Benchmark](https://www.crowdbenchmark.com)
3. Modify the path of dataset in [hat_config.py](hat_config.py)


## Train

1. Generating training data by [train_hat.py](crowd_hat/train_hat.py)
   

    prepare_training_data(cfg.img_root, cfg.json_root)


2. Train the count decoder.


    train_count_decoder(5, 120, resume=0)


    


## Evaluate

Run the [evaluation.py](crowd_hat/evaluation.py)


    topo = build_model()
    evaluate_counting(cfg.img_root, cfg.json_root, topo)


## Test

Test on NWPU-Crowd dataset by [evaluation.py](crowd_hat/evaluation.py)


    test_nwpu_counting(cfg.nwpu_test,topo)

The result will be save to the result root in [hat_config.py](hat_config.py)
You can directly submit the result to [NWPU-Crowd Counting Benchmark](https://www.crowdbenchmark.com/nwpucrowd.html)