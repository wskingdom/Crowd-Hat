# Crowd Hat + Detection-Based Method LSC-CNN

Crowd Hat module is a plug-and-play crowd-analysis enhancement network proposed in our paper [Boosting Detection in Crowd Analysis via Underutilized Output Features, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Boosting_Detection_in_Crowd_Analysis_via_Underutilized_Output_Features_CVPR_2023_paper.pdf)

Here is the pytorch implementation of Crowd Hat + LSC-CNN, which is a localization-based method proposed in the paper.


## Preparation
1. Download the checkpoint of LSC-CNN into 'checkpoints/' [Huggingface](https://huggingface.co/WSKINGDOM/lsc_nwpu/blob/main/scale_4_epoch_41.pth) 
2. Download NWPU-Crowd dataset from [NWPU-Crowd Benchmark](https://www.crowdbenchmark.com)
3. Modify the path of dataset in [hat_config.py](hat_config.py)


## Train

1. Generating training data by [train_hat.py](crowd_hat/train_hat.py)
   

    prepare_training_data(cfg.img_root, cfg.json_root)


2. Train the count decoder.


    train_count_decoder(5, 120, resume=0)


    


## Evaluate

Run the [evaluation.py](crowd_hat/evaluation.py)



    evaluate_counting(cfg.img_root,cfg.json_root,0.2)



## Test

Test on NWPU-Crowd dataset by [evaluation.py](crowd_hat/evaluation.py)

Crowd Counting
    test_nwpu_counting(cfg.nwpu_test)


Crowd Localization
    test_nwpu_localization(cfg.nwpu_test)

The result will be save to the result root in [hat_config.py](hat_config.py)
You can directly submit the result to [NWPU-Crowd Counting Benchmark](https://www.crowdbenchmark.com/nwpucrowd.html)
