# Boosting Detection in Crowd Analysis via Underutilized Output Features



### [Project Page](https://fredfyyang.github.io/Crowd-Hat/) |   [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Boosting_Detection_in_Crowd_Analysis_via_Underutilized_Output_Features_CVPR_2023_paper.pdf)
<br>

<img src='imgs/pipeline.png' align="right" width=960>  
  

<br><br><br>
This repository contains the official PyTorch implementation of our paper [Boosting Detection in Crowd Analysis via Underutilized Output Features](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Boosting_Detection_in_Crowd_Analysis_via_Underutilized_Output_Features_CVPR_2023_paper.pdf). We provide a plug-and-play module leveraging the detection outputs as features to boost the performance in crowd analysis.


[Boosting Detection in Crowd Analysis via Underutilized Output Features](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Boosting_Detection_in_Crowd_Analysis_via_Underutilized_Output_Features_CVPR_2023_paper.pdf)  
 [Shaokai Wu*](), [Fengyu Yang*](https://fredfyyang.github.io/)<br>
Jilin University and University of Michigan<br>
 In CVPR 2023

 This module works on top of the **Head** of detection methods and it can be tailored for each detection Head, thus we call it Crowd **Hat**. 
 In this repository, Hat_LSC-CNN means Crowd Hat implemented on LSC-CNN, and likewise the others.

### Updated 2024.4.19
The authors are currently immersed in various research projects and burdersome curriculums this quarter. The codebase will be updated as soon as possible when time permits. We appreciate your understanding.

### Updated 2023.12.12
This repository will be reorganized into a clearer format these days. 

### Implementation
Crowd Hat can be easily added on detection-based methods. Just follow the steps below:
1. Select a detection-based method and get the pipeline model. You can either train the model on crowd dataset or get the model from corresponding repositories (if any).
2. Fix the weights of the detection pipeline, do inference across the training dataset, and save the data to the disk.
3. Use the data above to train the Crowd Hat network, including the count decoder and region-adaptive NMS decoder.

This repository currently provide two pytorch implementation of Hat_LSC-CNN and Hat_TopoCount. 

### TODO
Crowd Hat + PSDNN and Crowd Hat + SDNet will be coming soon. [SDNet](https://github.com/WangyiNTU/Point-supervised-crowd-detection) is officially implemented with Tensorflow.

### Citation
If you use this code for your research, please cite our [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Boosting_Detection_in_Crowd_Analysis_via_Underutilized_Output_Features_CVPR_2023_paper.pdf).
```
@InProceedings{Wu_2023_boosting,
      author    = {Wu, Shaokai and Yang, Fengyu},
      title     = {Boosting Detection in Crowd Analysis via Underutilized Output Features},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      month     = {June},
      year      = {2023},
      pages     = {15609-15618}
}
```

