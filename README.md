## Self-Sustaining Representation Expansion forNon-Exemplar Class-Incremental Learning
This is the implementation of the paper "Self-Sustaining Representation Expansion forNon-Exemplar Class-Incremental Learning" (accepted to CVPR2022).

For more information, check out the paper on [[arXiv](https://arxiv.org/abs/2203.06359)].

## Requirements

- Python 3.8
- PyTorch 1.8.1 (>1.1.0)
- cuda 11.2

## Preparing Datasets
Download following datasets:

> #### 1. CIFAR-100

> #### 2. Tiny-ImageNet

> #### 3. ImageNet

Locate the above three datasets under ./data directory.


## Incremental Training.

> ### 1. Download pretrained models to the 'pre' folder.
> Pretrained models are available on our [[Google Drive](https://drive.google.com/file/d/1tjJ985pCidjH3NxaOt-M62YbJIB93hVx/view?usp=sharing)].


> ### 2. Training
> ```bash
> sh train_cvpr.sh 
> ```

## Base Training
> Coming soon.

## Requirements
> We thank the following repos providing helpful components/functions in our work.
- [PASS](https://github.com/Impression2805/CVPR21_PASS)
- [RepVGG](https://github.com/DingXiaoH/RepVGG)
   
## BibTeX
If you use this code for your research, please consider citing:
````BibTeX
@article{zhu2022self,
  title={Self-Sustaining Representation Expansion for Non-Exemplar Class-Incremental Learning},
  author={Zhu, Kai and Zhai, Wei and Cao, Yang and Luo, Jiebo and Zha, Zheng-Jun},
  journal={arXiv preprint arXiv:2203.06359},
  year={2022}
}
````
