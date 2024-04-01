# InterGen: Diffusion-based Multi-human Motion Generation under Complex Interactions
### [ProjectPage](https://tr3e.github.io/intergen-page/) | [Paper](https://doi.org/10.1007/s11263-024-02042-6)  | [Arxiv](https://arxiv.org/abs/2304.05684)  |  [InterHuman Dataset](https://drive.google.com/drive/folders/1oyozJ4E7Sqgsr7Q747Na35tWo5CjNYk3?usp=sharing)
[Han Liang](https://tr3e.github.io/), Wenqian Zhang, Wenxuan Li, [Jingyi Yu](http://www.yu-jingyi.com/), [Lan Xu](http://www.xu-lan.com/).</br>



This repository contains the official implementation for the paper: [InterGen: Diffusion-based Multi-human Motion Generation
under Complex Interactions (IJCV 2024)](https://doi.org/10.1007/s11263-024-02042-6). 

Our work is capable of simultaneously generating high-quality interactive motions of two people with only text guidance, enabling various downstream tasks including person-to-person generation, inbetweening, trajectory control and so forth. 

For more qualitative results, please visit our [webpage](https://tr3e.github.io/intergen-page/).

<p float="left">
  <img src="./readme/pipeline.png" width="900" />
</p>


<!-- ![trajectorycontrol](https://github.com/tr3e/InterGen/blob/main/trajectorycontrol.gif) -->

<!-- ### Person-to-person generation

<p float="left">
  <img src="./readme/a2b.gif" width="900" />
</p>

<!-- #### Person-to-person generation
![person-to-person](https://github.com/tr3e/InterGen/blob/main/a2b.gif) -->
<!--
### Inbetweening

<p float="left">
  <img src="./readme/inbetweening.gif" width="900" />
</p> -->

<!-- #### Inbetweening
![inbetweening](https://github.com/tr3e/InterGen/blob/main/inbetweening.gif) -->


<!-- ## Abstract
We have recently seen tremendous progress in diffusion advances for generating realistic human motions. Yet, they largely disregard the rich multi-human interactions.

In this paper, we present InterGen, an effective diffusion-based approach that incorporates human-to-human interactions into the motion diffusion process, which enables layman users to customize high-quality two-person interaction motions, with only text guidance.

We first contribute a multimodal dataset, named InterHuman. It consists of about 107M frames for diverse two-person interactions, with accurate skeletal motions and 23,337 natural language descriptions. For the algorithm side, we carefully tailor the motion diffusion model to our two-person interaction setting. Then, we propose a novel representation for motion input in our interaction diffusion model, which explicitly formulates the global relations between the two performers in the world frame. We further introduce two novel regularization terms to encode spatial relations, equipped with a corresponding damping scheme during the training of our interaction diffusion model. -->


## InterHuman Dataset
<!-- ![interhuman](https://github.com/tr3e/InterGen/blob/main/interhuman.gif) -->

InterHuman is a comprehensive, large-scale 3D human interactive motion dataset encompassing a diverse range of 3D motions of two interactive people, each accompanied by natural language annotations.

<p float="left">
  <img src="./readme/interhuman.gif" width="900" />
</p>

It is made available under [Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license. You can access the dataset in our [webpage](https://tr3e.github.io/intergen-page/) with the google drive link for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicating any changes** that you've made. The redistribution of the dataset is **prohibited**. 



## Getting started

This code was tested on `Ubuntu 20.04.1 LTS` and requires:

* Python 3.8
* conda3 or miniconda3
* CUDA capable GPU (one is enough)

### 1. Setup environment

```shell
conda create --name intergen
conda activate intergen
pip install -r requirements.txt
```

### 2. Get data


Download the data from [webpage](https://tr3e.github.io/intergen-page/). And put them into ./data/.

#### Data Structure
```sh
<DATA-DIR>
./annots                //Natural language annotations where each file consisting of three sentences.
./motions               //Raw motion data standardized as SMPL which is similiar to AMASS.
./motions_processed     //Processed motion data with joint positions and rotations (6D representation) of SMPL 22 joints kinematic structure.
./split                 //Train-val-test split.
```


## Demo



### 1. Download the checkpoint
Run the shell script:

```shell
./prepare/download_pretrain_model.sh
```

### 2. Modify the configs
Modify config files ./configs/model.yaml and ./configs/infer.yaml


### 3. Modify the input file ./prompts.txt like:

```sh
In an intense boxing match, one is continuously punching while the other is defending and counterattacking.
With fiery passion two dancers entwine in Latin dance sublime.
Two fencers engage in a thrilling duel, their sabres clashing and sparking as they strive for victory.
The two are blaming each other and having an intense argument.
Two good friends jump in the same rhythm to celebrate.
Two people bow to each other.
Two people embrace each other.
...
```

### 4. Run
```shell
python tools/infer.py
```
The results will be rendered and put in ./results/


## Train


Modify config files ./configs/model.yaml ./configs/datasets.yaml and ./configs/train.yaml, and then run:

```shell
python tools/train.py
```


## Evaluation



### 1. Modify the configs
Modify config files ./configs/model.yaml and ./configs/datasets.yaml

### 2. Run
```shell
python tools/eval.py
```


## Application
<p float="left">
  <img src="./readme/trajectorycontrol.gif" width="900" />
</p>



## Citation

If you find our work useful in your research, please consider citing:

```
@article{liang2024intergen,
  title={Intergen: Diffusion-based multi-human motion generation under complex interactions},
  author={Liang, Han and Zhang, Wenqian and Li, Wenxuan and Yu, Jingyi and Xu, Lan},
  journal={International Journal of Computer Vision},
  pages={1--21},
  year={2024},
  publisher={Springer}
}
```




## Licenses
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

All material is made available under [Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license. You can **use and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicating any changes** that you've made.
