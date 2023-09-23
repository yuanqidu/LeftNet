# A New Perspective on Building Efficient and Expressive 3D Equivariant Graph Neural Networks

This is the official implementation of the **LEFTNet** method proposed in the following paper.

Weitao Du, Yuanqi Du, Limei Wang, Dieqiao Feng, Guifeng Wang, Shuiwang Ji, Carla Gomes, Zhi-Ming Ma. "[A New Perspective on Building Efficient and Expressive 3D Equivariant Graph Neural Networks](https://arxiv.org/abs/2304.04757)". [NeurIPS 2023]

## Local Hierarchy of 3D Isomorphism
<p align="center">
<img src="https://github.com/yuanqidu/LeftNet/blob/main/assets/local.png" width="700" class="center" alt="local"/>
    <br/>
</p>

## From Local to Global
<p align="center">
<img src="https://github.com/yuanqidu/LeftNet/blob/main/assets/global.png" width="250" class="center" alt="global"/>
    <br/>
</p>

## LEFTNet implementation (LSE+FTE)
<p align="center">
<img src="https://github.com/yuanqidu/LeftNet/blob/main/assets/model.png" width="700" class="center" alt="model"/>
    <br/>
</p>


## Requirements
We include key dependencies below. The versions we used are in the parentheses. 
* PyTorch (1.9.0)
* PyTorch Geometric (1.7.2)

## Run

### QM9
```python
device=0
target='homo' # 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve','U0', 'U', 'H', 'G', 'Cv'
python main_qm9.py --device $device --target $target
```
### MD17
```python
device=9
name='aspirin' #aspirin, benzene2017, ethanol, malonaldehyde, naphthalene, salicylic, toluene, uracil
python main_md17.py --device 0 --name $name
``` 


## Citation
```
@article{du2023new,
  title={A new perspective on building efficient and expressive 3D equivariant graph neural networks},
  author={Du, Weitao and Du, Yuanqi and Wang, Limei and Feng, Dieqiao and Wang, Guifeng and Ji, Shuiwang and Gomes, Carla and Ma, Zhi-Ming},
  journal={arXiv preprint arXiv:2304.04757},
  year={2023}
}
```

## Acknowledgments

