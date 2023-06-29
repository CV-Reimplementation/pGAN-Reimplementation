# pGAN

This repository is unofficial implementation of [pGAN](https://ieeexplore.ieee.org/abstract/document/8653423) with PyTorch.

A refined version of [Original implementation](https://github.com/icon-lab/pGAN-cGAN).

## Fix several problems
1. Code updated to newest version
2. Serveral bugs fixed
3. Better training and testing process

### Training

```python
python pGAN.py --dataroot datasets/IXI --name pGAN_run --direction BtoA --training
```

name - name of the experiment

direction - direction of synthesis. If it is set to 'AtoB' synthesis would be from data_x to data_y, and vice versa

### Testing
```python
python pGAN.py --dataroot datasets/IXI --name pGAN_run --direction BtoA --phase test --results_dir results/
```

name - name of the experiment

direction - direction of synthesis. If it is set to 'AtoB' synthesis would be from data_x to data_y, and vice versa

# Citation
You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.
```
@article{dar2019image,
  title={Image Synthesis in Multi-Contrast MRI with Conditional Generative Adversarial Networks},
  author={Dar, Salman UH and Yurt, Mahmut and Karacan, Levent and Erdem, Aykut and Erdem, Erkut and {\c{C}}ukur, Tolga},
  journal={IEEE Transaction on Medical Imaging},
  year={2019},
  publisher={IEEE}
}
```
For any questions, comments and contributions, please contact Salman Dar (salman[at]ee.bilkent.edu.tr) <br />

(c) ICON Lab 2019

## Acknowledgments
This code is based on implementations by [pGAN-cGAN](https://github.com/icon-lab/pGAN-cGAN) and [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
