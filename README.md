# rep10825984
## Requirements
- fastmri
- torch
- pytorch-lighting
- ...
## Usage
[example data](https://drive.google.com/file/d/10WlIlpawxdgC5TkuuAKmfmgrkVpwLQrN/view?usp=sharing) (data from fastmri validation set and computed coil sensitivity)
[example existing model checkpoints for evaluating](https://drive.google.com/file/d/1SeCDGQWEkq5ClYaz11Lvsc9jBHtC2pIm/view?usp=sharing) (model trained independently with labeled data)
### for evaluating existing model
create directory results/xxxx and put corresponding checkpoints in proper position, then modify *bootstrap_estimate_test.yaml*
```bash
python main.py --cfg-path ./bootstrap_estimate_test.yaml
```
### for zero-shot training
```bash
python main_per.py --cfg-path ./bootrec2d_zs_train.yaml
```
