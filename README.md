# rep10825984
## Requirements
- fastmri
- torch
- pytorch-lighting
- ...
## Usage
### for evaluating existing model
create directory results/xxxx and put corresponding checkpoints in proper position, then modify *bootstrap_estimate_test.yaml*
```bash
python main.py --cfg-path ./bootstrap_estimate_test.yaml
```
### for zero-shot training
```bash
python main_per.py --cfg-path ./bootrec2d_zs_train.yaml
```
### for normal multiple images training
```bash
python main.py --cfg-path ./bootrec_multi_train.yaml
```
