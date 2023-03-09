An official implementation of paper "Semi-Supervised Cell Recognition under Point Supervision"



## Data preparation

Two choices.

- You can download the raw data from [CoNIC](https://conic-challenge.grand-challenge.org/) to **datasets/conic** folder and then run this [script](https://github.com/windyzy/SSPCR/blob/main/datasets/conic/prepare_data.py)) to obtain training/validation/test subsets . 
- A more convenient way is to download the ready-made data subsets from Google Drive (after review).



## Train

To reproduce baseline models:

```python
python train_base.py --dataset conic --space 8 --num_classes 6 --eos_coef 0.4 --match_dis 6 --output_dir=he_sup_5_base --ratio 5
```

To train PCR models under our proposed framework:

```python
python train_semi.py --dataset conic --space 8 --num_classes 6 --eos_coef 0.4 --match_dis 6 --output_dir=he_sup_5_semi --ratio 5 --enable_semi_sup
```



## Test

To test baseline models, run

```
python train_base.py --dataset conic --space 8 --num_classes 6 --match_dis 6 --ratio 5 --test
```

To test models trained using our framework, run

```python
python train_semi.py --dataset conic --space 8 --num_classes 6 --match_dis 6 --ratio 5 -test
```



The checkpoints will be also released here after review.

