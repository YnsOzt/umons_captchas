## Acknowledgements
This is a slightly modified version of [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)

```
@inproceedings{baek2019STRcomparisons,
  title={What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis},
  author={Baek, Jeonghun and Kim, Geewook and Lee, Junyeop and Park, Sungrae and Han, Dongyoon and Yun, Sangdoo and Oh, Seong Joon and Lee, Hwalsuk},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year={2019},
  note={to appear},
  pubstate={published},
  tppubtype={inproceedings}
}
```

## Download dataset from [here](https://drive.google.com/open?id=1hBwTmuuWXRd5T7MxXXJHS_4qQqB6DXv0)
* raw_datasets contains:
  * train.txt / test.txt / val.txt which are the alignment of captchas with their respective labels
  * captchas which contains the train, test and validation catpchas
  
* lmdb_datasets contains:
  * train_constrained which contains the ready to use training lmdb dataset
  * test_constrained which contains the ready to use test lmdb dataset
  * val_constrained which contains the ready to use validation lmdb dataset

## Our pretrained models
Download pretrained model from [here](https://drive.google.com/open?id=1nTP0ZOm97qSKlr8RpZUXXpgKWMH7bSQt). You can get more information about the model used by reading [this](https://arxiv.org/abs/1904.01906) paper.

The drive link contains those models:
  * TPS-ResNet-BiLSTM-Attn : full implementation of the model presented in the paper above
  * resnet_34_frozen_pretrained_constrained : model using PyTorch ResNet34 as the feature extractor with frozen parameters and trained on a constrained dataset
  * resnet_34_finetuned_pretrained_constrained : model using PyTorch ResNet34 as the feature extractor with finetuned parameters and trained on a constrained dataset
  * resnet_34_finetuned_pretrained_unconstrained_10k : model using PyTorch ResNet34 as the feature extractor with finetuned parameters and trained on an unconstrained dataset which contains 10k data
  * resnet_34_finetuned_pretrained_unconstrained_15k : model using PyTorch ResNet34 as the feature extractor with finetuned parameters and trained on an unconstrained dataset which contains 15k data

## Training our models from scratch
  * resnet_34_frozen_pretrained_constrained : 
```
CUDA_VISIBLE_DEVICES=0 python3 train.py \
--train_data path_to_lmdb_train_folder --valid_data path_to_lmdb_validation_folder \
--select_data '/' --batch_ratio 1  --character "krosdgtupweavih lcmnbyzjxfq?,'!-.&" \
--Transformation TPS --FeatureExtraction ResNet_PyTorch --SequenceModeling BiLSTM --Prediction Attn \
--batch_size 64 --valInterval {79} \
--imgH 224 --imgW 224 --rgb \
--early_stopping_param 'accuracy' --early_stopping_patience 20 \
--freeze_FeatureExtraction \
--experiment_name ResNet32_Frozen \
--ignore_x_vals 10
```
  * resnet_34_finetuned_pretrained_constrained :
 ```
test
```
  * resnet_34_finetuned_pretrained_unconstrained_10k : 
```
test
```
  * resnet_34_finetuned_pretrained_unconstrained_15k :
  
```
test
```
