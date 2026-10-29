#!/usr/bin/env sh

echo "preprocessing: noop"
python src/preprocessing.py 0
python src/train.py
mv checkpoints/best_unet_resnet34.pt checkpoints/noop.pt
rm -rf data/building_dataset
mv data/building_dataset.* data/building_dataset

echo "preprocessing: bilaterial"
python src/preprocessing.py 1
python src/train.py
mv checkpoints/best_unet_resnet34.pt checkpoints/bilaterial.pt
rm -rf data/building_dataset
mv data/building_dataset.* data/building_dataset

echo "preprocessing: clahe"
python src/preprocessing.py 2
python src/train.py
mv checkpoints/best_unet_resnet34.pt checkpoints/clahe.pt
rm -rf data/building_dataset
mv data/building_dataset.* data/building_dataset

echo "preprocessing: gamma"
python src/preprocessing.py 3
python src/train.py
mv checkpoints/best_unet_resnet34.pt checkpoints/gamma.pt
rm -rf data/building_dataset
mv data/building_dataset.* data/building_dataset

echo "preprocessing: sharpen"
python src/preprocessing.py 4
python src/train.py
mv checkpoints/best_unet_resnet34.pt checkpoints/sharpen.pt
rm -rf data/building_dataset
mv data/building_dataset.* data/building_dataset

echo "preprocessing: chainA"
python src/preprocessing.py 5
python src/train.py
mv checkpoints/best_unet_resnet34.pt checkpoints/chainA.pt
rm -rf data/building_dataset
mv data/building_dataset.* data/building_dataset

echo "preprocessing: chainB"
python src/preprocessing.py 6
python src/train.py
mv checkpoints/best_unet_resnet34.pt checkpoints/chainB.pt
rm -rf data/building_dataset
mv data/building_dataset.* data/building_dataset

echo "preprocessing: chainC"
python src/preprocessing.py 7
python src/train.py
mv checkpoints/best_unet_resnet34.pt checkpoints/chainC.pt
rm -rf data/building_dataset
mv data/building_dataset.* data/building_dataset
