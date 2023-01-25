# U-Net for Semantic Segmentation
This is a implementation of U-Net for semantic segmentation using Tensorflow. 

### Requirements
The project requires the following libraries to be installed:

- Tensorflow 2.x
- numpy
- cv2
- sklearn
- argparse

## Usage
To train a model, run the following command:

```
python train.py --data <data_folder> --csv <csv_output> --model <model_output> --epochs <epochs> --batch-size <batch_size> --img_ext <image_extension> --mask_ext <mask_extension> --imgsz <image_size>
```

You can also specify the following arguments:

`--data`: Data folder name. Default: data
`--csv`: CSV to output name. Default: results_unet_train.csv
`--model`: Model to output name. Default: model.h5
`--epochs`: Number of epochs for training. Default: 100
`--batch-size`: Batch size for training. Default: 2
`--img_ext`: Image extension. Default: .jpg
`--mask_ext`: Masks extension. Default: .png
`--imgsz`: Image size for inference. Default: [304, 3072]

## Architecture
The model use a U-Net architecture with the following structure:

![](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)

```
- Input layer: (304, 3072, 3)
- Convolutional layer 1: 32 filters, kernel size 3x3, stride 1, padding 1
- Convolutional layer 2: 32 filters, kernel size 3x3, stride 1, padding 1
- Max pooling layer: kernel size 2x2, stride 2
- Convolutional layer 3: 64 filters, kernel size 3x3, stride 1, padding 1
- Convolutional layer 4: 64 filters, kernel size 3x3, stride 1, padding 1
- Max pooling layer: kernel size 2x2, stride 2
- Convolutional layer 5: 128 filters, kernel size 3x3, stride 1, padding 1
- Convolutional layer 6: 128 filters, kernel size 3x3, stride 1, padding 1
- Max pooling layer: kernel size 2x2, stride 2
- Convolutional layer 7: 256 filters, kernel size 3x3, stride 1, padding 1
- Convolutional layer 8: 256 filters, kernel size 3x3, stride 1, padding 1
- Max pooling layer: kernel size 2x2, stride 2
- Convolutional layer 9: 512 filters, kernel size 3x3, stride 1, padding 1
- Convolutional layer 10: 512 filters, kernel size 3x3, stride 1, padding 1
- Max pooling layer: kernel size 2x2, stride 2
- Convolutional layer 11: 1024 filters, kernel size 3x3, stride 1, padding 1
- Convolutional layer 12: 1024 filters, kernel size 3x3, stride 1, padding 1
- Up sampling layer: size 2x2, stride 2
- Concatenation layer
- Convolutional layer 13: 512 filters, kernel size 3x3, stride 1, padding 1
- Convolutional layer 14: 512 filters, kernel size 3x3, stride 1, padding 1
- Up sampling layer: size 2x2, stride 2
- Concatenation layer
- Convolutional layer 15: 256 filters, kernel size 3x3, stride 1, padding 1
- Convolutional layer 16: 256 filters, kernel size 3x3, stride 1, padding
```