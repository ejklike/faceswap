# faceswap
experimental faceswap repository

## Dependencies
this code is tested on python>=3.5 and CUDA 8.0 environment with additional packages below:

```
pathlib==1.0.1
scandir==1.6
h5py==2.7.1
Keras==2.1.2
opencv-python==3.3.0.10
tensorflow-gpu==1.4.0
scikit-image
dlib # need some attention to install!
face_recognition
tqdm
```

## How to

1\. 얼굴 데이터를 준비한다.

- 데이터를 준비해둘 경로의 기본값은 `./data`이다.
  - 이 때, 사람 별로 얼굴 데이터를 폴더에 담아 준비해야 한다.
  - 예: `./data/trump`, `./data/cage`, ...
- 얼굴 이미지는 bounding box에 맞게 오려져 **mean facial landmark**에 맞추어 transformation 되어 있는 상태여야 한다.
  - 만약 그렇지 않다면, `extract.py`를 활용하여 학습 데이터를 준비하자.

```bash
$ python extract.py -ag -i ../faceon-dataset/5/face -o ./data/5
```

usage statement


```
usage: extract.py [-h] [-i INPUT_DIR] [-o OUTPUT_DIR] [-D {hog,cnn}] [-dl]
                  [-ag] [-c CUDA_VISIBLE_DEVICES]

Extract the faces from a pictures.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_DIR, --input-dir INPUT_DIR
                        Input directory. A directory containing the files you
                        wish to process. Defaults to 'input'
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Output directory. This is where the converted files
                        will be stored. Defaults to 'output'
  -D {hog,cnn}, --detector {hog,cnn}
                        Detector to use. 'cnn' detects much more angles but
                        will be much more resource intensive and may fail on
                        large files.
  -dl, --debug-landmarks
                        Draw landmarks for debug.
  -ag, --allow-growth   Sets allow_growth option of Tensorflow to spare memory
                        on some configs
  -c CUDA_VISIBLE_DEVICES, --cuda_visible_devices CUDA_VISIBLE_DEVICES
                        CUDA_VISIBLE_DEVICES value (e.g., -c 0,1)
```

2\. 모델을 학습한다.

- 학습된 모델은 기본으로 `./model/default/`에 저장된다.
- 학습 가능한 모델들은 추가예정
  - AE
  - GAN
  - **multiAE (AE에서 #face>2로 확장한 버전)**
  - TODO: Fast face-swap using CNN
  - TODO: unet

```bash
$ python train.py -t multiAE -c 0,1 -s 10 -ag -b 8 -si
```

usage statement

```
usage: train.py [-h] [-m MODEL_DIR] [-s SAVE_INTERVAL] [-si]
                [-t {AE,multiAE,GAN}] [-pl] [-bs BATCH_SIZE] [-ag]
                [-ep EPOCHS] [-c CUDA_VISIBLE_DEVICES]

train faceswap model

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_DIR, --model-dir MODEL_DIR
                        Model directory. This is where the training data will
                        be stored. Defaults to 'model/default'
  -s SAVE_INTERVAL, --save-interval SAVE_INTERVAL
                        Sets the number of iterations before saving the model.
  -si, --save-image     Sets save_image option to save current model results
  -t {AE,multiAE,GAN}, --trainer {AE,multiAE,GAN}
                        Select which trainer to use.
  -pl, --use-perceptual-loss
                        Use perceptual loss while training
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size, as a power of 2 (64, 128, 256, etc)
  -ag, --allow-growth   Sets allow_growth option of Tensorflow to spare memory
                        on some configs
  -ep EPOCHS, --epochs EPOCHS
                        Length of training in epochs.
  -c CUDA_VISIBLE_DEVICES, --cuda_visible_devices CUDA_VISIBLE_DEVICES
                        CUDA_VISIBLE_DEVICES value (e.g., -c 0,1)
```

3\. 원하는 사진을 입력하여 faceswap 결과를 살펴본다.

```
$ python convert.py -t multiAE -o ./output/trump -m ./results/default -tg trump -ag -c 2 -b 0
```

usage statement

```
usage: convert.py [-h] [-i INPUT_DIR] [-o OUTPUT_DIR] [-m MODEL_DIR]
                  [-t {AE,multiAE,GAN}] [-tg TARGET] [-ct {Masked,Adjust}]
                  [-D {hog,cnn}] [-b BLUR_SIZE] [-S]
                  [-M {rect,facehull,facehullandrect}]
                  [-e EROSION_KERNEL_SIZE] [-mh] [-sm] [-aca] [-ag]
                  [-c CUDA_VISIBLE_DEVICES]

Convert a source image to a new one with the face swapped

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_DIR, --input-dir INPUT_DIR
                        Input directory. A directory containing the files you
                        wish to process. Defaults to 'input'
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Output directory. This is where the converted files
                        will be stored. Defaults to 'output'
  -m MODEL_DIR, --model-dir MODEL_DIR
                        Model directory. A directory containing the trained
                        model you wish to process. Defaults to
                        './results/model/'
  -t {AE,multiAE,GAN}, --trainer {AE,multiAE,GAN}
                        Select the trainer that was used to create the model.
  -tg TARGET, --target TARGET
                        Select target.
  -ct {Masked,Adjust}, --converter {Masked,Adjust}
                        Converter to use.
  -D {hog,cnn}, --detector {hog,cnn}
                        Detector to use. 'cnn' detects much more angles but
                        will be much more resource intensive and may fail on
                        large files.
  -b BLUR_SIZE, --blur-size BLUR_SIZE
                        Blur size. (Masked converter only)
  -S, --seamless        Use cv2's seamless clone. (Masked converter only)
  -M {rect,facehull,facehullandrect}, --mask-type {rect,facehull,facehullandrect}
                        Mask to use to replace faces. (Masked converter only)
  -e EROSION_KERNEL_SIZE, --erosion-kernel-size EROSION_KERNEL_SIZE
                        Erosion kernel size. (Masked converter only). Positive
                        values apply erosion which reduces the edge of the
                        swapped face. Negative values apply dilation which
                        allows the swapped face to cover more space.
  -mh, --match-histgoram
                        Use histogram matching. (Masked converter only)
  -sm, --smooth-mask    Smooth mask (Adjust converter only)
  -aca, --avg-color-adjust
                        Average color adjust. (Adjust converter only)
  -ag, --allow-growth   Sets allow_growth option of Tensorflow to spare memory
                        on some configs
  -c CUDA_VISIBLE_DEVICES, --cuda_visible_devices CUDA_VISIBLE_DEVICES
                        CUDA_VISIBLE_DEVICES value (e.g., -c 0,1)
```

## References
- [deepfakes](https://github.com/deepfakes/faceswap): faceswap model code
