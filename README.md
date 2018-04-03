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

1\. `source`, `target` 얼굴 데이터를 준비한다.

- 이 때, 각 데이터를 `./source/`와 `./target/`에 넣어 준비하거나, 별도의 폴더 위치를 인자로 넣어주자.
- 얼굴 이미지는 **mean facial landmark**에 맞추어 transformation 되어 있는 상태여야 한다.

2\. 모델을 학습한다.

- 학습된 모델은 `./model/modelname/`에 저장된다.
- 학습 가능한 모델들은 추가예정
  - AE
  - GAN

```bash
$ python train.py
```

인자 설명

```usage statement

```

3\. 원하는 사진을 입력하여 faceswap 결과를 살펴본다.

- 모델명...
- target 누구...?

```
python convert.py
```

## References
- [deepfakes](https://github.com/deepfakes/faceswap): faceswap model code
