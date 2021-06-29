# ETH Voice Classfication Demo 
This is the repository for ETH Voice Classification Demo 

## Dataset 
Datasets are collected on youtube (singing, speaking with chinese , english and german languages), and then use
```
 ffmpeg -i input.wav  -f segment -segment_time 3 -c copy input%4d.wav
```

Make sure create a following dataset folder and put .wav under each categories

-- data
  -- train 
    -- silence 
    -- singing 
    -- speaking 

  -- val 
    -- silence 
    -- singing 
    -- speaking 

  -- test 
    -- silence 
    -- singing 
    -- speaking 

## Training 

```
python train_val.py 

```

## Compile Anddroid

```
cp model.tflite Android_Tensorflow_AudioClassifier/app/src/main/assets/
```

Then Run Android Studio to compile the app, with usb debugging turn on. 


## Reference and Acknowledgement
This code borrowed a lot from this github repository (https://github.com/VVasanth/Android_Tensorflow_AudioClassifier), but with robust android development (original app is buggy) as well as new strategies for classfy mfcc features. 
