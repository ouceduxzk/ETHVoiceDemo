import glob
import os
import librosa
import numpy as np
import tensorflow 
import tensorflow as tf 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Dropout
from tensorflow.keras import Model


from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

'''0 = silence
1 = singing
2 = speaking
'''

input_dim = 1280

def extract_features(file_name):
    X, sample_rate = librosa.load(file_name)
    X = X[:2*sample_rate]
    
    #stft = np.abs(librosa.stft(X))
    mfccs = np.array(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40, n_fft=1024, hop_length=512).T)
    print(mfccs.shape)
    # pad to 3600
    paded_feature = np.zeros((input_dim))
    mfccs =  mfccs.flatten()
    if (mfccs.shape[0] > input_dim):
        paded_feature = mfccs[:input_dim]
    else:
        paded_feature[:mfccs.shape[0]] = mfccs 
    return paded_feature

def get_label(sub_dir):
    if sub_dir == "singing":
        return 1
    elif sub_dir == "speaking":
        return 2
    else:
        return 0

def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav'):
    ignored = 0
    features, labels, name = [], np.empty(0), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        print(sub_dir, label)
        cur_category_files = glob.glob(os.path.join(parent_dir, sub_dir, file_ext))

        sub_features = np.zeros((len(cur_category_files), input_dim))
        len 
        for i, fn in enumerate(cur_category_files):
            #import pdb; pdb.set_trace()
            mfccs = extract_features(fn)
            #ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            sub_features[i, :] = mfccs
            #l = [get_label(sub_dir)] * (mfccs.shape[0])
            labels = np.append(labels, get_label(sub_dir))
            
        features.append(sub_features)
        
    return np.vstack(features), np.array(labels, dtype = np.int)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


if __name__ == "__main__":
    parent_dir = 'data'

    sub_dirs = ['silence', 'singing', 'speaking']

    try:
        label = np.load('labels.npy')
        y_train, y_val = label["train"], label["val"]
        features = np.load('features.npy')
        x_train, x_val = features["train"], features["val"]
        
        test_feautre_labels=  np.load("test.npy")
        x_test, y_test = test_feautre_labels["x_test"], test_feautre_labels["y_test"]
        
        print("Features and labels found!")
    except:
        print("Extracting features...")
        x_test, y_test = parse_audio_files(parent_dir+"/test", sub_dirs)

        x_train, y_train = parse_audio_files(parent_dir + "/train",sub_dirs)
        x_val, y_val = parse_audio_files(parent_dir + "/val", sub_dirs)
        
        with open('features.npy', 'wb') as f1:
            np.savez(f1,train =x_train, val = x_val)
            
        with open('labels.npy', 'wb') as f2:
            np.savez(f2, train= y_train, val = y_val)
            
        with open('test.npy', 'wb') as f1:
            np.savez(f1,x_test =x_test, y_test = y_test)
            
    y_train = one_hot_encode(y_train)
    y_val = one_hot_encode(y_val)
    y_test = one_hot_encode(y_test)

    idxs = np.arange(x_train.shape[0])
    random_idxs = np.random.shuffle(idxs)

    x_train = x_train[random_idxs]
    y_train = y_train[random_idxs]

    val_idxs = np.arange(x_val.shape[0])
    val_random_idxs = np.random.shuffle(val_idxs)

    x_val = x_val[val_random_idxs]
    y_val = y_val[val_random_idxs]

    print("Training...")


    n_classes = 3

    x_train = x_train.squeeze()
    x_val = x_val.squeeze()

    x_train=np.reshape(x_train,(x_train.shape[0], input_dim,1,1))
    x_val=np.reshape(x_val,(x_val.shape[0], input_dim,1, 1))

    x_test=np.reshape(x_test,(x_test.shape[0], input_dim,1, 1))

    y_train = y_train.squeeze()
    y_val = y_val.squeeze()
    y_test = y_test.squeeze()

    print(x_train.shape, x_val.shape, x_test.shape)


    model=Sequential()

    #adding layers and forming the model
    model.add(Conv2D(8,kernel_size=15,strides=1,padding="Same",activation="relu",input_shape=(input_dim,1, 1)))
    model.add(MaxPooling2D(padding="same"))

    model.add(Conv2D(16,kernel_size=7,strides=1,padding="same",activation="relu"))
    model.add(MaxPooling2D(padding="same"))

    model.add(Conv2D(32,kernel_size=3,strides=1,padding="same",activation="relu"))
    model.add(MaxPooling2D(padding="same"))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(64,activation="relu"))
    model.add(Dropout(0.1))

    # model.add(Dense(128,activation="relu"))
    # model.add(Dropout(0.3))

    model.add(Dense(n_classes,activation="softmax"))


    opt = tensorflow.keras.optimizers.Adam(learning_rate=0.0001)

    #compiling
    model.compile(optimizer=opt, loss="categorical_crossentropy",metrics=["accuracy"])

    #training the model
    model.fit(x_train,y_train,batch_size=32,epochs=10,validation_data=(x_val,y_val))


    #train and test loss and scores respectively
    train_loss_score=model.evaluate(x_train,y_train)
    test_loss_score=model.evaluate(x_val,y_val)
    print(train_loss_score)
    print(test_loss_score)


    model.summary()

    model.save('voice_models')


    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TF Lite model.
    with tf.io.gfile.GFile('model.tflite', 'wb') as f:
        f.write(tflite_model)

    print("predicting on test dataset")
    prediction = model.predict_classes(x_test)

    y_test_label = np.argmax(y_test, axis = 1)

    print("acc ={}".format(accuracy_score(y_test_label, prediction)))
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


    testData = np.expand_dims(x_val[5],axis=0)
    atData = np.float32(testData)

    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], atData)
    print(input_data.dtype)
    print(testData.shape)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)