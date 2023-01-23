
# langkah-langkah pre-processing atau pembersihan data

# 1. mengimport library
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from keras.datasets import fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

#jenis kategori
kategori = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal',
            'Shirt','Sneakers','Bag','Ankle boot']

# 0 = T-shirt/top
# 1 = Trouser
# 2 = Pullover
# 3 = Dress
# 4 = Coat
# 5 = Sandal
# 6 = Shirt
# 7 = Sneakers
# 8 = Bag
# 9 = Ankle boot

# 2. menampilkan gambar 

i = random.randint(1, len(X_train))
plt.figure()
plt.imshow(X_train[i,:,:], cmap='gray')
plt.title('item ke {} - Kategori = {}'.format(i, kategori[y_train[i]]))
plt.show()


#menampilkan banyak gambar sekaligus
nrow = 3
ncol = 3
fig, axes = plt.subplots (nrow, ncol)
axes = axes.ravel() #gunakan ravel jika ncol dan nrow >1 (fungsi meratakan)
ntraining = len(X_train)
for i in np.arange(0, nrow*ncol):
    indexky = np.random.randint(0, ntraining)
    axes[i].imshow(X_train[indexky, :, :], cmap = 'gray')
    axes[i].set_title(int(y_train[indexky]), fontsize = 10)
    axes [i].axis ('off') #untuk menghilangkan garis sumbu
plt.subplots_adjust(hspace=0.3)


# 3. mengubah data pixel dari 0-255 menjadi 0-1
# untuk memudahkan komputasi (normalize datasets)
X_train = X_train/255
X_test = X_test/255


# 4. split datasets menjadi training dan validate set
# untuk menghindari overfitting/underfitting
from sklearn.model_selection import train_test_split
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train,
                                                            test_size=0.2,
                                                            random_state=123)


# 5. Mengubah dimensi data set
# Karena untuk masuk ke "keras" harus ada dimensi tambahan (channelnya = 28,28,1)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], *(28, 28, 1)) # (*) unpackage untuk membongkar tupple 3d yang kita inginkan
X_validate = X_validate.reshape(X_validate.shape[0], *(28, 28, 1))


# Langkah-langkah Deep Learning

# 1. Mengimpor library Keras
# convolution dan flatenning
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam #Adam = fungsi optimizer di deep learning
# mendefinisikan model CNN / classifier
classifier = Sequential()
classifier.add(Conv2D(32,(3, 3), input_shape=(28, 28, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size= (2, 2))) #ukuran dimana kita mencari nilai maksimumnya
classifier.add(Dropout(0.25)) #Model akan mudah digeneralisir dan meghindari overfitting

# Flatenning dan membuat FC - NN
classifier.add(Flatten())
classifier.add(Dense(activation='relu', units=32))
classifier.add(Dense(activation='sigmoid', units=10)) #kita harus punya 10 neuron, karena ada 10  szie
classifier.compile(loss='sparse_categorical_crossentropy',
                   optimizer=Adam(lr= 0.001),
                   metrics=['accuracy'])

# melihat ringkasan yang sudah kita buat
classifier.summary()

# visualisasi neural network (need answer)
from keras.utils.vis_utils import plot_model
plot_model (classifier, to_file='model_NN.png',
            show_shapes=True,
            show_layer_names=False)

# 2. Melakukan training model
run_model = classifier.fit(X_train, y_train, 
                            batch_size = 500,# batch size bisa habis dibagi data (membuat training lebih cepat)
                            nb_epoch = 30, #proses pembelajaran dari input layer sampe output (berulang sebanyak nilai bobot)
                            verbose = 1, #optional animation (bobot 0 jika tanpa animasi)
                            validation_data = (X_validate, y_validate))

# 3. menampilkan parameter yang disimpan selama proses training
print (run_model.history.keys())

#proses plotting accuracy selama training
plt.plot (run_model.history['acc'])
plt.plot (run_model.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validate'], loc='upper left')
plt.show()

#proses plotting loss selama training
plt.plot (run_model.history['loss'])
plt.plot (run_model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validate'], loc='upper left')
plt.show()

# 4. Mengevaluasi Model CNN (test accuracy)
evaluasi = classifier.evaluate(X_test, y_test)
print ('Test Accuracy = {:.2f}%'.format(evaluasi[1]*100))

# 5. save model (bisa running diwebsite dan dimanapun tanpa loading epoch dsb)
classifier.save('model_cnn.hd5', include_optimizer=True)
print ('Model sudah disimpan')

# 6. load model
# jika kita punya datasets yang agak mirip kita bisa mentraining lagi sehingga model kita belajar lebih banyak
'''
from keras.models import load_model
classifier = load_model(model_cnn_fashion.hd5)
'''

#memprediksi kategori di test set
hasil_prediksi_test = classifier.predict_classes(X_test)

#membuat plot hasil prediksi
fig, axes = plt.subplots(3,3)
axes = axes.ravel()
for i in np.arange (0, 3*3) :
    axes [i].imshow(X_test[i].reshape(28,28), cmap='gray')
    axes [i].set_title('Hasil Prediksi = {}\n Label Asli = {}'.format(hasil_prediksi_test[i], y_test[i]),fontsize = 8)
    axes [i].axis('off')
plt.subplots_adjust(hspace=0.6)
#plt.subplots_adjust(wspace=0.3)

# mengevaluasi hasil prediksi dengan confusion matrix
from sklearn.metrics import confusion_matrix
import pandas as pd
cm = confusion_matrix (y_test, hasil_prediksi_test)
cm_label = pd.DataFrame(cm, columns = np.unique(y_test), index = np.unique(y_test))
cm_label.index.name = 'asli' #sumbu Y
cm_label.columns.name = 'prediksi' #sumbu X
plt.figure(figsize=(14,10))
sns.heatmap(cm_label, annot=True)

#membuat ringkasan performa model
from sklearn.metrics import classification_report
jumlah_kategori = 10
nama_target = ['kategori {}'.format(i) for i in range(jumlah_kategori)]
print (classification_report(y_test, hasil_prediksi_test, target_names=nama_target))







