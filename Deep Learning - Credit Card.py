# mengimport library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# membaca dataset
dataku = pd.read_csv('data_kartu_kredit.csv')

# tsandarisasi nilai asli 0 - 1 (amount)
from sklearn.preprocessing import StandardScaler
dataku['NewAmount'] = StandardScaler().fit_transform(dataku['Amount'].values.reshape(-1,1)) 

#mendefinisikan var dependen (y) dan independen (X)
y = np.array (dataku.iloc[:,-2])
X = np.array(dataku.drop(['Time','Amount','Class'], axis = 1)) #inplace = True untuk menghilangkan kolom

# membagi data training dan test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y,
                                                     test_size=0.2, random_state=111)

from sklearn.model_selection import train_test_split
X_train, X_validate, y_train, y_validate = train_test_split (X_train, y_train,
                                                     test_size=0.2, random_state=111)

# Membuat design ANN
from keras.models import Sequential #suply dari keras untuk membuat layers
from keras.layers import Dense, Dropout #Dense untuk menentukan neuron disetiap layer
                                        #dropout adalah suplier menentukan probabilitas hilangnya nodes secara random untuk menghilangkan overfiting
classifier = Sequential()
classifier.add(Dense(units=16, input_dim=29, activation='relu')) #units untuk + neuron #input_dim digunakan jika data kita hanya 1 kolom #kalo lebih gunakan input_shape
classifier.add(Dense(24, activation='relu')) #hidden layer
classifier.add(Dropout(0.25)) #angka probabilitas persen nods yang igngin dimatikan
classifier.add(Dense(20, activation='relu')) #hidden layer
classifier.add(Dense(24, activation='relu'))#hidden layer
classifier.add(Dense(1, activation='sigmoid')) #output layer
classifier.compile(optimizer ='adam', loss='binary_crossentropy',
                   metrics = ['accuracy'])

classifier.summary()

#visualisasi model ANN 
from keras.utils.vis_utils import plot_model
plot_model(classifier, to_file= 'model_ann.png', show_shapes=True,
           show_layer_names=False)

#Proses traininig model ANN
run_model = classifier.fit(X_train, y_train,
                           batch_size = 32,#angka yang habis membagi data / angka 8 bit
                           epochs = 10,
                           verbose = 1, #tanpa animasi 0
                           validation_data = (X_validate, y_validate))

# melihat parameter apa saja yang disimpan
print(run_model.history.keys())

#Plot accuracy training dan validation 
plt.plot(run_model.history['acc'])
plt.plot(run_model.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()

#Mengevaluasi model ANN 
evaluasi = classifier.evaluate (X_test, y_test)
print('accuracy:{:.2f}'.format(evaluasi[1]*100))

# Memprediksi test set 
hasil_prediksi = classifier.predict_classes(X_test)

# Membuat confusion matrix  (sederhana)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, hasil_prediksi)

# Membuat confusion matrix  (complex)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, hasil_prediksi)
cm_label = pd.DataFrame (cm, columns=np.unique(y_test), index=np.unique(y_test))
cm_label.index.name = 'actual'
cm_label.columns.name = 'predict'

# Membuat Cm dengan seaborn
sns.heatmap(cm_label, annot=True, cmap='Blues', fmt='g')

# Membuat classification report
from sklearn.metrics import classification_report
jumlah_kategori = 2
target_names = ['Class {}'.format(i) for i in range (jumlah_kategori)]
print(classification_report(y_test, hasil_prediksi, target_names = target_names))

# Our Data is UNBALANCE, how to deal with it?

# under sampling technic
# label 0 lebih banyak dari label 1 (label 1 dipakai semua, label 0 diambil sample random sampai jumlahnya sama seperti label 1)

# over sampling
# label 0 lebih banyak dari label 1 (label 0 dipakai apa adanya, dan lebel 1 dipakai berkali2 sampai jumlahnya mendekati kondisi real)
# tapi memiliki banyak kelemahan

# Melakukan teknik under sampling
index_fraud = np.array(dataku[dataku.Class==1].index)
n_fraud = len(index_fraud)
index_normal = np.array(dataku[dataku.Class==0].index)
index_data_normal = np.random.choice(index_normal, n_fraud, replace=False)
index_data_baru = np.concatenate([index_fraud, index_data_normal])
data_baru = dataku.iloc[index_data_baru,:]

# Membagi ke Var dependen (y) dan independen (X)
y_baru = np.array(data_baru.iloc[:, -2])
X_baru = np.array(data_baru.drop(['Time','Amount','Class'], axis=1))

# Membagi ke train dan test set 
X_train2, X_test_final, y_train2, y_test_final = train_test_split (X_baru, y_baru,
                                                                  test_size= 0.1,
                                                                  random_state = 111)

X_train2, X_test2, y_train2, y_test2 = train_test_split (X_train2, y_train2,
                                                        test_size= 0.1,
                                                        random_state=111)

X_train2, X_validate2, y_train2, y_validate2 = train_test_split (X_train2, y_train2,
                                                        test_size= 0.2,
                                                        random_state=111)

# Membuat design ANN baru (model balanced)
classifier2 = Sequential()
classifier2.add(Dense(units=16, input_dim=29, activation='relu')) #units untuk + neuron #input_dim digunakan jika data kita hanya 1 kolom #kalo lebih gunakan input_shape
classifier2.add(Dense(24, activation='relu')) #hidden layer
classifier2.add(Dropout(0.25)) #angka probabilitas persen nods yang igngin dimatikan
classifier2.add(Dense(20, activation='relu')) #hidden layer
classifier2.add(Dense(24, activation='relu'))#hidden layer
classifier2.add(Dense(1, activation='sigmoid')) #output layer
classifier2.compile(optimizer ='adam', loss='binary_crossentropy',
                   metrics = ['accuracy'])

classifier2.summary()

#Proses traininig model ANN
run_model2 = classifier2.fit(X_train2, y_train2,
                           batch_size = 8,#angka yang habis membagi data / angka 8 bit
                           epochs = 5,
                           verbose = 1, #tanpa animasi 0
                           validation_data = (X_validate2, y_validate2))

# Plot accuracy training dan validation 
plt.plot(run_model2.history['acc'])
plt.plot(run_model2.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()

plt.plot(run_model2.history['loss'])
plt.plot(run_model2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()

# Mengevaluasi model ANN2
evaluasi2 = classifier2.evaluate (X_test2, y_test2)
print('accuracy:{:.2f}'.format(evaluasi2[1]*100))

# Memprediksi test set 2
hasil_prediksi2 = classifier2.predict_classes(X_test2)

# Confusion matrix
cm2 = confusion_matrix(y_test2, hasil_prediksi2)
cm_label2 = pd.DataFrame (cm2, columns=np.unique(y_test2), index=np.unique(y_test2))
cm_label2.index.name = 'actual'
cm_label2.columns.name = 'predict'
sns.heatmap(cm_label2, annot=True, cmap='Blues', fmt='g')

# Membuat classification report
print(classification_report(y_test2, hasil_prediksi2, target_names = target_names))

# Memprediksi test set final
hasil_prediksi3 = classifier2.predict_classes(X_test_final)
cm3 = confusion_matrix(y_test_final, hasil_prediksi3)
cm_label3 = pd.DataFrame (cm3, columns=np.unique(y_test_final), index=np.unique(y_test_final))
cm_label3.index.name = 'actual'
cm_label3.columns.name = 'predict'
sns.heatmap(cm_label3, annot=True, cmap='Blues', fmt='g')

print(classification_report(y_test_final, hasil_prediksi3, target_names = target_names))



