import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.models import Sequential

path = "dataset/"
benign = pd.read_csv(path + "benign.csv")

mirai_scan = pd.read_csv(path + "mirai_scan.csv")
mirai_syn = pd.read_csv(path + "mirai_syn.csv")
mirai_ack = pd.read_csv(path + "mirai_ack.csv")
mirai_udp = pd.read_csv(path + "mirai_udp.csv")
mirai_udpplain = pd.read_csv(path + "mirai_udpplain.csv")
gafgyt_junk = pd.read_csv(path + "gafgyt_junk.csv")
gafgyt_scan = pd.read_csv(path + "gafgyt_scan.csv")
gafgyt_tcp = pd.read_csv(path + "gafgyt_tcp.csv")
gafgyt_udp = pd.read_csv(path + "gafgyt_udp.csv")

gafgyt_list = [gafgyt_junk, gafgyt_scan, gafgyt_tcp, gafgyt_udp]
mirai_list = [mirai_scan, mirai_syn, mirai_ack, mirai_udp, mirai_udpplain]

malicious_gafgyt = pd.concat(gafgyt_list)
malicious_mirai = pd.concat(mirai_list)

malicious_mirai['class'] = "mirai"
malicious_gafgyt['class'] = "gafgyt"
benign['class'] = "benign"

print(malicious_mirai.shape)
print(malicious_gafgyt.shape)
print(benign.shape)

df = benign.append(malicious_gafgyt.sample(n=benign.shape[0], random_state=17)).append(malicious_mirai.sample(n=benign.shape[0], random_state=17))

def create_model(input_dim, hidden_layer_size, num_of_classes):
    model = Sequential()
    model.add(Dense(hidden_layer_size, activation="tanh", input_shape=(input_dim,)))
    model.add(Dense(hidden_layer_size, activation="tanh"))
    model.add(Dense(num_of_classes))
    model.add(Activation('softmax'))
    return model


x_train, x_test, y_train, y_test = train_test_split(df, pd.get_dummies(df['class']), test_size=0.2, random_state=42)

classes = ['benign', 'gafgyt', 'mirai']

scored = []
indices = {}
shps = {}
for cl in classes:
    indices[cl] = x_train['class'] == cl
    shps[cl] = x_train[indices[cl]].shape[0]

for col in x_train.columns:
    if col == 'class':
        continue
    num = 0
    den = 0
    m = x_train[col].mean()

    for cl in classes:
        num += (shps[cl] / x_train.shape[0]) * (m - x_train[indices[cl]][col].mean()) ** 2
        den += (shps[cl] / x_train.shape[0]) * x_train[indices[cl]][col].var()
    score = {'feature': col, 'score': num / den}
    scored.append(score)
    # print(score)
scored.sort(key=lambda x: x['score'], reverse=True)
scored[:3]

with open('classification_scores.csv', 'w+') as file:
    lines = ['Feature,Score\n']
    for s in scored:
        lines.append(s['feature'] + ',' + "{0:.2f}".format(s['score']) + '\n')
    file.writelines(lines)

    acs = []
    for top_n_features in [115]:
        fs = [it['feature'] for it in scored[:top_n_features]]
        X_train = x_train[fs]
        X_test = x_test[fs]
        scaler = StandardScaler()
        print('Transforming data')
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model = create_model(top_n_features, 8, 3)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        cp = ModelCheckpoint(filepath=f'./models/model_{top_n_features}.h5',
                             save_best_only=True,
                             verbose=0)
        es = EarlyStopping(patience=3, monitor='val_acc')
        epochs = 5
        start = time.time()
        history = model.fit(X_train, y_train,
                            epochs=epochs,
                            validation_split=0.2,
                            verbose=1,
                            callbacks=[cp, es])
        # model = load_model(f'./models/model_{top_n_features}.h5')
        print('time ' + str(time.time() - start))
        # print('Model evaluation')
        print('Loss, Accuracy')
        ev = model.evaluate(X_test, y_test)
        print(ev)
        # print()
        print()
        y_pred_proba = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred_proba, axis=1)
        y_test_classes = np.argmax(y_test.values, axis=1)
        print('accuracy')
        acc = accuracy_score(y_test_classes, y_pred_classes)
        acs.append(acc)
        print(acc)
        cnf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
        print('Confusion matrix')
        print('benign  gafgyt  mirai')
        print(cnf_matrix)

plt.xlabel('Features')
plt.ylabel('Accuracy')
plt.plot([2, 3, 4, 5, 6, 7, 8, 9, 10], acs)


epochs = len(history.history['acc'])
plt.plot(range(1,epochs+1), history.history['acc'])
plt.plot(range(1,epochs+1), history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.xticks(list(range(1,epochs+1)))
plt.savefig("RNN.png")

model.summary()
