import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, precision_score, \
    recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


def load_data():
    """
    This method will load UCI data
    :return: benign, malicious
    """
    path = "dataset/"
    benign = pd.read_csv(path + "benign.csv")

    mirai_scan = pd.read_csv(path + "mirai_scan.csv").head(19472)
    mirai_syn = pd.read_csv(path + "mirai_syn.csv").head(19471)
    mirai_ack = pd.read_csv(path + "mirai_ack.csv").head(19471)
    mirai_udp = pd.read_csv(path + "mirai_udp.csv").head(19471)
    mirai_udpplain = pd.read_csv(path + "mirai_udpplain.csv").head(19471)
    gafgyt_junk = pd.read_csv(path + "gafgyt_junk.csv").head(19471)
    gafgyt_scan = pd.read_csv(path + "gafgyt_scan.csv").head(19471)
    gafgyt_tcp = pd.read_csv(path + "gafgyt_tcp.csv").head(19471)
    gafgyt_udp = pd.read_csv(path + "gafgyt_udp.csv").head(19471)

    malicious_gafgyt_list = [gafgyt_junk, gafgyt_scan, gafgyt_tcp, gafgyt_udp]
    malicious_mirai_list = [mirai_scan, mirai_syn, mirai_ack, mirai_udp, mirai_udpplain]
    malicious_gafgyt_concat = pd.concat(malicious_gafgyt_list)
    malicious_mirai_concat = pd.concat(malicious_mirai_list)

    malicious_mirai_concat['Detection'] = "mirai"
    malicious_gafgyt_concat['Detection'] = "gafgyt"
    benign['Detection'] = "benign"

    combine_data = pd.concat([benign, malicious_mirai_concat, malicious_gafgyt_concat], axis=0)
    combine_data = shuffle(combine_data)

    return combine_data


def preprocess_data(comb_data):
    """
    This method will preprocess the data. Both benign and malicious data will be combined
    Label for Malicious data is marked as 1 and for benign data it is marked as 0
    :return: X_train, X_test, y_train, y_test
    """

    labels = comb_data.iloc[:, -1]
    labels = np.array(labels).flatten()

    no_labels_data = comb_data.iloc[:, :28]

    # Standardizing the data
    scale = StandardScaler()
    scale.fit(no_labels_data)
    scale.transform(no_labels_data)

    # Performing train test split
    X_train, X_test, y_train, y_test = train_test_split(no_labels_data, labels, test_size=0.25, shuffle=True)
    return X_train, X_test, y_train, y_test


def create_and_train_model():
    """
    This method will create and train the model
    :return: trained model
    """
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    return knn_classifier.fit(X_train, np.ravel(y_train))


def predict_model():
    """
    This method will predict the model
    :return: predicted value
    """
    return knn_model.predict(X_test)


def evaluate_metrics():
    acc = accuracy_score(y_test, knn_predictions)
    prec_score = precision_score(y_test, knn_predictions, average='weighted')
    r_score = recall_score(y_test, knn_predictions, average='weighted')
    fscore = f1_score(y_test, knn_predictions, average='weighted')
    print("accuracy score of KNN: %.5f" % acc)
    print("precision score of KNN: %.5f" % prec_score)
    print("recall score of KNN: %.5f" % r_score)
    print("f score of KNN: %.5f" % fscore)
    print("Confusion Matrix of KNN:\n", confusion_matrix(y_test, knn_predictions))
    print("Classification Report of KNN:\n",
          classification_report(y_test, knn_predictions, zero_division=0))


def plot_observation():
    """
    This method will plot observation captured by fitting the model
    :return: nothing
    """
    y_test_predict = y_test[:200]
    knn_predict = knn_predictions[:200]
    plt.xlabel('X(Time->)')
    plt.ylabel('0 for Benign Traffic(LOW) and 1 for Malicious Traffic(HIGH)')
    plt.plot(y_test_predict, c='g', label="Benign data")
    plt.plot(knn_predict, c='b', label="Malicious data")
    plt.legend(loc='upper left')
    plt.savefig('KNN.png')

    classes = np.unique(y_test)
    fig, ax = plt.subplots(figsize=(5, 3))
    cm = metrics.confusion_matrix(y_test, knn_predictions, labels=classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
    ax.set(xlabel="Pred", ylabel="True", title="Confusion matrix for KNN")
    ax.set_yticklabels(labels=classes, rotation=0)
    plt.savefig('confusion_KNN.png')


if __name__ == '__main__':
    """
    Main Method : Execution starts here
    # """
    comb_data = load_data()

    X_train, X_test, y_train, y_test = preprocess_data(comb_data)

    # create and train KNN
    knn_model = create_and_train_model()

    # predict model
    knn_predictions = predict_model()

    # evaluate and print metrics
    evaluate_metrics()

    # plot observation
    plot_observation()
