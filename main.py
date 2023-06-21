import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix

def predicct(path):
       data = pd.read_csv(path,encoding="Windows-1252")
       y = data['label'].values
       x = data[['mfcc_mean1', 'mfcc_mean2', 'mfcc_mean3', 'mfcc_mean4',
              'mfcc_mean5', 'mfcc_mean6', 'mfcc_mean7', 'mfcc_mean8', 'mfcc_mean9',
              'mfcc_mean10', 'mfcc_mean11', 'mfcc_mean12', 'mfcc_mean13',
              'mfcc_mean14', 'mfcc_mean15', 'mfcc_mean16', 'mfcc_mean17',
              'mfcc_mean18', 'mfcc_mean19', 'mfcc_mean20', 'mfcc_std1', 'mfcc_std2',
              'mfcc_std3', 'mfcc_std4', 'mfcc_std5', 'mfcc_std6', 'mfcc_std7',
              'mfcc_std8', 'mfcc_std9', 'mfcc_std10', 'mfcc_std11', 'mfcc_std12',
              'mfcc_std13', 'mfcc_std14', 'mfcc_std15', 'mfcc_std16', 'mfcc_std17',
              'mfcc_std18', 'mfcc_std19', 'mfcc_std20', 'cent_mean', 'cent_std',
              'cent_skew', 'rolloff_mean', 'rolloff_std']]

       x= preprocessing.StandardScaler().fit(x).transform(x)

       x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=17)

       print ('Train set:', x_train.shape, y_train.shape)
       print ('Test set:', x_test.shape, y_test.shape)

       clf = svm.SVC(kernel='rbf')   #  функция ядра - RBF (радиальная базисная функция)
       clf.fit(x_train, y_train)     # Обучение модели на тренировочном наборе
       yhat = clf.predict(x_test)    # для прогнозирования новых значений:

       print("Train set Accuracy: ", metrics.accuracy_score(y_train, clf.predict(x_train)))
       print("Test set Accuracy: ",metrics.accuracy_score(y_test, yhat))
       print('CONFUSION_MATRIX :\n')
       print(confusion_matrix(y_test,yhat))
       print('\n')
       print('REPORT :\n')
       print(classification_report(y_test,yhat))

predicct("")
