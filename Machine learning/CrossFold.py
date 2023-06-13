from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import numpy as np
def CrossFold(subject, model, all_data_df, data = 'False'):
    global train_df
    global X_train_t
    global X_test_t
    global y_test, y_train

    accuracy_score = []
    f1_score = []
    mcc_score = []
    cross_fold = deque(subject)
    for _ in range(len(subject)):
        print("Check")   
        
        train_df = all_data_df.loc[all_data_df['Subject']['Subject'].isin(list(cross_fold)[0:len(subject)-1])]
        test_df = all_data_df.loc[all_data_df['Subject']['Subject'] == cross_fold[-1]]

        if(data == 'False'):
            x_train = train_df.to_numpy()[0:,1:-1].astype('float')
            y_train = train_df.to_numpy()[0:,-1].astype('int')
            y_train = y_train.flatten()

            x_test = test_df.to_numpy()[0:,1:-1].astype('float')
            y_test = test_df.to_numpy()[0:,-1].astype('int')
        else:
            train_df = train_df.loc[:,['Subject', 'Gender', data, 'Out']]
            test_df = test_df.loc[:,['Subject', 'Gender', data, 'Out']]

            x_train = train_df.to_numpy()[1:,1:-1].astype('float')
            y_train = train_df.to_numpy()[1:,-1].astype('int')
            y_train = y_train.flatten()
            x_test = test_df.to_numpy()[1:,1:-1].astype('float')
            y_test = test_df.to_numpy()[1:,-1].astype('int')


        sc = StandardScaler()
        print(x_train)
        X_train_t = sc.fit_transform(x_train)
        X_test_t = sc.transform(x_test)

        model.fit(X_train_t, y_train)
        y_pred = model.predict(X_test_t)

        accuracy = metrics.accuracy_score(y_test, y_pred)
        f1  = metrics.f1_score(y_test, y_pred)
        mcc = metrics.matthews_corrcoef(y_test, y_pred)

        accuracy_score.append(np.mean(accuracy))
        f1_score.append(np.mean(f1))
        mcc_score.append(np.mean(mcc))
        cross_fold.rotate(-1)

    return accuracy_score, f1_score, mcc_score