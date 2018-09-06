import pandas as pd
import classifier
import features.heart
import evaluation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


if __name__ == '__main__':
    data_train = pd.read_csv("data/Heart.csv")

    # Categorize target values
    data_train['AHD'] = 1.0 * (data_train['AHD'] == 'Yes')

    # Remove all rows with NaN values
    cleaned_train_data = data_train.dropna(axis=0, how='any')


    # Encode all categorical features

    prepared_data = (features.heart.encode_feature(cleaned_train_data['Thal'],
                                                   'Thal',{0: 'Fixed', 1: 'Normal', 2: 'Reversable'},
                                                   cleaned_train_data))

    prepared_data = (features.heart.encode_feature(prepared_data['ChestPain'],'ChestPain',
                                                   {0: 'typical', 1: 'asymptomatic', 2: 'nonanginal', 3: 'nontypical'},
                                                   prepared_data))

    prepared_data = (features.heart.encode_feature(prepared_data['Sex'], 'Sex',
                                                   {0: 'male', 1: 'female'},
                                                   prepared_data))

    prepared_data = (features.heart.encode_feature(prepared_data['Fbs'], 'Fbs',
                                                   {0: 'Fbs_true', 1: 'Fbs_false'},
                                                   prepared_data))

    prepared_data = (features.heart.encode_feature(prepared_data['ExAng'], 'ExAng',
                                                   {0: 'ExAng_no', 1: 'ExAng_yes'},
                                                   prepared_data))

    prepared_data = (features.heart.encode_feature(prepared_data['RestECG'], 'RestECG',
                                                   {0: 'RestECG_abnormal', 1: 'RestECG_normal',2:'RestECG_damage'},
                                                   prepared_data))

    prepared_data = (features.heart.encode_feature(prepared_data['Slope'], 'Slope',
                                                   {0: 'downsloping', 1: 'flat', 2: 'upsloping'},
                                                   prepared_data))


    # Test/Train Split (1/3)
    data_train, data_test = train_test_split(prepared_data, test_size=0.33, random_state=11,
                                             stratify=prepared_data['AHD'])

    # Remove labels, remove instance id/name, rename test and train sets
    y_train = data_train['AHD']
    X_train = data_train.drop(columns=['AHD','Unnamed: 0'])
    y_test = data_test['AHD']
    X_test = data_test.drop(columns=['AHD','Unnamed: 0'])


    # Logistic Regression with all features
    log_reg = LogisticRegression()


    classifier.fit(log_reg, X_train, y_train)
    pred, pred_proba = classifier.predict(log_reg,X_test)

    evaluation.print_errors(y_test, pred)
    print("")


    # Pimp out the features with some polynomial transforming
    for i in range(0, len(X_train)):
        X_train_new = features.heart.feature_polynomial(X_train.iloc[i, 0], X_train.iloc[i, 2],
                                      'Age^2', 'Chol^2', 'Age*Chol',X_train)

    # This doesn't really work, I get a bunch of NaN values, perhaps with the list declaration
    for i in range(0, len(X_test)):
       X_test_new = features.heart.feature_polynomial(X_test.iloc[i, 0], X_test.iloc[i, 2],
                                                   'Age^2', 'Chol^2', 'Age*Chol',X_test)



    # Log reg with advanced feature set (testing on the training set for now)
    log_regr = LogisticRegression()

    classifier.fit(log_regr, X_train_new, y_train)
    pred, pred_proba = classifier.predict(log_regr, X_train_new)

    evaluation.print_errors(y_train, pred)
    print("")




