import pandas as pd
import classifier
import sklearn.linear_model as lm
import features.zip_codes
import evaluation


if __name__ == '__main__':
    data_train = pd.read_csv("data/zip.train", header = None, sep =" ")
    cleaned_train_data = data_train.dropna(axis=1, thresh=2)

    input_data = cleaned_train_data.iloc[:, 1:].values
    targets = cleaned_train_data[0].values

    input_data2 = features.zip_codes.multires(input_data)

    # log reg with simple feature set
    print("Evaluating simple feature set")
    log_reg = lm.SGDClassifier(n_jobs=1, loss="log", max_iter = 50)

    classifier.fit(log_reg, input_data, targets)
    pred, pred_proba = classifier.predict(log_reg, input_data)

    evaluation.print_errors(targets, pred)
    print("")

    # log reg with advanced feature set
    print("Evaluating modified feature set")
    log_reg2 = lm.SGDClassifier(n_jobs=1, loss="log", max_iter=50)

    classifier.fit(log_reg2, input_data2, targets)
    pred, pred_proba = classifier.predict(log_reg2, input_data2)

    evaluation.print_errors(targets, pred)

