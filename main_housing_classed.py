
import utils
import features.housing
import features.heart
import classifier.tree_regression
from sklearn.model_selection import train_test_split
import pandas as pd


if __name__ == '__main__':
    housing_data = utils.read_file('csv', 'data/housing.csv')
    housing_data = features.housing.set_labels(housing_data, 8)


# Rename labels
    housing_data = features.housing.set_labels(housing_data,'median_housing_value')

# Fill in missing data
    housing_data['total_bedrooms'] = features.housing.fill_w_mean(housing_data['total_bedrooms'])

# Encode categorical data
    housing_data = features.heart.encode_feature(housing_data['ocean_proximity'],'ocean_proximity',
                                    {0: 'NEAR BAY', 1: '<1H OCEAN', 2: 'INLAND', 3: 'NEAR OCEAN',4: 'ISLAND'},
                                                  housing_data)

# Sometime I need to handle the capped values in a better way
# For now: remove capped values from housing_data
    housing_data = housing_data.drop(housing_data[(housing_data['Labels'] > 500000) == True].index)
    housing_data = housing_data.drop(housing_data[(housing_data['housing_median_age'] > 51) == True].index)


# Split test and train
    data_train, data_other = train_test_split(housing_data, test_size=0.20, random_state=11)
    data_test, data_validation = train_test_split(data_other, test_size=0.25, random_state=11)



# Rename labels, initialize train and test result dataframes
    X_train = features.housing.remove_labels(data_train)
    y_train = features.housing.get_labels(data_train)

    X_test = features.housing.remove_labels(data_test)
    y_test = features.housing.get_labels(data_test)

    X_validation = features.housing.remove_labels(data_validation)
    y_validation = features.housing.get_labels(data_validation)

    results_train = pd.DataFrame(y_train)
    results_test = pd.DataFrame(y_test)



# Decision Tree Regression: initialize, fit, predict, evaluate
# Using input features: Latitude and Longitude
# Training data
    features.housing.decision_tree(5, X_train[['latitude', 'longitude']],
                                   y_train, X_train[['latitude', 'longitude']], y_train, results_train)

    features.housing.decision_tree(10, X_train[['latitude', 'longitude']],
                                  y_train, X_train[['latitude', 'longitude']], y_train, results_train)

    features.housing.decision_tree(15, X_train[['latitude', 'longitude']],
                                   y_train, X_train[['latitude', 'longitude']], y_train, results_train)

    features.housing.decision_tree(20,X_train[['latitude','longitude']],
                                   y_train,X_train[['latitude','longitude']],y_train,results_train)


# Looks a bit OVERFIT to me

# Test data

    features.housing.decision_tree(15, X_train[['latitude', 'longitude']],
                                   y_train, X_test[['latitude', 'longitude']], y_test, results_test)



# Using other input features: Latitude and income
# Training data

    features.housing.decision_tree(6, X_train[['latitude', 'median_income']],
                                   y_train, X_train[['latitude', 'median_income']], y_train, results_train)

    features.housing.decision_tree(16, X_train[['latitude', 'median_income']],
                                  y_train, X_train[['latitude', 'median_income']], y_train, results_train)


    #print (results_train)
    #print(results_test)

# Oh yeah! But depth of 20 still seems to be too good to be true

# Evaluation using Cross-Validation

    #tree_clf = (features.housing.decision_tree_clf(16, X_train[['latitude', 'median_income']],
    #                              y_train, X_train[['latitude', 'median_income']], y_train))
    #evaluation.display_scores(tree_clf,X_train[['latitude', 'median_income']],y_train)
#
#
#
    #tree_clf2 = (features.housing.decision_tree_clf(10, X_train[['latitude', 'median_income']],
    #                                               y_train, X_train[['latitude', 'median_income']], y_train))
    #evaluation.display_scores(tree_clf2, X_train[['latitude', 'median_income']], y_train)
#
#
    #tree_clf3 = (features.housing.decision_tree_clf(5, X_train[['latitude', 'median_income']],
    #                                                y_train, X_train[['latitude', 'median_income']], y_train))
    #evaluation.display_scores(tree_clf3, X_train[['latitude', 'median_income']], y_train)

# So, with a smaller depth, the scores improve. What does this say about the model?



    lat_income_train = classifier.tree_regression.Tree(data_train[['latitude','median_income']],
                            data_train[['Labels']],
                            data_train[['latitude','median_income']],
                            data_train[['Labels']])

    lat_income_test =  classifier.tree_regression.Tree(data_train[['latitude','median_income']],
                            data_train[['Labels']],
                            data_test[['latitude','median_income']],
                            data_test[['Labels']])

    loc_income_train = classifier.tree_regression.Tree(data_train[['latitude','longitude','median_income']],
                            data_train[['Labels']],
                            data_train[['latitude','longitude','median_income']],
                            data_train[['Labels']])

    loc_income_test =  classifier.tree_regression.Tree(data_train[['latitude','longitude','median_income']],
                            data_train[['Labels']],
                            data_test[['latitude','longitude','median_income']],
                            data_test[['Labels']])



    all_train = classifier.tree_regression.Tree(data_train.drop('Labels',axis=1),
                            data_train[['Labels']],
                            data_train.drop('Labels',axis=1),
                            data_train[['Labels']])

    all_test = classifier.tree_regression.Tree(data_train.drop('Labels',axis=1),
                            data_train[['Labels']],
                            data_test.drop('Labels',axis=1),
                            data_test[['Labels']])

# Plotting Errors
    classifier.tree_regression.plot_errors(all_train,all_test)
    classifier.tree_regression.plot_errors(lat_income_train,lat_income_test)
    classifier.tree_regression.plot_errors(loc_income_train, loc_income_test)