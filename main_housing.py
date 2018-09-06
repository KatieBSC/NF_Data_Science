
import utils
import evaluation
import features.housing
import features.heart
import classifier.tree_regression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


from sklearn.metrics import mean_squared_error
from sklearn.tree import _tree
from sklearn.tree._tree import TREE_LEAF
import sys




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


# NEW TREE FOR PRUNING:
    my_tree = DecisionTreeRegressor(max_depth=2)
    my_tree.fit(data_train[['median_income']], data_train[['Labels']])
    n_nodes = my_tree.tree_.node_count

# NEW TREE USING CLASS
#    my_tree_obj = classifier.tree_regression.Tree(data_train[['median_income']], data_train[['Labels']],
#                                                  data_validation[['median_income']], data_validation[['Labels']])
#
#    my_tree = my_tree_obj.tree_clf(2)
#    my_tree_obj.fit_tree(2)
#    print(my_tree)
#
# DataFrame that might not be necessary
    tree_frame = pd.DataFrame(my_tree.apply(data_train[['median_income']]),columns=['Node'])
    actual_value = pd.DataFrame(data_train[['Labels']])
    actual_value.reset_index(drop=True, inplace=True)
    pred = my_tree.predict(data_train[['median_income']])
    tree_frame['Predict'] = pred
    tree_frame = pd.concat([tree_frame, actual_value], axis=1)
    print(tree_frame.head())
    print(pd.value_counts(tree_frame.iloc[:, 0]))



#    df = tree_frame.copy()
#    df_list = []
#    def node_df(df, df_copy,num):
#        for i in range(0, len(df)):
#            if df.iloc[i, 0] == num:
#                df_copy['Diff',i] = (df.iloc[i,2]-df.iloc[i,3])**2
#            else :
#                df_copy['Diff',i] = 'Omit'
#            return df_copy

    #print(my_tree.tree_.impurity)
    #print(my_tree.tree_.n_node_samples)
    print (classifier.tree_regression.tree_info_df(my_tree))

    print (classifier.tree_regression._calc_impurity(my_tree,0))


# TESTING OUT THIS CODE!!!!
    print ('To check: ')
    print(classifier.tree_regression.determine_alpha(my_tree))












