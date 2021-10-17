# cs-6350-hw1
This is a machine learning library developed by Paul Wissler for CS 6350 in University of Utah

# Instructions
main.py can run most of this code, but bear in mind that many of the algorithms take a very long time to run, even with multiprocessing.

There are three modules of interest to the TA's: DecisionTree, EnsembleLearning, and LinearRegression. These are all packages, so all you need to do is import them like so:

```language[python]
import DecisionTree as dtree
import EnsembleLearning as ensl
import LinearRegression as lr

```

Within each of these packages are classes which will learn the appropriate model. For each class, when you instantiate the model, it will automatically create the model as an attribute in the `__init__` method, then will delete the supplied X (but not y for implementation purposes) to help conserve memory. To test, simply do:

```language[python]
tree = dtree.DecisionTreeModel(X.copy(), y.copy(), error_f=dtree.calc_entropy, 
            max_tree_depth=None, default_value_selection='subset_majority')
accuracy = tree.test(X_test, y_test)
```

There are optional kwargs you can pass in when you instantiate the model, these being error_f, max_tree_depth, and default_value_selection (the default values can be seen in the code block above). If you wish to change how information gain is calculated, you may use `dtree.calc_entropy`, `dtree.calc_majority_error`, or `dtree.calc_gini_index`. However, I do not think it will be necessary to change that kwarg. If you wish to set the maximum allowable tree depth, you may do so by changing the `max_tree_depth` kwarg to any value >0. Otherwise, it will go as deep as it can every single time. As for selecting default values, you may do so by changing the `default_value_selection` kwarg between 'subset_majority' and 'majority' (see the `default_value` method).

It should be noted that with all of my code, I generally assume that the user will input a pandas DataFrame for X and a pandas Series for y. To make sure there are no weird aliasing errors, please be sure to input `X.copy()` and `y.copy()`.

If the decision tree fails to find an output from any given input, it will return the mode of the training y.
