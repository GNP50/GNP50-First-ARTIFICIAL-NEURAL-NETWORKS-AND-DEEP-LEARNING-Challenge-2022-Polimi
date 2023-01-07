
# First-ARTIFICIAL-NEURAL-NETWORKS-AND-DEEP-LEARNING-Challenge-2022-Polimi

A simple Cnn experiments!
In this notebook is illustrated a way to create a Cnn to classify the photos of eigth different plant species.

To do that I wrote some useful functions and objects that make to me easier the code to read

We have:
e have:

1.  covnet module that have

-   a basic function to create a stratified cnn with a specified hidden layers
-   basic functions to create retrain a model specifying for how many epochs, if we want to merge classes and if we want to delete some of them.

2.  learningStages module that have learner objects that in general

-   prepare the dataset basing on the value of internal variable
-   train the model for 3 rounds where the following operation are performed
    -   choose for how many epochs the model should retrain based on the state of internal variables
    -   choose if the training should be done using a class-balanced dataset
    -   save the model and evaluate

3.  dataExpander and dataPrep module which contain the logic to augment the dataset based on a customizable pipeline composed of multi output stages

All attempts are described in further detail in the Jupiter Notebook and in the relation.pdf. 
