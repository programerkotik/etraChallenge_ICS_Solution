# Saccade Pattern Analysis
This repository contains Python code for analyzing saccade patterns using eye tracking data. The code uses the Pandas library for data manipulation and visualization. The goal of this project is to study saccade patterns and how they change as people complete a task or become more familiar with a scene, and to provide insights into how visual attention is directed in different conditions.

## Usage
To use the code, you will need to have the following libraries installed:

- numpy
- matplotlib
- pandas
- itertools

You can install these libraries using pip by running the following command:

`pip install numpy matplotlib pandas itertools` 

The main function of the code is get_saccades(df, l=5), which takes as input a DataFrame df containing the eye tracking data, and an optional parameter $\lambda$ which is the value of lambda to use in the saccade detection algorithm. The function returns a copy of the input DataFrame with an additional column 'Saccade' indicating whether each sample corresponds to a saccade, and columns 'Speed in x coordinate' and 'Speed in y coordinate' containing the speed of the eye in each coordinate.

The function `aggregate_fixations(df)` takes as input the DataFrame returned by get_saccades and it will add a new column `'Saccade.event'` indicating the start of a saccade and a fixation.

The code also includes example usage of the functions to load and process a sample data set.
The notebook utilises the following code to analyse eye tracking data from [etraChallenge_ICS](https://etra.acm.org/2019/challenge.html). The following analysis is part of solution for the exam from [Informatics and Cognitive Science](http://csng.mff.cuni.cz/ikv1.html) from Faculty of Math and Physics of Charles University in Prague. 
## Reference
The algorithm used in this code is based on the implementation described in https://github.com/tmalsburg/saccades
