Saccade Pattern Analysis
This repository contains Python code for analyzing saccade patterns using eye tracking data. The code uses the Pandas library for data manipulation and visualization. The goal of this project is to study how saccade patterns change as people complete a task or become more familiar with a scene, and to provide insights into how visual attention is directed in different conditions.

Usage
To use the code, you will need to have the following libraries installed:

numpy
matplotlib
pandas
itertools
You can install these libraries using pip by running the following command:

Copy code
pip install numpy matplotlib pandas itertools
The main function of the code is get_saccades(df, l=5), which takes as input a DataFrame df containing the eye tracking data, and an optional parameter l which is the value of lambda to use in the saccade detection algorithm. The function returns a copy of the input DataFrame with an additional column 'Saccade' indicating whether each sample corresponds to a saccade, and columns 'Speed in x coordinate' and 'Speed in y coordinate' containing the speed of the eye in each coordinate.

The function aggregate_fixations(df) takes as input the DataFrame returned by get_saccades and it will add a new column 'Saccade.event' indicating the start of a saccade and a fixation.

The code also includes example usage of the functions to load and process a sample data set.

Example
Copy code
import pandas as pd
from saccade_pattern_analysis import get_saccades, aggregate_fixations

# load data from a csv file
data = pd.read_csv("sample_data.csv")

# process data
processed_data = get_saccades(data)
aggregate_fixations(processed_data)

# visualize results
processed_data.plot(x="Time", y="Speed in x coordinate", style='.')
plt.show()
Contributing
This is a open-source project, contributions are welcomed and encouraged!

License
This project is released under the MIT License. See the LICENSE file for details.

Reference
The algorithm used in this code is based on the implementation described in https://github.com/tmalsburg/saccades
