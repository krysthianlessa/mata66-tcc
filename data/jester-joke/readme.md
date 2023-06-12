## Jester Datasets for Recommender Systems and Collaborative Filtering Research
<br />
"
Over 100,000 new ratings from 7,699 total users: data collected 
- from April 2015 - Nov 2019
The text of the jokes: jester_dataset_4_joke_texts.zip (30KB)

**Format:**
Includes 8 new jokes 151-158.
An excel sheet with 158 rows.
The row number corresponds to the joke ID referred to in the Excel files below
The first 150 jokes and their ID's are consistent with the jokes from earlier datasets
The Ratings Data: Save to disk, then unzip: jester_dataset_4.zip (1.4MB)

**Format:**

The data is formatted as an excel file representing a 7699 by 159 matrix with rows as users and columns as jokes. The left-most column represents the amount of jokes rated by each user. There are a total of 7699 users and 158 jokes in this dataset.

22 of the jokes don't have ratings, their ids are: {1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 20, 27, 31, 43, 51, 52, 61, 73, 80, 100, 116}.

8 jokes were added to this version {151-158}
Each rating is from (-10.00 to +10.00) and 99 corresponds to a null rating (user did not rate that joke).

Note that the ratings are real values ranging from -10.00 to +10.00. The jokes {1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 20, 27, 31, 43, 51, 52, 61, 73, 80, 100, 116} have been removed (i.e. they are never displayed or rated). As of April 2015, 8 jokes were added.
"
Informations found at https://eigentaste.berkeley.edu/dataset/

Explore more: https://www.kdnuggets.com/2016/02/nine-datasets-investigating-recommender-systems.html


## Update the datasets:

- Download "Dataset 1: 4.1 million ratings", "Dataset 3: 2.3 million ratings" or other which have millions of ratings
- Umpack the two files.
- Export as CSV and name the file with the ratings in matrix shape as "ratings_matrix.csv"
- Export as CSV and name the file with the details in matrix shape as "descriptions_matrix.csv"
- Put it on data/jester-joke