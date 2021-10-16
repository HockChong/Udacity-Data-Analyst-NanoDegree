# Investigate TMVB Movie Dataset

Data Source: https://www.kaggle.com/tmdb/tmdb-movie-metadata

The dataset comes from Udacity Data Analyst Nano Degree, originally from the kaggle. This data set contains information about 10,000 movies collected from The Movie Database (TMDb), including user ratings and revenue.
We interest to learn more about
1. Which genres are most popular from year to year?
2. What kinds of properties are associated with movies that have high revenues?
3. What are the top 10 most profitable movies?
4. Which director produces most movies?

Tool used: Python ( Pandas, Seaborn, Matplotlib )

# The data limitations of this dataset:

1. Almost every movie has multiple genres, and we have included all the genres during our calculation.
2. Lack of details on how the vote_count and vote_average calculation.
3. There are 5674 columns less than 0 in budget columns.
4. Do take note I use different data frame to answer different questions. I create a new data frame to answer Q1 & Q2. The remaining questions are used the original data frame.
5. The missing values in the data will affect our accuracy of analysis, if further investigation is needed, we can perform a statistical test to determine our result whether is statistically significant. The statistical test can help us determine whether our results in data are not determined by chance alone.

# Conclusions:
_For our analysis, we found that the movie genres with higher revenues come with higher popularity, higher budget, and higher net profit. We found the same findings when we look at the individual movies with higher revenues. We can conclude that production films are willing to invest more money in popular movies genres. Based on the data, the higher budget will allow firms to produce high-quality movies which generate higher revenue and net profit. As we can see from the top 10 most profitable movies, the return on investments is 4 times to 10.73 times on their budget._

Analysis:  [Click Here](https://medium.com/geekculture/investigate-tmvb-movie-dataset-28a27b1f3912)

References: [Pandas GroupBy](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html)

