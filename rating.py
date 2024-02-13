###################################################
# Rating Products
###################################################

# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating

#Bu bölümde amaç : Bir ürüne verilen puanlar üzerinden çeşitli değerlendirmeler yaparak
# en doğru puanın nasıl hesaplanabileceğine dair bir uygulama yapmak

############################################
# Application : User and Time Weighted Course Score Calculation
############################################

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# (50+ Hours) Python A-Z™: Data Science and Machine Learning
# Score: 4.8 (4.764925)
# Total Points: 4611
# Score Percentages: 75, 20, 4, 1, <1
# Approximate Numerical Correspondents: 3458, 922, 184, 46, 6


df = pd.read_csv(r"datasets\course_reviews.csv")
df.head()
df.shape

# rating distribution
df["Rating"].value_counts()
# distribution of questions asked
df["Questions Asked"].value_counts() 

# what is the score given in the question breakdown of questions
df.groupby("Questions Asked").agg({"Questions Asked": "count", "Rating": "mean"})


df.head()

####################
# Average
####################


df["Rating"].mean()

# We may be missing the satisfaction trend when averaged directly.
# What can we do to better reflect the current trend in the average?

####################
# Time-Based Weighted Average
####################


df.head()
df.info()

# We need to express the comments made in days, we need current date = maximum date in the dataset
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
current_date = pd.to_datetime('2021-02-10 0:0:0')
df["days"] = (current_date - df["Timestamp"]).dt.days

df.loc[df["days"] <= 30, "Rating"].mean()   # Average of total points in 30 days and below
df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean()  # Average of total scores between 30 and 90 days
df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() # Average of total points between 90 and 180 days
df.loc[(df["days"] > 180), "Rating"].mean()   # Average of total points greater than #180 days

# Looking at the results, it can be seen that there has been a recent increase in satisfaction with this course
# The effect of time can be reflected in the weight calculation by giving different weights for the results obtained


# 26 weighting given to part 2 because it is less important in terms of time than the first part  

# The thing to note here is that the sum of all given weights should be 100.

df.loc[df["days"] <= 30, "Rating"].mean() * 28/100 + \
    df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 26/100 + \
    df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 24/100 + \
    df.loc[(df["days"] > 180), "Rating"].mean() * 22/100



def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["days"] <= 30, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["days"] > 180), "Rating"].mean() * w4 / 100

time_based_weighted_average(df)
time_based_weighted_average(df, 30, 26, 22, 22)



####################
# User-Based Weighted Average
####################

# Should all users' scores have the same weight?
# so for example, the person who watches the whole course and the person who watches 1% of the course should have the same weight ?

df.head()

# Can be grouped according to progress and averaged
df.groupby("Progress").agg({"Rating": "mean"})
# From this table, it looks like there is an increase in progress.


df.loc[df["Progress"] <= 10, "Rating"].mean() * 22 / 100 + \
    df.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * 24 / 100 + \
    df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * 26 / 100 + \
    df.loc[(df["Progress"] > 75), "Rating"].mean() * 28 / 100

# The person who follows the course completely has recognized the course and the weight of the points given by this person should not be the same as the weight of the points given by others.
# This part is not a comment but a common practice in the sector.

def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return dataframe.loc[dataframe["Progress"] <= 10, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 10) & (dataframe["Progress"] <= 45), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 45) & (dataframe["Progress"] <= 75), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 75), "Rating"].mean() * w4 / 100


user_based_weighted_average(df, 20, 24, 26, 30)


####################
# Weighted Rating
####################

def course_weighted_rating(dataframe, time_w=50, user_w=50):
    return time_based_weighted_average(dataframe) * time_w/100 + user_based_weighted_average(dataframe)*user_w/100

course_weighted_rating(df)

course_weighted_rating(df, time_w=40, user_w=60)

# score was calculated with a predefined 50% weighting on time_base and a predefined 50% weighting on user-based.

