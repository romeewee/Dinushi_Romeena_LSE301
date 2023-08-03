#!/usr/bin/env python
# coding: utf-8

# ### LSE Data Analytics Online Career Accelerator 
# 
# # DA301:  Advanced Analytics for Organisational Impact

# ## Assignment template

# ### Scenario
# You are a data analyst working for Turtle Games, a game manufacturer and retailer. They manufacture and sell their own products, along with sourcing and selling products manufactured by other companies. Their product range includes books, board games, video games and toys. They have a global customer base and have a business objective of improving overall sales performance by utilising customer trends. In particular, Turtle Games wants to understand: 
# - how customers accumulate loyalty points (Week 1)
# - how useful are remuneration and spending scores data (Week 2)
# - can social data (e.g. customer reviews) be used in marketing campaigns (Week 3)
# - what is the impact on sales per product (Week 4)
# - the reliability of the data (e.g. normal distribution, Skewness, Kurtosis) (Week 5)
# - if there is any possible relationship(s) in sales between North America, Europe, and global sales (Week 6).

# # Week 1 assignment: Linear regression using Python
# The marketing department of Turtle Games prefers Python for data analysis. As you are fluent in Python, they asked you to assist with data analysis of social media data. The marketing department wants to better understand how users accumulate loyalty points. Therefore, you need to investigate the possible relationships between the loyalty points, age, remuneration, and spending scores. Note that you will use this data set in future modules as well and it is, therefore, strongly encouraged to first clean the data as per provided guidelines and then save a copy of the clean data for future use.
# 
# ## Instructions
# 1. Load and explore the data.
#     1. Create a new DataFrame (e.g. reviews).
#     2. Sense-check the DataFrame.
#     3. Determine if there are any missing values in the DataFrame.
#     4. Create a summary of the descriptive statistics.
# 2. Remove redundant columns (`language` and `platform`).
# 3. Change column headings to names that are easier to reference (e.g. `renumeration` and `spending_score`).
# 4. Save a copy of the clean DataFrame as a CSV file. Import the file to sense-check.
# 5. Use linear regression and the `statsmodels` functions to evaluate possible linear relationships between loyalty points and age/renumeration/spending scores to determine whether these can be used to predict the loyalty points.
#     1. Specify the independent and dependent variables.
#     2. Create the OLS model.
#     3. Extract the estimated parameters, standard errors, and predicted values.
#     4. Generate the regression table based on the X coefficient and constant values.
#     5. Plot the linear regression and add a regression line.
# 6. Include your insights and observations.

# ## 1. Load and explore the data

# In[563]:


# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm 
from statsmodels.formula.api import ols


# In[565]:


# Load the CSV file(s) as reviews.
reviews = pd.read_csv('turtle_reviews.csv')

# View the DataFrame.
reviews.head()


# In[566]:


# Any missing values?
# Check for missing values
print(reviews.isnull().sum())


# In[567]:


# Explore the data.
print(reviews.columns)  #Column names
print(reviews.shape)    #Number of rows and columnsprint
print(reviews.dtypes)   #Data types


# In[568]:


print(reviews.info())   #Determine the metadata of the data set


# In[569]:


# Descriptive statistics.
print(reviews.describe())


# ## 2. Drop columns

# In[570]:


# Drop unnecessary columns.
reviews = reviews.drop(['language', 'platform'], axis=1)

print(reviews.shape)    #Number of rows and columnsprint
# View column names.
print(reviews.columns)  #Column names


# ## 3. Rename columns

# In[571]:


# Rename the column headers.
reviews = reviews.rename(columns={'remuneration (k£)': 'renumeration', 'spending_score (1-100)': 'spending'})

# View column names.
print(reviews.columns)  #Column names


# ## 4. Save the DataFrame as a CSV file

# In[572]:


# Create a CSV file as output.
# Save the clean DataFrame as a CSV file called clean_reviews
reviews.to_csv('clean_reviews.csv')


# In[573]:


# Import new CSV file with Pandas.
clean_reviews = pd.read_csv('clean_reviews.csv')

# View DataFrame.
print(clean_reviews.shape)    #Number of rows and columnsprint
print(clean_reviews.info())   #Determine the metadata of the data set


# ## 5. Linear regression

# ### 5a) spending vs loyalty

# In[574]:


# Independent variable.
X = clean_reviews["loyalty_points"]

# Dependent variable.
y = clean_reviews["spending"]

# OLS model and summary.
# Run the OLS test.
f = 'y ~ X'
test = ols(f, data = clean_reviews).fit()

# View the output.
test.summary()


# In[575]:


# Extract the estimated parameters.
print("Parameters:", test.params)

# Extract the standard errors.
print("Standard errors:", test.bse)

# Extract the predicted values.
print("Predict values:", test.predict)


# In[576]:


# Set the X coefficient and the constant to generate the regression table.
# x coef: 0.0137.
# Constant coef: 28.4260.
# Create the linear equation.
y_pred = 28.4260 + 0.013671 * X

# View the output.
y_pred


# In[577]:


# Plot the graph with a regression line.
# Plot the data points.
plt.scatter(X, y)

# Plot the line.
plt.plot(X, y_pred, color='black')

#Label
plt.xlabel('Loyalty')
plt.ylabel('Spending')
plt.title('Loyalty vs Spending')

# View the plot
plt.show()


# ### 5b) renumeration vs loyalty

# In[578]:


# Independent variable.
X = clean_reviews["loyalty_points"]

# Dependent variable.
y = clean_reviews["renumeration"] 

# OLS model and summary.
# Run the OLS test.
f = 'y ~ X'
test = ols(f, data = clean_reviews).fit()

# View the output.
test.summary()


# In[579]:


# Extract the estimated parameters.
print("Parameters:", test.params)

# Extract the standard errors.
print("Standard errors:", test.bse)

# Extract the predicted values.
print("Predict values:", test.predict)


# In[580]:


# Set the X coefficient and the constant to generate the regression table.
# x coef: 0.011101.
# Constant coef: 30.560555.
# Create the linear equation.
y_pred = 30.560555 + 0.011101 * X

# View the output.
y_pred


# In[581]:


# Plot the graph with a regression line.
# Plot the data points.
plt.scatter(X, y)

# Plot the line.
plt.plot(X, y_pred, color='black')

#Label
plt.xlabel('Loyalty')
plt.ylabel('Renumeration')
plt.title('Loyalty vs Renumeration')

# View the plot
plt.show()


# ### 5c) age vs loyalty

# In[582]:


# Independent variable.
X = clean_reviews["loyalty_points"]

# Dependent variable.
y = clean_reviews["age"] 

# OLS model and summary.
# Run the OLS test.
f = 'y ~ X'
test = ols(f, data = clean_reviews).fit()

# View the output.
test.summary()


# In[583]:


# Extract the estimated parameters.
print("Parameters:", test.params)

# Extract the standard errors.
print("Standard errors:", test.bse)

# Extract the predicted values.
print("Predict values:", test.predict)


# In[584]:


# Set the X coefficient and the constant to generate the regression table.
# x coef: -0.000449.
# Constant coef: 40.203457.
# Create the linear equation.
y_pred = 40.203457 + (-0.000449) * X

# View the output.
y_pred


# In[585]:


# Plot the graph with a regression line.
# Plot the data points.
plt.scatter(X, y)

# Plot the line.
plt.plot(X, y_pred, color='black')

#Label
plt.xlabel('Age')
plt.ylabel('Loyalty')
plt.title('Loyaly vs Age')

# View the plot
plt.show()


# ## 6. Observations and insights

# ***Your observations here...***
# 
# 
# 
# 
# 

# The initial data exploration and cleaning process ensured the dataset's quality and consistency.Descriptive statistics provided an overview of variable distributions and central tendencies.The removal of redundant columns streamlined the analysis and enhanced clarity.Linear regression analysis will help determine if age, remuneration, and spending scores can predict loyalty points.The OLS model's parameter estimates and standard errors will aid in understanding variable significance.The regression table and scatter plots will visualize the relationships and provide insights into their strengths and directions.

# # 

# # Week 2 assignment: Clustering with *k*-means using Python
# 
# The marketing department also wants to better understand the usefulness of renumeration and spending scores but do not know where to begin. You are tasked to identify groups within the customer base that can be used to target specific market segments. Use *k*-means clustering to identify the optimal number of clusters and then apply and plot the data using the created segments.
# 
# ## Instructions
# 1. Prepare the data for clustering. 
#     1. Import the CSV file you have prepared in Week 1.
#     2. Create a new DataFrame (e.g. `df2`) containing the `renumeration` and `spending_score` columns.
#     3. Explore the new DataFrame. 
# 2. Plot the renumeration versus spending score.
#     1. Create a scatterplot.
#     2. Create a pairplot.
# 3. Use the Silhouette and Elbow methods to determine the optimal number of clusters for *k*-means clustering.
#     1. Plot both methods and explain how you determine the number of clusters to use.
#     2. Add titles and legends to the plot.
# 4. Evaluate the usefulness of at least three values for *k* based on insights from the Elbow and Silhoutte methods.
#     1. Plot the predicted *k*-means.
#     2. Explain which value might give you the best clustering.
# 5. Fit a final model using your selected value for *k*.
#     1. Justify your selection and comment on the respective cluster sizes of your final solution.
#     2. Check the number of observations per predicted class.
# 6. Plot the clusters and interpret the model.

# ## 1. Load and explore the data

# In[587]:


# Import necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings('ignore')


# In[588]:


# Load the CSV file(s) as df2 from the CSV file created in the previous sector
df2 = pd.read_csv('clean_reviews.csv')

# View DataFrame.
df2.head()
print(df2.shape)    #Number of rows and columnsprint
print(df2.info())   #Determine the metadata of the data set


# In[589]:


# Drop unnecessary columns.
# Keep the new df with renumeration and spending only
df2_clean = df2[['renumeration', 'spending']]

# View the DataFrame.
print(df2_clean.shape)    #Number of rows and columnsprint
print(df2_clean.info())   #Determine the metadata of the data set


# In[590]:


# Explore the data.
print(df2_clean.columns)  #Column names
print(df2_clean.dtypes)   #Data types


# In[591]:


# Descriptive statistics.
print(df2_clean.describe())


# ## 2. Plot

# In[592]:


# Create a scatterplot with Seaborn.
sns.scatterplot(x='renumeration',
                y='spending',
                data=df2_clean)

#Labels
plt.xlabel('Remuneration (Salary)')
plt.ylabel('Spending Score')
plt.title('Remuneration vs. Spending Score')

#View data
plt.show()


# In[593]:


# Create a pairplot with Seaborn.
x = df2_clean[['renumeration', 'spending']]

sns.pairplot(df2_clean,
             vars=x,
            diag_kind='kde')

plt.suptitle('Pairplot of Remuneration and Spending Score', y=1.02)

#View data
plt.show()


# ## 3. Elbow and silhoutte methods

# In[594]:


# Determine the number of clusters: Elbow method.

# Import the KMeans class.
from sklearn.cluster import KMeans 

# Elbow chart for us to decide on the number of optimal clusters.
ss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i,
                    init = 'k-means++', 
                    max_iter = 500,
                    n_init = 10,
                    random_state = 42)
    kmeans.fit(x)
    ss.append(kmeans.inertia_)

plt.plot(range(1, 11),
         ss,
         marker='o')

#Titles
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("SS")

#View data
plt.show()


# In[595]:


# Determine the number of clusters: Silhouette method.
# Import silhouette_score class from sklearn.
from sklearn.metrics import silhouette_score

# Find the range of clusters to be used using silhouette method.
sil = []
kmax = 10

for k in range(2, kmax+1):
    kmeans_s = KMeans(n_clusters = k).fit(x)
    labels = kmeans_s.labels_
    sil.append(silhouette_score(x,
                                labels,
                                metric = 'euclidean'))

# Plot the silhouette method.
plt.plot(range(2, kmax+1),
         sil,
         marker='o')

# Insert labels and title
plt.title("The Silhouette Method")
plt.xlabel("Number of clusters")
plt.ylabel("Sil")

#View data
plt.show()


# ## 4. Evaluate k-means model at different values of *k*

# In[54]:


# Use 3 clusters:
kmeans = KMeans(n_clusters = 3,
                max_iter = 500,
                init='k-means++',
                random_state=42).fit(x)

clusters = kmeans.labels_
x['K-Means Predicted'] = clusters

# Plot the predicted.
sns.pairplot(x,
             hue='K-Means Predicted',
             diag_kind= 'kde')


# In[55]:


# Check the number of observations per predicted class with 3 clusters.
x['K-Means Predicted'].value_counts()


# In[56]:


# View the K-Means predicted with 3 clusters.
print(x.head())


# In[57]:


#Create a copy of the file to explore graphics with different clusters
df2_clusters=df2_clean.copy()


# In[58]:


# Standardize the data before clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df2_clusters)

# Evaluate the k-means model for different values of k
k_values = [3, 4, 5]

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    df2_clusters['cluster'] = kmeans.fit_predict(scaled_data)

    # Plot the clusters for each value of k
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='renumeration', y='spending', data=df2_clusters, hue='cluster', palette='viridis', s=50)
    plt.xlabel('Remuneration')
    plt.ylabel('Spending')
    plt.title(f'K-means Clustering with k={k}')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


# In[59]:


# Use 5 clusters:
kmeans = KMeans(n_clusters = 5,
                max_iter = 500,
                init='k-means++',
                random_state=42).fit(x)

clusters = kmeans.labels_
x['K-Means Predicted'] = clusters


# In[60]:


# Check the number of observations per predicted class.
x['K-Means Predicted'].value_counts()


# ## 5. Fit final model and justify your choice

# In[61]:


# Apply the final model.
# View the K-Means predicted.
print(x.head())


# In[62]:


# Apply the final model.
# Visualising the clusters.
# Set plot size.
sns.set(rc = {'figure.figsize':(12, 8)})

sns.scatterplot(x='spending' , 
                y ='renumeration',
                data=x ,
                hue='K-Means Predicted',
                palette=['red', 'green', 'blue', 'black', 'orange'])


# In[63]:


# Check the number of observations per predicted class.

# Fit the k-means clustering with k=5
kmeans = KMeans(n_clusters=5, random_state=42)
df2_clusters['cluster'] = kmeans.fit_predict(scaled_data)

# Check the number of observations per predicted class
cluster_counts = df2_clusters['cluster'].value_counts()
print(cluster_counts)

# Evaluate the cluster sizes as a percentage of total observations
cluster_sizes = cluster_counts.values / len(df2_clusters) * 100
print(cluster_sizes)


# ## 6. Plot and interpret the clusters

# In[65]:


# View the DataFrame
print(df2_clusters.head())

# Plot the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='renumeration', y='spending', data=df2_clusters, hue='cluster', palette='viridis', s=50)
plt.xlabel('Remuneration')
plt.ylabel('Spending')
plt.title('K-means Clustering - Remuneration vs. Spending Score')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# ## 7. Discuss: Insights and observations
# 
# ***Your observations here...***

# Overall, this approach guides you through the entire process of data analysis, from data preparation and exploration to clustering and interpretation of results. The use of k-means clustering allows to identify distinct customer segments based on remuneration and spending scores, which can help the marketing department target specific market segments more effectively. The insights gained from this analysis can inform marketing strategies and decision-making processes.

# # 

# # Week 3 assignment: NLP using Python
# Customer reviews were downloaded from the website of Turtle Games. This data will be used to steer the marketing department on how to approach future campaigns. Therefore, the marketing department asked you to identify the 15 most common words used in online product reviews. They also want to have a list of the top 20 positive and negative reviews received from the website. Therefore, you need to apply NLP on the data set.
# 
# ## Instructions
# 1. Load and explore the data. 
#     1. Sense-check the DataFrame.
#     2. You only need to retain the `review` and `summary` columns.
#     3. Determine if there are any missing values.
# 2. Prepare the data for NLP
#     1. Change to lower case and join the elements in each of the columns respectively (`review` and `summary`).
#     2. Replace punctuation in each of the columns respectively (`review` and `summary`).
#     3. Drop duplicates in both columns (`review` and `summary`).
# 3. Tokenise and create wordclouds for the respective columns (separately).
#     1. Create a copy of the DataFrame.
#     2. Apply tokenisation on both columns.
#     3. Create and plot a wordcloud image.
# 4. Frequency distribution and polarity.
#     1. Create frequency distribution.
#     2. Remove alphanumeric characters and stopwords.
#     3. Create wordcloud without stopwords.
#     4. Identify 15 most common words and polarity.
# 5. Review polarity and sentiment.
#     1. Plot histograms of polarity (use 15 bins) for both columns.
#     2. Review the sentiment scores for the respective columns.
# 6. Identify and print the top 20 positive and negative reviews and summaries respectively.
# 7. Include your insights and observations.

# ## 1. Load and explore the data

# In[374]:


# Import all the necessary packages.
import pandas as pd
import numpy as np
import nltk 
import os 
import matplotlib.pyplot as plt

# nltk.download ('punkt').
# nltk.download ('stopwords').

from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from textblob import TextBlob
from scipy.stats import norm

# Import Counter.
from collections import Counter

import warnings
warnings.filterwarnings('ignore')


# In[375]:


# Load the data set as df3.
# Create a copy of df2
df3=df2.copy()

# View DataFrame.
df3.head()
print(df3.shape)    #Number of rows and columnsprint
print(df3.info())   #Determine the metadata of the data set


# In[376]:


# Drop unnecessary columns.
# Keep the new df with review and summary only
df3_clean = df3[['review', 'summary']]

# View the DataFrame.
print(df3_clean.shape)    #Number of rows and columnsprint
print(df3_clean.info())   #Determine the metadata of the data set


# In[377]:


# Explore the data.
print(df3_clean.columns)  #Column names
print(df3_clean.dtypes)   #Data types


# In[378]:


# Determine if there are any missing values.
# Check for missing values
print(df3_clean.isnull().sum())


# ## 2. Prepare the data for NLP
# ### 2a) Change to lower case and join the elements in each of the columns respectively (review and summary)

# In[379]:


# Review: Change all to lower case and join with a space.

# Transform data to lowercase.
df3_clean['review'] = df3_clean['review'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# Preview the result.
df3_clean['review'].head()


# In[380]:


# Summary: Change all to lower case and join with a space.

# Transform data to lowercase.
df3_clean['summary'] = df3_clean['summary'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# Preview the result.
df3_clean['summary'].head()


# ### 2b) Replace punctuation in each of the columns respectively (review and summary)

# In[381]:


# Replace all the punctuations in review column.
# Remove punctuation.
df3_clean['review'] = df3_clean['review'].str.replace('[^\w\s]','')

# View output.
df3_clean['review'].head()


# In[382]:


# Replace all the puncuations in summary column.
# Remove punctuation.
df3_clean['summary'] = df3_clean['summary'].str.replace('[^\w\s]','')

# View output.
df3_clean['summary'].head()


# ### 2c) Drop duplicates in both columns

# In[383]:


df3_clean.head()


# In[384]:


df3_clean.shape


# In[385]:


# Check the number of duplicate values in the COMMENTS column.
df3_clean.review.duplicated().sum()
df3_clean.summary.duplicated().sum()


# In[386]:


# Drop duplicates.
df3d = df3_clean.drop_duplicates(subset=['review'])
df3d = df3_clean.drop_duplicates(subset=['summary'])

# Preview data.
df3d.reset_index(inplace=True)
df3d.head()


# In[387]:


df3d.shape


# ## 3. Tokenise and create wordclouds

# In[389]:


# Create new DataFrame (copy DataFrame).
df3t = df3d.copy()

# View DataFrame.
df3t.head()
print(df3t.shape)    #Number of rows and columnsprint
print(df3t.info())   #Determine the metadata of the data set


# In[390]:


# Tokenise the words.
df3t['tokens_review'] = df3t['review'].apply(word_tokenize)

# Preview data.
df3t['tokens_review'].head()


# In[391]:


# Tokenise the words.
df3t['tokens_summary'] = df3t['summary'].apply(word_tokenize)

# Preview data.
df3t['tokens_summary'].head()


# In[397]:


df3t.head()


# In[399]:


# Create a word cloud for 'review' column.
wordcloud_review = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df3t['tokens_review'].sum()))

# Plot the WordCloud image for 'review' column.
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_review, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Reviews')
plt.show()


# In[400]:


# Create a word cloud for 'summary' column.
wordcloud_summary = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df3t['tokens_summary'].sum()))

# Plot the WordCloud image for 'summary' column.
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_summary, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Summaries')
plt.show()


# ## 4. Frequency distribution and polarity
# ### 4a) Create frequency distribution

# In[413]:


# Define an empty list of tokens.
all_tokens_review = []

for i in range(df3t.shape[0]):
    # Add each token to the list.
    all_tokens_review = all_tokens_review + df3t['tokens_review'][i]


# In[414]:


# Define an empty list of tokens.
all_tokens_summary = []

for i in range(df3t.shape[0]):
    # Add each token to the list.
    all_tokens_summary = all_tokens_summary + df3t['tokens_summary'][i]


# In[416]:


all_tokens=all_tokens_review+all_tokens_summary


# In[417]:


# Calculate the frequency distribution.
fdist = FreqDist(all_tokens)

# Preview data.
fdist


# ### 4b) Remove alphanumeric characters and stopwords

# In[418]:


# Delete all the alpanum.
# Filter out tokens that are neither alphabets nor numbers (to eliminate punctuation marks, etc.).
all_tokens = [word for word in all_tokens if word.isalnum()]


# In[419]:


# Remove all the stopwords
# Create a set of English stopwords.
english_stopwords = set(stopwords.words('english'))

# Create a filtered list of tokens without stopwords.
tokens2 = [x for x in all_tokens if x.lower() not in english_stopwords]

# Define an empty string variable.
tokens2_string = ''

for value in tokens:
    # Add each filtered token word to the string.
    tokens2_string = tokens2_string + value + ' '


# ### 4c) Create wordcloud without stopwords

# In[421]:


# Create a wordcloud without stop words.
wordcloud = WordCloud(width = 1600, height = 900, 
                background_color ='white', 
                colormap='plasma', 
                min_font_size = 10).generate(tokens2_string) 


# In[423]:


# Plot the WordCloud image.                        
plt.figure(figsize = (10, 5), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis('off') 
plt.tight_layout(pad = 0) 
plt.show()

# Note that your word cloud might differ slightly to the one provided.


# ### 4d) Identify 15 most common words and polarity

# In[434]:


# Determine the 15 most common words.

# Generate a DataFrame from Counter.
counts = pd.DataFrame(Counter(tokens2).most_common(15),
                      columns=['Word', 'Frequency']).set_index('Word')

# Preview data.
counts


# In[487]:


# Set the plot type.
ax = counts.plot(kind='barh', figsize=(10, 5), fontsize=12,
                 colormap ='plasma')

# Set the labels.
ax.set_xlabel('Count', fontsize=12)
ax.set_ylabel('Word', fontsize=12)
ax.set_title("Safety survey responses: Count of the 15 most frequent words",
             fontsize=20)

# Draw the bar labels.
for i in ax.patches:
    ax.text(i.get_width()+.41, i.get_y()+.1, str(round((i.get_width()), 2)),
            fontsize=12, color='red')


# ## 5. Review polarity and sentiment: Plot histograms of polarity (use 15 bins) and sentiment scores for the respective columns.

# In[531]:


# Import the necessary package
from textblob import TextBlob


# In[532]:


# Provided function.
def generate_polarity(comment):
    '''Extract polarity score (-1 to +1) for each comment'''
    return TextBlob(comment).sentiment[0]


# In[533]:


# Create new DataFrame (copy DataFrame).
df4=df3d.copy()

# View DataFrame.
df4


# In[534]:


# Populate a new column with polarity scores for each comment.
df4['review_polarity'] = df4['review'].apply(generate_polarity)

# Populate a new column with polarity scores for each comment.
df4['summary_polarity'] = df4['summary'].apply(generate_polarity)

# Preview the result.
df4.head()


# In[535]:


# Provided function.
def generate_subjectivity(comment):
    '''Extract polarity score (-1 to +1) for each comment'''
    return TextBlob(comment).sentiment[1]


# In[536]:


# Populate a new column with subjectivity scores for each comment.
df4['review_subjectivity'] = df4['review'].apply(generate_subjectivity)

# Populate a new column with subjectivity scores for each comment.
df4['summary_subjectivity'] = df4['summary'].apply(generate_subjectivity)

# Preview the result.
df4.head()


# In[537]:


# Review: Create a histogram plot with bins = 15.
# Histogram of polarity

# Set the number of bins.
num_bins = 15

# Set the plot area.
plt.figure(figsize=(10,5))

# Define the bars.
n, bins, patches = plt.hist(df4['review_polarity'], num_bins, facecolor='red', alpha=0.6)

# Set the labels.
plt.xlabel('Polarity', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Histogram of review_polarity', fontsize=20)
          
# Histogram of review_polarity
plt.show()


# In[538]:


# Review: Create a histogram plot with bins = 15.
# Histogram of subjectivity

# Set the number of bins.
num_bins = 15

# Set the plot area.
plt.figure(figsize=(10,5))

# Define the bars.
n, bins, patches = plt.hist(df4['review_subjectivity'], num_bins, facecolor='blue', alpha=0.6)

# Set the labels.
plt.xlabel('Subjectivity', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Histogram of sentiment score review_subjectivity', fontsize=20)
          
# Histogram of review sentiment score
plt.show()


# In[540]:


# Summary: Create a histogram plot with bins = 15.
# Histogram of polarity

# Set the number of bins.
num_bins = 15

# Set the plot area.
plt.figure(figsize=(10,5))

# Define the bars.
n, bins, patches = plt.hist(df4['summary_polarity'], num_bins, facecolor='red', alpha=0.6)

# Set the labels.
plt.xlabel('Polarity', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Histogram of summary_polarity', fontsize=20)
          
# Histogram of summary_polarity
plt.show()


# In[541]:


# Summary: Create a histogram plot with bins = 15.
# Histogram of subjectivity

# Set the number of bins.
num_bins = 15

# Set the plot area.
plt.figure(figsize=(10,5))

# Define the bars.
n, bins, patches = plt.hist(df4['summary_subjectivity'], num_bins, facecolor='blue', alpha=0.6)

# Set the labels.
plt.xlabel('Subjectivity', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Histogram of sentiment score summary_subjectivity', fontsize=20)
          
# Histogram of summary sentiment score
plt.show()


# ## 6. Identify top 20 positive and negative reviews and summaries respectively

# In[542]:


# Top 20 negative reviews.
# Create a DataFrame.
top_negative_reviews = df4.nsmallest(20, 'review_polarity')['review']

# View output.
print('Top 20 Negative Reviews:')
print(top_negative_reviews)


# In[543]:


# Top 20 negative summaries.
# Create a DataFrame.
top_negative_summaries = df4.nsmallest(20, 'summary_polarity')['summary']

# View output.
print('Top 20 Negative Summaries:')
print(top_negative_summaries)


# In[545]:


# Top 20 positive reviews.
top_positive_reviews = df4.nlargest(20, 'review_polarity')['review']

# View output.
print('Top 20 Positive Reviews:')
print(top_positive_reviews)


# In[547]:


# Top 20 positive summaries.
top_positive_summaries = df4.nlargest(20, 'summary_polarity')['summary']

# View output.
print('Top 20 Positive Summaries:')
print(top_positive_summaries)


# ## 7. Discuss: Insights and observations
# 
# ***Your observations here...***

# In summary, this approach involves extracting valuable insights from customer reviews using NLP techniques. By analyzing sentiment, identifying commonly used words, and exploring positive/negative reviews, the marketing department can gain a better understanding of customer perceptions and make informed decisions for future campaigns. Further exploration could involve advanced sentiment analysis methods, topic modeling, or sentiment changes over different product releases.

# # 
