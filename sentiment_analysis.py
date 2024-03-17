"""
This program will read in a csv of Amazon reviews and demonstrate the use of spaCy
it will first store the sentiment and polarity of the reviews in a df
It will also demonstrate the sentiment and polarity functions on random sample of reviews
Finally it will ask the user for 2 reviews and come back with a similarity score
"""


print("\nThis program will get the sentiment for a sample set of Amazon reviews\n")
print("Starting program....")

print("Starting imports....")
# import
import pandas as pd
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

user_input = input("Default library is the medium. Would you like to use the small library instead? [y/n]: ")

if user_input.lower() == "y":
    print("Loading small library...")
    nlp = spacy.load('en_core_web_sm')
else:
    print("Loading medium library...")    
    nlp = spacy.load('en_core_web_md')


print("Finished imports....")

# Function to return the polarity and sentiment from text
def analyze_polarity(text):
    
    # Analyze sentiment with TextBlob
    blob = TextBlob(text)

    # select the polarity from the sentiment tuple
    polarity = blob.sentiment.polarity

    if polarity > 0:
        sentiment = "positive"
    elif polarity < 0:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return polarity, sentiment

# Function to process text, remove stop words, punctuation and non-ASCII characters
def preprocess(text):
    doc = nlp(text)
    
    # Filter out stop words, punctuation, and non-ASCII characters and put it in a list
    filtered_tokens = [token.text for token in doc if not token.is_stop
                       and not token.is_punct and all(ord(char) < 128 for char in token.text)]
    
    # Join the filtered tokens back into a single string
    filtered_text = ' '.join(filtered_tokens)
    
    return filtered_text

# Function to lemmatize text
def lemmatext(text):
    doc = nlp(text)
    return ' '.join([token.lemma_.lower() for token in doc])

print("Reading Amazon product reviews....")
# Read csv into a dataframe
df = pd.read_csv('Consumer_Reviews_of_Amazon_Products.csv', low_memory=False)

# focus on the columns reviews.text. Drop missing values
# from reviews.text
df = df[['reviews.text']].dropna(subset=['reviews.text'])

print("Applying preprocessing to clean data...")
# Add a new column that has removed stop words, punct and non-ascii characters
df['processed_reviews'] = df['reviews.text'].apply(preprocess)

# Lists to store polarity and sentiment
polarity_scores = []
sentiment_scores = []

# For each review in the dataframe get the polarity and sentiment and append it to the list
for review in df['processed_reviews']:
    score, sentiment = analyze_polarity(review)
    
    polarity_scores.append(score)
    sentiment_scores.append(sentiment)


# Create colummns for sentiment and polarity in the data frame
df['polarity'] = polarity_scores
df['sentiment'] = sentiment_scores

# Ask the user for a number of samples to test sentiment on
while True:
    sample_num = input("Please enter a sample number of reviews to test their sentiment: ")
    if sample_num.isdigit():
        sample_num = int(sample_num)
        break
    else:
        print("That is not a number.")

test_df = df.sample(sample_num)

# Print the results from calling the functions (note that this section is purely to test the functions
# The actual results are already in the df so could be returned from the df directly)
print(f"\n--------------- Sentiment Results ---------------\n")
for review in test_df['reviews.text']:
    processed_review = preprocess(review)
    score, sentiment = analyze_polarity(processed_review)
    print(f"Review: {review}\nPolarity score: {score}\nSentiment: {sentiment}\n")


# Test similarity
print(f"--------------------------------------------------\n")
# Ask the user for a number of a review to check similarity
print("\nWe will now take 2 reviews and check their similarity score.\n")
while True:
    review_1 = input(f"Please enter a review number to process [0 - {df['reviews.text'].count()-1}]: ")
    if review_1.isdigit() and int(review_1) >= 0 and int(review_1) <= (df['reviews.text'].count()-1) :
        review_1 = int(review_1)
        break
    else:
        print("Please try again.")   

# Ask the user for a second number of a review to check similarity
while True:
    review_2 = input(f"Please enter another review number to process [0 - {df['reviews.text'].count()-1}]: ")
    if review_2.isdigit() and int(review_2) >= 0 and int(review_2) <= (df['reviews.text'].count()-1) :
        review_2 = int(review_2)
        break
    else:
        print("Please try again.")

# Test similarity, and also call the function to lematize the preprocessed text first
query_doc = nlp(lemmatext(df['processed_reviews'][review_1]))
desc_doc = nlp(lemmatext(df['processed_reviews'][review_2]))
# Get a similarity score
similarity = query_doc.similarity(desc_doc)

# Print the similarity results
print(f"\n--------------- Similarity Results ---------------\n")
print(f"Review[{review_1}]: {df['reviews.text'][review_1]}")
print(f"\nReview[{review_2}]: {df['reviews.text'][review_2]}")
print(f"\nSimilarity score: {similarity}")
print(f"--------------------------------------------------\n")







