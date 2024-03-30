from nltk.corpus import stopwords
from string import punctuation 
import re

import nltk
nltk.download('stopwords')

def clean_punctuations(text):
    translator = str.maketrans('', '', punctuation)
    return text.translate(translator)

def cleaning_repeating_char(text):
    return re.sub(r'(.)1+', r'1', text)

def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)

#Pre-Processing the text 
def cleaning(text):

    # convert to lower case
    text = text.apply(lambda x: ' '.join(x.lower() for x in x.split()))

    # Removing stop words
    stop_words = stopwords.words('english')
    text = text.apply(lambda x: ' '.join(x for x in x.split() if x not in stop_words))

    # Removing punctuation
    text= text.apply(lambda x: clean_punctuations(x))

    # Removing digits
    text = text.apply(lambda x: cleaning_numbers(x))
    
    return text


# ------------------------------------------------------------------------------------
def process(df):
    print(f"There are total {len(df)} rows. ")
    df = df.dropna()
    print(f"After dropping missing values, there are {len(df)} rows remaining.")

    df.rename(columns={'Spam/Ham': 'Label'}, inplace=True)
    print(f"Among the remaining data, {df['Label'].value_counts()[0]} are spam, {df['Label'].value_counts()[1]} are ham email")

    df['Label'] = df['Label'].replace({'ham': 0, 'spam': 1})
    df = df[['Subject', 'Message', 'Label']]

    df['Subject'] = cleaning(df['Subject'])
    df['Message'] = cleaning(df['Message'])

    return df
