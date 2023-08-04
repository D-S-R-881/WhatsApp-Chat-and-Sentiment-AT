import matplotlib.pyplot as plt
from wordcloud import WordCloud
from urlextract import URLExtract
import nltk
from nltk.corpus import stopwords
import pandas as pd
from collections import Counter
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sentiments = SentimentIntensityAnalyzer()

# nltk.download('stopwords')
# stop_words = stopwords.words('English')

f = open('hinglish_stopwords.txt', 'r')
stop_words = f.read()

extract = URLExtract()

def fetch_stats(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    # 1. Fetching number of messages
    number_of_messages = df.shape[0]

    # 2. Fetching total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # 3. Fetching total number of media messages
    number_of_media_messages = df[df['message'] == '<Media omitted>'].shape[0]

    # 4. Fetching number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return number_of_messages, len(words), number_of_media_messages, len(links)

def most_active_users(df):
    x = df['user'].value_counts().head()

    df = round(df['user'].value_counts()/df.shape[0] * 100, 2).reset_index().rename(columns = {'index':'Name', 'user':'Percentage'})
    return x, df

def creating_wordcloud(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']    # Removing Group Notification
    temp = temp[temp['message'] != '<Media omitted>']  # Removing media omitted

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500, height=500, min_font_size=12, background_color='black')

    temp['message'] = temp['message'].apply(remove_stop_words)

    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']    # Removing Group Notification
    temp = temp[temp['message'] != '<Media omitted>']  # Removing media omitted

    wrds = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                wrds.append(word)

    most_common_word_df = pd.DataFrame(Counter(wrds).most_common(20))
    
    return most_common_word_df

def most_used_emoji(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF\u2600-\u26FF\u2700-\u27BF]')  # Regular expression pattern to match emojis

    get_emojis = lambda message: emojis_pattern.findall(message)

    most_used_emoji_df = pd.DataFrame(Counter([emoji for msg in df['message'] for emoji in get_emojis(msg)]).most_common())

    most_used_emoji_df.rename(columns={0:'Emoji', 1:'Frequency'}, inplace = True)

    return most_used_emoji_df

def monthly_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    monthly_timeline_df = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(monthly_timeline_df.shape[0]):
        time.append(monthly_timeline_df['month'][i] + '-' + str(monthly_timeline_df['year'][i]))

    monthly_timeline_df['time'] = time

    return monthly_timeline_df

def daily_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline_df = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline_df

def weekly_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def monthly_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    activity_heatmap_df = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return activity_heatmap_df

def sentiment_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]


    temp = df[df['user'] != 'group_notification']    # Removing Group Notification
    temp = temp[temp['message'] != '<Media omitted>']  # Removing media omitted

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    temp['message'] = temp['message'].apply(remove_stop_words)
    

    temp['Positive'] = [sentiments.polarity_scores(i)['pos'] for i in temp['message']]
    temp['Negative'] = [sentiments.polarity_scores(i)['neg'] for i in temp['message']]
    temp['Neutral'] = [sentiments.polarity_scores(i)['neu'] for i in temp['message']]

    def sentiment(temp):
        if temp['Positive'] >= temp['Negative'] and temp['Positive'] >= temp['Neutral']:
            return 1
        if temp['Negative'] >= temp['Positive'] and temp['Negative'] >= temp['Neutral']:
            return -1
        if temp['Neutral'] >= temp['Positive'] and temp['Neutral'] >= temp['Negative']:
            return 0

    temp['value'] = temp.apply(lambda row: sentiment(row), axis=1)

    return temp