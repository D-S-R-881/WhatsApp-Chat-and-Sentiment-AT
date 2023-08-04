import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns

st.markdown("<h1 style='text-align: center;'>WhatsApp Chat and Sentiment Analyzer</h1>", unsafe_allow_html=True)

header_placeholder = st.empty()
header_placeholder.header('Steps to follow :-')

write_placeholder = st.empty()
write_placeholder.write('1. To use WhatsApp Chat and Sentiment Analyzer. First Export the chat from your WhatsApp.\n\n2. Open a whatsapp conversation you wish to analyze and use the “Export Chat” functionality to send the entire conversation in text format to your email ID. \n\n3. Select the Export the chat without media option for faster analysis. \n\n4. Then upload that file in the sidebar and click on "Show Analysis" to see the analysis.')

st.sidebar.title('WhatsApp Chat and Sentiment Analyzer')

uploaded_file = st.sidebar.file_uploader('Choose a file')

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    # File is in bytes data
    # Now we will convert file into string data
    data = bytes_data.decode('utf-8')
    df = preprocessor.preprocess(data)

    # Fetching unique users
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, 'Overall')
    selected_user = st.sidebar.selectbox("Show Analysis wrt different Users", user_list)

    if st.sidebar.button('Show Analysis'):

        header_placeholder.empty()
        write_placeholder.empty()

        number_of_messages, words, number_of_media_messages, number_of_links = helper.fetch_stats(selected_user, df)
        st.title('Top Statistics')

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header('Total Messages')
            st.title(number_of_messages)

        with col2:
            st.header('Total Words')
            st.title(words)
        
        with col3:
            st.header('Media Shared')
            st.title(number_of_media_messages)

        with col4:
            st.header('Links Shared')
            st.title(number_of_links)

        # Daily and Monthly Analysis

        # Monthly
        st.title('Monthly Timeline')
        monthly_timeline_df = helper.monthly_analysis(selected_user, df)
        fig, ax = plt.subplots()

        ax.plot(monthly_timeline_df['time'], monthly_timeline_df['message'], color='red')
        plt.xticks(rotation = 90)
        plt.xlabel('Timeline')
        plt.ylabel('Number of Messages')
        st.pyplot(fig)

        # Daily
        st.title('Daily Timeline')
        daily_timeline_df = helper.daily_analysis(selected_user, df)
        fig, ax = plt.subplots()

        ax.plot(daily_timeline_df['only_date'], daily_timeline_df['message'], color='tomato')
        plt.xticks(rotation = 90)
        plt.xlabel('Timeline')
        plt.ylabel('Number of Messages')
        st.pyplot(fig)

        # Activity Maps

        st.title('Activity Maps')

        col1, col2 = st.columns(2)

        with col1:
            st.header('Most Busy Day')
            busy_day_df = helper.weekly_activity_map(selected_user, df)

            fig, ax = plt.subplots()
            ax.bar(busy_day_df.index, busy_day_df.values, color='cyan')
            plt.xticks(rotation = 90)
            plt.xlabel('Day')
            plt.ylabel('Number of Messages')
            st.pyplot(fig)

        with col2:
            st.header("Most Busy Month")
            busy_month_df = helper.monthly_activity_map(selected_user, df)

            fig, ax = plt.subplots()
            ax.bar(busy_month_df.index, busy_month_df.values, color='magenta')
            plt.xticks(rotation = 90)
            plt.xlabel('Month')
            plt.ylabel('Number of Messages')
            st.pyplot(fig)

        # Plotting Activity Heatmap

        st.title('Weekly Activity Map')
        st.write('The light color in the Heatmap indicates high activity of the users.')
        st.write('While the dark color indicates low activity of the users.')

        activity_heatmap_df = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()

        ax = sns.heatmap(activity_heatmap_df)
        plt.xlabel('Time')
        plt.ylabel('Day')
        st.pyplot(fig)

        # Finding the most active users in the group (Only for groups)

        if selected_user == 'Overall':
            st.title('Most Active Users')
            x, new_df = helper.most_active_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)
            with col1:
                ax.bar(x.index, x.values, color = 'turquoise')
                plt.xticks(rotation = 90)
                plt.xlabel('User Name')
                plt.ylabel('Number of Messages')
                st.pyplot(fig)

            with col2:
                st.dataframe(new_df)

        # Creating Wordcloud

        st.title('Word Cloud')
        df_wc = helper.creating_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # Most Common words

        most_common_word_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        
        ax.barh(most_common_word_df[0], most_common_word_df[1], color='fuchsia')
        plt.xticks(rotation = 90)

        st.title('Most Used Words (Top-20)')
        plt.ylabel('Kind of Word')
        plt.xlabel('Frequency')
        st.pyplot(fig)

        # Most Used Emojis
        most_used_emoji_df = helper.most_used_emoji(selected_user, df)
        st.title('Most Used Emojis')

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(most_used_emoji_df)

        with col2:
            fig, ax = plt.subplots()
            ax.pie(most_used_emoji_df['Frequency'].head(), labels = most_used_emoji_df['Emoji'].head(), autopct = "%0.2f")
            plt.title('Distribution of Top-5 Emojis')
            st.pyplot(fig)

        # Sentiment Analysis

        sentiment_df = helper.sentiment_analysis(selected_user, df)
        st.title('Sentiment Analysis')
        col1, col2 = st.columns(2)

        x = sum(sentiment_df['Positive'])
        y = sum(sentiment_df['Negative'])
        z = sum(sentiment_df['Neutral'])

        with col1:
            if (x>=y) and (x>=z):
                st.header('The Sentiment of the Chat is "Positive 🎉🥳"')
            if (y>=x) and (y>=z):
                st.header('The Sentiment of the Chat is "Negative 😞"')
            if (z>=y) and (z>=x):
                st.header('The Sentiment of the Chat is "Neutral 😊"')

        with col2:
            sentiments_count = sentiment_df[['Positive','Negative','Neutral']].sum()
            fig, ax = plt.subplots()
            ax.pie(sentiments_count, labels=sentiments_count.index, autopct="%0.2f")
            st.pyplot(fig)

        if selected_user == 'Overall':
            col1, col2, col3 = st.columns(3)

            def percentage(df,k):
                df = round((df['user'][df['value']==k].value_counts() / df[df['value']==k].shape[0]) * 100, 2).reset_index().rename(
                    columns={'index': 'Name', 'user': 'Percent'})
                return df
            
            with col1:
                st.header('Most Positive Contribution')
                a = percentage(sentiment_df, 1)
                st.dataframe(a)

            with col2:
                st.header('Most Neutral Contribution')
                b = percentage(sentiment_df, 0)
                st.dataframe(b)

            with col3:
                st.header('Most Negative Contribution')
                c = percentage(sentiment_df, -1)
                st.dataframe(c)