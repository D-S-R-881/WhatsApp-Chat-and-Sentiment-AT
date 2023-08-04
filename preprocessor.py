import re
import pandas as pd

def preprocess(data):
    # Define a regex pattern to extract the desired information
    pattern = r'(.+?) - (.+)'

    # Use the findall method to get all occurrences of the pattern
    matches = re.findall(pattern, data)

    # Create a DataFrame from the extracted matches
    df = pd.DataFrame(matches, columns=['date', 'Message'])

    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y, %I:%M %p', errors= 'coerce')

    users = []
    msgs = []
    for message in df['Message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:  # user name
            users.append(entry[1])
            msgs.append(" ".join(entry[2:]))
        else:
            users.append('group_notification')
            msgs.append(entry[0])

    df['user'] = users
    df['message'] = msgs
    df.drop(columns=['Message'], inplace=True)

    df = df.dropna(subset='date')

    df['only_date'] = df['date'].dt.date

    df['year'] = df['date'].dt.year

    df['month_num'] = df['date'].dt.month

    df['month'] = df['date'].dt.month_name()

    df['day'] = df['date'].dt.day

    df['day_name'] = df['date'].dt.day_name()

    df['hour'] = df['date'].dt.hour

    df['minute'] = df['date'].dt.minute

    period = []

    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 00:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))
    
    df['period'] = period

    return df