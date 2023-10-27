import streamlit as st
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from matplotlib import rcParams
from sklearn.linear_model import LogisticRegression
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from tqdm import tqdm

from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())

    
    st.markdown(
        f"""
        <style>
        
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
            background-size: cover;
            color: black;
        }}

        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
            color: black;
        }}

    </style>
        """,
        unsafe_allow_html=True
    )
add_bg_from_local('data/restaurant-bg.jpg')

st.write("""
# Restaurant Recommendation App

**This app predicts the best location for opening a restaurant and suggests best cuisines for the restaurant.**
""")

st.sidebar.header('User Inputs')

st.sidebar.markdown("""
[Example CSV input file](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])


def user_input_features():
    url = st.sidebar.text_input('Url','https://www.zomato.com/bangalore/', key='rest_url')
    address = st.sidebar.text_input('Address', 'Bangalore', key='address')
    name = st.sidebar.text_input('Name', 'Jalsa', key='name')
    online_order = st.sidebar.selectbox('Online Order',('Yes','No'), key='online_order')
    book_table = st.sidebar.selectbox('Book Table',('Yes','No'), key='book_table')
    rate = st.sidebar.number_input('Rate', min_value=0.0, max_value=5.0, value=5.0, step=0.1, key='rate')
    votes = st.sidebar.number_input("Votes", min_value=0, value=0, step=1, key='votes')
    phone = st.sidebar.text_input("Phone", "0000000000", max_chars=10, key='phone')
    rest_type = st.sidebar.text_input("Restaurant type", "Casual Dining", key='rest_type')
    dish_liked = st.sidebar.text_input("Dish liked", "Biryani", key='dish_liked')
    cuisines = st.sidebar.text_input("Cuisines", "North Indian", key='cuisines')
    approx_cost = st.sidebar.number_input('Cost for two', value=5.0, step=0.1, key='approx_cost')
    review =  st.sidebar.text_input("Review", "Good", key='review')
    reviews_list = [] 
    if review:
        reviews_list.append((f'Rated {rate}', review))
    menu_items = st.sidebar.text_input('Menu items(comma separated)', key='menu_items')
    menu_items = pd.Series([i.strip() for i in menu_items.split(',')])
    listed_in_type = st.sidebar.text_input("Listed in(type)", key='listed_in_type')
    listed_in_city = st.sidebar.text_input("listed_in(city)", key='listed_in_city')
    data = {
                'url': url,
                'address': address,
                'name': name,
                'online_order': online_order,
                'book_table': book_table,
                'rate': rate,
                'votes': votes,
                'phone': phone,
                'rest_type': rest_type,
                'dish_liked': dish_liked,
                'cuisines': cuisines,
                'approx_cost(for two people)': approx_cost,
                'reviews_list': reviews_list, 
                'menu_item': menu_items,
                'listed_in(type)': listed_in_type,
                'listed_in(city)': listed_in_city
            }
    features = pd.DataFrame(data, index=[0])
    return features

def parse_list(x):
    try:
        return ast.literal_eval(x)
    except:
        return None

# read the csv file 
zomato_raw=pd.read_csv('data/zomato.csv', on_bad_lines='skip', engine="python")

df = zomato_raw
input_df = user_input_features()



if st.sidebar.button('Predict'):
    st.write("""### **Restaurant Dataset** \n\n""")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, on_bad_lines='skip', engine="python")
    else:
        # make sure the inputs are entered
        if input_df['menu_item'] is not None and input_df['listed_in(type)'] is not None and input_df['listed_in(city)'] is not None:
            df = pd.concat([zomato_raw, input_df], axis=0, ignore_index=True)

    for lst in ('reviews_list', 'menu_item'):
        df[lst] = df[lst].apply(parse_list)

    st.write(df.tail())    

    data = df


    #### Cleaning begins #######
    # renaming columns
    data = data.rename(columns = {'listed_in(type)':'food_type', 'listed_in(city)':'city','approx_cost(for two people)':'cost for two'})

    # Report the percentage of missing values per column
    data_length = len(data)
    sum_null_values = data.isnull().sum()
    percent_missing = sum_null_values * 100 / data_length
    missing_value_data = pd.DataFrame({'column_name': data.columns,
                                    'percent_missing': percent_missing})

    # Dropping columns with > 50% null rows
    for index, row in missing_value_data.iterrows():
        if row['percent_missing'] > 50:
            del data[row['column_name']]

    # Handling invalid values with nan
    data.rate = data.rate.replace("NEW", np.nan).replace("-", np.nan).replace(' ', '')

    # Changing data type of columns
    data.rate = data.rate.str.split('/').str[0]
    data['rate'] = data.rate.astype(float)

    # Filling null values of rate with mean rating
    mean_value = data["rate"].mean() 
    data["rate"].fillna(mean_value,inplace=True)

    # Drop rows with Null values
    data.dropna(inplace = True)

    # Remove special characters
    data = data.replace(r'\r\n',' ', regex=True)
    data = data.replace(r'\n',' ', regex=True)

    # Remove duplicate rows
    data = data.loc[data.astype(str).drop_duplicates().index]

    # columns pre-processed from EDA 
    cols_to_use = ['rate', 'votes', 'cost for two']
    data[cols_to_use] = data[cols_to_use].astype(str)
    data[cols_to_use] = data[cols_to_use].apply(lambda x: pd.to_numeric(x.str.replace('[^\d.]', ''), errors='coerce'))
    data.dropna(inplace=True)

    # Pre-processing : Encoding of data
    def Encode(data):
        for column in data.columns[~data.columns.isin(['rate', 'cost for two', 'votes'])]:
            data[column] = data[column].astype(str).factorize()[0]
        return data

    zomato_en = Encode(data.copy())


    st.write(""" 
            ### **Visualizations** \n\n
            """)

    #----------------  EDA  -----------
    # Finding best restaurants in bengaluru
    import plotly.express as px
    def plot_top_restaurant_locations(data):
        """
        Plots the top 10 restaurants.

        Parameters: data (pandas.DataFrame) : The restaurant data with columns address and location.
        Returns: None
        """

        location_data = data.groupby(['address', 'location']).location.nunique().index.to_frame()
        top_locations = location_data['location'].value_counts()[:10]

        fig = px.bar(x=top_locations, 
                    y=top_locations.index, 
                    orientation='h', 
                    title='Unique restaurants per location',
                    labels={'x': 'Unique restaurants', 'y': 'Location'}, 
                    color=top_locations, 
                    color_continuous_scale='Blues'
                    )
        fig.update_layout(font=dict(size=14))

        st.plotly_chart(fig)

    plot_top_restaurant_locations(data)


    #Analysis of Restaurant Count Vs. Rating bu Delivery Type
    grouped_data = data.groupby(['rate', 'online_order']).size().reset_index(name='count')

    fig = px.scatter(grouped_data, x='rate', y='count', color='online_order', size='count',
                    hover_data=['count'], title='Restaurant Count vs. Rating by Delivery Type')
    fig.update_layout(xaxis_title='Rate', yaxis_title='Restaurant Count')

    st.plotly_chart(fig)


    #Relationship between Number of Votes and Rating
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data['votes'], data['rate'])
    ax.set_xlabel('Number of Votes')
    ax.set_ylabel('Rate')
    ax.set_title('Relationship between Number of Votes and Rating in Restaurants')

    # Display the plot in Streamlit
    st.pyplot(fig)


    votes_by_rating = data.groupby('rate')['votes'].sum()

    fig, ax = plt.subplots()
    ax.plot(votes_by_rating.index, votes_by_rating.values, 'bo-')
    ax.set_xlabel('Ratings')
    ax.set_ylabel('Total Votes')
    ax.set_title('Total Votes Grouped by Ratings')

    st.pyplot(fig)


    # Find the nearest cost value
    def find_nearest_cost(data, cost_for_two):
        idx = (np.abs(data['cost for two'] - cost_for_two)).idxmin()
        nearest_row = data.loc[idx]
        return nearest_row['cost for two']

    st.empty()
    st.write(' ')
    st.empty()

    st.write(""" 
            ### **Model Analysis** 
            """)


    # ----------- KNN   -------------------


    def recommend_location(df):
        X = df[['votes', 'cost for two']].values
        y = (df['rate'] >= 4.0).astype(int).values 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        max_votes = df['votes'].max()
        max_cost = df['cost for two'].max()
        possible_votes = np.linspace(0, max_votes, 101)
        possible_cost = np.linspace(0, max_cost, 101)
        possible_locations = [(votes, cost) for votes in possible_votes for cost in possible_cost]
        new_df = pd.DataFrame(possible_locations, columns=['votes', 'cost for two'])

        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        importances = result.importances_mean

        y_pred_prob = model.predict_proba(new_df[['votes', 'cost for two']].values)[:,1]

        # location with the highest predicted probability of success
        new_df['predicted probability'] = y_pred_prob
        best_location = new_df.loc[new_df['predicted probability'].idxmax()]
        feature_names = ['votes', 'cost for two']
        y_ticks = np.arange(0, len(feature_names))

        fig, ax = plt.subplots(figsize=(16, 6))
        ax.barh(y_ticks, importances, align='center')
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importances')

        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(new_df['votes'], new_df['cost for two'], c=new_df['predicted probability'])
        ax.set_xlabel('Votes')
        ax.set_ylabel('Cost for two')
        ax.set_title('Success Probability of Restaurants')
        ax.scatter(best_location[0], best_location[1], c='red', marker='X')
        ax.set_xlim(0, 5000)
        ax.set_ylim(0, 10000)
        cbar = fig.colorbar(sc)

        st.pyplot(fig)
        
        return (f"Model accuracy: {accuracy}", best_location['votes'], best_location['cost for two'], best_location['predicted probability'])

    (acc, votes,cost_for_2,predicted_rating) = recommend_location(zomato_en)

    def text_box(text):
        image_file = 'data/wood-bg.jpeg'
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
            f"""
            <div style="
                background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
                background-repeat: repeat;
                border-radius: 10px;
                padding: 10px;
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                color: white;
            "/>
            <style>
            .stApp h4 {{
                color: white;
            }}
            </style> 
            {text}
            """,
            unsafe_allow_html=True
        )

    val = data[data['cost for two'] == find_nearest_cost(data,cost_for_2)]['city'].unique()[0]
    text_box(f"""
            #### KNN Model Results:\n
            Cost for two people: {cost_for_2}\n\n
            Predicted Probability of success: {predicted_rating}\n\n
            {acc} \n\n
            Recommended location for opening a new restaurant: {val}
            """)
