import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

"""## Import Dataset"""

df = pd.read_csv('D:\PROJECT\Capstone Project\GoLap\Dataset\Golap_Dataset_Final.csv')

df.head()

"""## Exploratory Data Analysis"""

df.shape

df.info()

df.describe()

df.isna().sum()

df.duplicated().sum()

"""## Data Preprocessing

#### Changing currency features

###### change currency from india rupee to indonesian rupiah
"""

df.rename(columns={'Price_in_Rupee': 'Price_in_IDR'}, inplace=True)

exchange_rate_rupee_to_rupiah = 188.37

df['Price_in_IDR'] = df['Price_in_IDR'] * exchange_rate_rupee_to_rupiah

df['Price_in_IDR']

"""#### Untokenized the tokenized features

###### untokenized laptop type features
"""

laptop_type_mapping = {
    1: 'Gaming Laptop',
    2: 'Thin and Light Laptop',
    3: '2 in 1 Laptop',
    4: 'Notebook',
    5: 'Normal Laptop',
    6: '2 in 1 Gaming Laptop',
    7: 'Business Laptop',
    8: 'Chromebook',
    9: 'Creator Laptop'
}

df['Laptop_Type'] = df['Laptop_Type'].map(laptop_type_mapping)

df['Laptop_Type'].unique()

"""###### untokenized processor brand features"""

processor_brand_mapping = {
    1: 'Intel',
    2: 'AMD',
    3: 'Qualcomm',
    4: 'Apple',
    5: 'Mediatek'
}

df['Processor_Brand'] = df['Processor_Brand'].map(processor_brand_mapping)

df['Processor_Brand'].unique()

"""###### untokenized storage type features"""

storage_type_mapping = {
    0: 'HDD',
    1: 'SSD'
}

df['Storage_Type'] = df['Storage_Type'].map(storage_type_mapping)

df['Storage_Type'].unique()

"""###### untokenized memory type features"""

memory_type_mapping = {
    0: 'Non Expandable',
    1: 'Expandable'
}

df['Memory_Type'] = df['Memory_Type'].map(memory_type_mapping)

df['Memory_Type'].unique()

"""###### untokenized operating system features"""

os_mapping = {
    1: 'Windows',
    2: 'Chrome OS',
    3: 'Dos',
    4: 'Mac',
    5: 'Ubuntu'
}

df['OS'] = df['OS'].map(os_mapping)

df['OS'].unique()

"""###### untokenized touchsreen features"""

touchscreen_mapping = {
    0: 'Non Touchscreen',
    1: 'Touchsreen'
}

df['Touchscreen_Features'] = df['Touchscreen_Features'].map(touchscreen_mapping)

df['Touchscreen_Features'].unique()

"""###### untokenized laptop company features"""

company_mapping = {
    1: "Asus",
    2: "HP",
    3: "Lenovo",
    4: "Dell",
    5: "MSI",
    6: "Realme",
    7: "Avita",
    8: "Acer",
    9: "Samsung",
    10: "Infinix",
    11: "LG",
    12: "Apple",
    13: "Nokia",
    14: "Redmibook",
    15: "MI",
    16: "VAIO"
}

df['Laptop_Company'] = df['Laptop_Company'].map(company_mapping)

df['Laptop_Company'].unique()

"""#### - Handling NaN Value

###### user rating features
"""

df['User_Rating'] = df['User_Rating'].fillna(0)

df['User_Rating'].isna().sum()

"""#### Removing Unnecessary Words

###### laptop name features for laptop description
"""

df['Laptop_Name'] = df['Laptop_Name'].str.replace(r'-.*$', '', regex=True)
df['Laptop_Name']

def cleaning_name(row):
    keywords = row['Processor'].split()
    pattern = r'\b(' + '|'.join(keywords) + r')\b'
    cleaned_name = re.sub(pattern, '', row['Laptop_Name'], flags=re.IGNORECASE).strip()
    cleaned_name = re.sub(r'\s+', ' ', cleaned_name)
    return cleaned_name

df['Laptop_Name'] = df.apply(cleaning_name, axis=1)
df['Laptop_Name']

"""## Data Visualization"""

def pie_plot(features):
    col_counts = features.value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(col_counts, labels=col_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    plt.title(f'Distribution of {features.name}')
    plt.show()

def scatter_plot(features):
    plt.figure(figsize=(15, 6))
    plt.scatter(features.index, features.values)
    plt.xlabel('Index')
    plt.ylabel(features.name)
    plt.title(f'Scatter plot of {features.name}')
    plt.show()

def histogram_plot(features, bins=10):
    plt.figure(figsize=(15, 6))
    plt.hist(features, bins=bins, color='skyblue', edgecolor='black')
    plt.xlabel(features.name)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {features.name}')
    plt.show()

df['Laptop_Name'].value_counts()

df['User_Rating'].value_counts()

scatter_plot(features=df['User_Rating'])

df['Price_in_IDR'].value_counts()

scatter_plot(features=df['Price_in_IDR'])

df['Laptop_Type'].value_counts()

pie_plot(features=df['Laptop_Type'])

df['Dedicated_GPU_in_GB'].value_counts()

df['Storage_Type'].value_counts()

pie_plot(features=df['Storage_Type'])

df['RAM_in_GB'].value_counts()

pie_plot(features=df['RAM_in_GB'])

df['RAM_Type'].value_counts()

df['Memory_Type'].value_counts()

pie_plot(features=df['Memory_Type'])

df['OS'].value_counts()

pie_plot(features=df['OS'])

df['Touchscreen_Features'].value_counts()

pie_plot(features=df['Touchscreen_Features'])

df['Screen_Size_in_Inch'].value_counts()

df['Weight_in_Kg'].value_counts()

scatter_plot(features=df['Weight_in_Kg'])

df['Refresh_Rate'].value_counts()

df['Screen_Resolution'].value_counts()

pie_plot(features=df['Screen_Resolution'])

df['Laptop_Company'].value_counts()

histogram_plot(features=df['Laptop_Company'])

df['Storage_Capacity'].value_counts()

df['Processor'].value_counts()

df['CPU_Rank'].value_counts()

df['Battery_Lifetime_in_Hrs'].value_counts()

pie_plot(features=df['Battery_Lifetime_in_Hrs'])

df['GPU'].value_counts()

df['GPU_Benchmark_Score'].value_counts()

df['RAM_Type_Tokenized'].value_counts()

df['GPU_Processor_Tokenized'].value_counts()

"""## Recommendation System - Content-Based Filtering"""

# method 1 rekomendasi ketika user memilih laptop atau ketika suatu laptop terdisplay (index)
# method 2 ketika user memilih feature (seperti nama laptop, atau spek ram atau storage)

df['Laptop_Full_Description'] = df['Laptop_Name'] + ' ' + df['Laptop_Type'] + ' ' + str(df['Storage_Capacity']) + 'GB ' + df['Storage_Type'] + ' ' + df['Memory_Type'] + ' ' + str(df['RAM_in_GB']) + 'GB '+ df['RAM_Type'] + ' ' + df['Processor'] + ' ' + df['GPU'] + str(df['Refresh_Rate']) +'Hz '+ str(df['Weight_in_Kg']) + 'Kg ' + str(df['Screen_Size_in_Inch']) + 'Inch ' + df['Touchscreen_Features'] + ' ' + df['OS'] + ' ' + str(df['Price_in_IDR']) + ' Rupiah'

df_vectorize = df[['Laptop_Index', 'Laptop_Full_Description']]

df_vectorize

"""#### Vectorizing and Cosine Similarity

###### method 1 - based on selected indexing (all feature)
"""

cv = CountVectorizer()

laptop_vector = cv.fit_transform(df_vectorize['Laptop_Full_Description'])
laptop_vector.shape

laptop_similarity = cosine_similarity(laptop_vector)
laptop_similarity

laptop_similarity.shape

def indexing_recommendation(laptop, display_num: int=10):
    index = df_vectorize[df_vectorize['Laptop_Index'] == laptop.index[0]].index[0]
    distances = laptop_similarity[index]
    recommendation_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:display_num+1]

    return recommendation_list

index_parsing = df[df.index == 1]
index_parsing

index_recom = indexing_recommendation(laptop=index_parsing, display_num=5)

for i in index_recom:
    print(df.iloc[i[0]])
    print('-' * 50)

"""###### method 2 - based on selected features"""

tfidf = TfidfVectorizer()

tfidf_matrix = tfidf.fit_transform(df_vectorize['Laptop_Full_Description'])
tfidf_matrix.shape

def text_recommendation(input_text, num_display: int=10):
    if not isinstance(input_text, str):
        input_text = str(input_text)

    input_vector = tfidf.transform([input_text])
    cosine_sim = linear_kernel(input_vector, tfidf_matrix).flatten()

    similarity_score = sorted(enumerate(cosine_sim), key=lambda x: x[1], reverse=True)
    similarity_score = similarity_score[1:num_display+1]

    return similarity_score

search_text = 'Asus ROG'

text_recom = text_recommendation(input_text=search_text, num_display=5)

for i in text_recom:
    print(df.iloc[i[0]])
    print('-' * 50)

"""## Add On Filtering"""

df.columns

df['Storage_Type'].unique()

"""#### Normal Filtering based on Features

###### by price ranges features
"""

def filter_by_price(min_price=None, max_price=None, show_num: int=None, reversed: bool=False):
    filtered_df = df.copy()

    if min_price is not None:
        filtered_df = filtered_df[filtered_df['Price_in_IDR'] >= min_price]
    if max_price is not None:
        filtered_df = filtered_df[filtered_df['Price_in_IDR'] <= max_price]

    sorted_df = filtered_df.sort_values(by='Price_in_IDR', ascending=reversed)

    if show_num is not None:
        return sorted_df.head(show_num)

    return sorted_df

min_price = 5000000
max_price = 10000000

filter_by_price(min_price=min_price, max_price=max_price, show_num=5, reversed=False)

"""###### by keywords arguments of choosen features"""

def filter_by_features(dataframe=None, display_num: int=None, **kwargs):
    if dataframe is not None:
        filtered_df = dataframe
    if dataframe is None:
        filtered_df = df.copy()
    column_map = {col.lower(): col for col in df.columns}

    valid_kwargs = {key: value for key, value in kwargs.items() if key.lower() not in {'display_num'}}

    for key, value in valid_kwargs.items():
        if key.lower() == 'price_in_idr':
            raise KeyError(f"Params '{key}' should not be use, instead change dataframe with filtered by price")
        if value is not None:
            normalized_key = column_map.get(key.lower())
            if normalized_key is None:
                raise KeyError(f"Column '{key}' not found in DataFrame")

            if isinstance(value, str):
                filtered_df = filtered_df[filtered_df[normalized_key].str.casefold() == value.casefold()]
            else:
                filtered_df = filtered_df[filtered_df[normalized_key] == value]

    if display_num is not None:
        return filtered_df.head(display_num)

    return filtered_df

features_option = {
    'User_Rating': None,
    'Laptop_Type': 'Notebook',
    'Processor_Brand': None,
    'Storage_Type': 'SSD',
    'Touchsreen_Features': None,
    'RAM_Type': None,
    'Memory_Type': None,
    'Laptop_Company': None,
    'GPU': None
}

filter_by_features(
    dataframe = filter_by_price(min_price=5000000, max_price=10000000),
    display_num = 5,
    **features_option
)

"""#### Top Rated based on Features

###### by laptop type features
"""

def rated_by_type(laptop_type: str, show_num: int=None, reversed: bool=False):
    filtered_df = df.copy()
    laptop_type = laptop_type.strip().upper()

    if laptop_type in df['Laptop_Type'].str.upper().unique():
        filtered_df = filtered_df[filtered_df['Laptop_Type'].str.upper() == laptop_type]

    sorted_df = filtered_df.sort_values(by='User_Rating', ascending=reversed)

    if show_num is not None:
        return sorted_df.head(show_num)

    return sorted_df

type_option = {
    1: 'Gaming Laptop',
    2: 'Thin and Light Laptop',
    3: '2 in 1 Laptop',
    4: 'Notebook'
}

rated_by_type(laptop_type=type_option[1])

"""###### by ranges price features"""

def rated_by_price(min_price=None, max_price=None, show_num: int=None, reversed: bool=False):
    filtered_df = df.copy()

    if min_price is not None:
        filtered_df = filtered_df[filtered_df['Price_in_IDR'] >= min_price]
    if max_price is not None:
        filtered_df = filtered_df[filtered_df['Price_in_IDR'] <= max_price]

    sorted_df = filtered_df.sort_values(by='User_Rating', ascending=reversed)

    if show_num is not None:
        return sorted_df.head(show_num)

    return sorted_df

min_price = 1000000
max_price = 5000000

rated_by_price(min_price=min_price, max_price=max_price)

"""###### by processor brand features"""

def rated_by_processor(processor_brand: str, show_num: int=None, reversed: bool=False):
    filtered_df = df.copy()
    processor_brand = processor_brand.strip().upper()

    if processor_brand in df['Processor_Brand'].str.upper().unique():
        filtered_df = filtered_df[filtered_df['Processor_Brand'].str.upper() == processor_brand]

    sorted_df = filtered_df.sort_values(by='User_Rating', ascending=reversed)

    return sorted_df.head(show_num)

brand_option = {
    1: 'Intel',
    2: 'AMD',
    3: 'Qualcomm',
    4: 'Apple',
    5: 'Mediatek'
}

rated_by_processor(processor_brand=brand_option[4], show_num=5)

"""###### by laptop company features"""

def rated_by_company(laptop_company: str, show_num: int=None, reversed: bool=False):
    filtered_df = df.copy()
    laptop_company = laptop_company.strip().upper()

    if laptop_company in df['Laptop_Company'].str.upper().unique():
        filtered_df = filtered_df[filtered_df['Laptop_Company'].str.upper() == laptop_company]

    sorted_df = filtered_df.sort_values(by='User_Rating', ascending=reversed)

    if show_num is not None:
        return sorted_df.head(show_num)

    return sorted_df

company_option = {
    1: 'Asus',
    2: 'HP',
    3: 'Lenovo',
    4: 'Dell',
    5: 'MSI',
    6: 'Realme',
    7: 'Avita',
    8: 'Acer',
    9: 'Samsung',
    10: 'Infinix',
    11: 'LG',
    12: 'Apple',
    13: 'Nokia',
    14: 'Redmibook',
    15: 'MI',
    16: 'VAIO'
}

rated_by_company(laptop_company=company_option[5], show_num=5)

"""###### by memory type features"""

def rated_by_memory(storage_type: str, show_num: int=None, reversed: bool=False):
    filtered_df = df.copy()
    storage_type = storage_type.strip().upper()

    if storage_type == "HDD" or storage_type == "SSD":
        filtered_df = filtered_df[filtered_df['Storage_Type'] == storage_type]

    sorted_df = filtered_df.sort_values(by='User_Rating', ascending=reversed)

    if show_num is not None:
        return sorted_df.head(show_num)

    return sorted_df

memory_option = {
    1: 'HDD',
    2: 'SSD'
}

rated_by_memory(storage_type=memory_option[1], show_num=5)

"""###### by combined features"""

def rated_by_fetures(min_price: int=None, max_price: int=None, laptop_type: str=None, storage_type: str=None, memory_type: str=None, laptop_company: str=None, processor_brand: str=None, show_num: int=None, reversed=False):
    filtered_df = df.copy()

    if laptop_type is not None:
        laptop_type = laptop_type.strip().upper()
        if laptop_type in df['Laptop_Type'].str.upper().unique():
            filtered_df = filtered_df[filtered_df['Laptop_Type'].str.upper() == laptop_type]

    if min_price is not None:
        filtered_df = filtered_df[filtered_df['Price_in_IDR'] >= min_price]
    if max_price is not None:
        filtered_df = filtered_df[filtered_df['Price_in_IDR'] <= max_price]

    if processor_brand is not None:
        processor_brand = processor_brand.strip().upper()
        if processor_brand in df['Processor_Brand'].str.upper().unique():
            filtered_df = filtered_df[filtered_df['Processor_Brand'].str.upper() == processor_brand]

    if storage_type is not None:
        storage_type = storage_type.strip().upper()
        if storage_type in df['Storage_Type'].str.upper().unique():
            filtered_df = filtered_df[filtered_df['Storage_Type'].str.upper() == storage_type]

    if memory_type is not None:
        memory_type = memory_type.strip().upper()
        if memory_type in df['Memory_Type'].str.upper().unique():
            filtered_df = filtered_df[filtered_df['Memory_Type'].str.upper() == memory_type]

    if laptop_company is not None:
        laptop_company = laptop_company.strip().upper()
        if laptop_company in df['Laptop_Company'].str.upper().unique():
            filtered_df = filtered_df[filtered_df['Laptop_Company'].str.upper() == laptop_company]

    sorted_df = filtered_df.sort_values(by='User_Rating', ascending=reversed)

    if show_num is not None:
        return sorted_df.head(show_num)

    return sorted_df

rated_by_fetures(laptop_type='Gaming Laptop', min_price=5000000, max_price=15000000, storage_type='SSD', show_num=5)