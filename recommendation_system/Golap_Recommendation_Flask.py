import re
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

app = Flask(__name__)

# Load dataset
df = pd.read_csv('Dataset/Golap_Dataset_Final.csv')

# Generate laptop full descriptions
df['Laptop_Full_Description'] = (
    df['Laptop_Name'] + ' ' + 
    df['Laptop_Type'] + ' ' + 
    str(df['Storage_Capacity']) + 'GB ' + 
    df['Storage_Type'] + ' ' + 
    df['Memory_Type'] + ' ' + 
    str(df['RAM_in_GB']) + 'GB ' + 
    df['RAM_Type'] + ' ' + 
    df['Processor'] + ' ' + 
    df['GPU'] + str(df['Refresh_Rate']) + 'Hz ' + 
    str(df['Weight_in_Kg']) + 'Kg ' + 
    str(df['Screen_Size_in_Inch']) + 'Inch ' + 
    df['Touchscreen_Features'] + ' ' + 
    df['OS'] + ' ' + 
    str(df['Price_in_IDR']) + ' Rupiah'
)

df_vectorize = df[['Laptop_Index', 'Laptop_Full_Description']]

# CountVectorizer for indexing recommendation
cv = CountVectorizer()
laptop_vector = cv.fit_transform(df_vectorize['Laptop_Full_Description'])
laptop_similarity = cosine_similarity(laptop_vector)

def indexing_recommendation(laptop_index, display_num=10):
    index = df_vectorize[df_vectorize['Laptop_Index'] == laptop_index].index[0]
    distances = laptop_similarity[index]
    recommendation_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:display_num+1]
    return recommendation_list

# TfidfVectorizer for text recommendation
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df_vectorize['Laptop_Full_Description'])

def text_recommendation(input_text, num_display=10):
    if not isinstance(input_text, str):
        input_text = str(input_text)

    input_vector = tfidf.transform([input_text])
    cosine_sim = linear_kernel(input_vector, tfidf_matrix).flatten()

    similarity_score = sorted(enumerate(cosine_sim), key=lambda x: x[1], reverse=True)
    similarity_score = similarity_score[1:num_display+1]

    return similarity_score

# Filter by price function
def filter_by_price(min_price=None, max_price=None, show_num=None, reversed=False):
    filtered_df = df.copy()

    if min_price is not None:
        filtered_df = filtered_df[filtered_df['Price_in_IDR'] >= min_price]
    if max_price is not None:
        filtered_df = filtered_df[filtered_df['Price_in_IDR'] <= max_price]

    sorted_df = filtered_df.sort_values(by='Price_in_IDR', ascending=reversed)

    if show_num is not None:
        return sorted_df.head(show_num)

    return sorted_df

# Flask API endpoints
@app.route('/api/index-recommendation', methods=['GET'])
def recommend_by_index():
    try:
        laptop_index = request.args.get('laptop_index', type=int)
        display_num = request.args.get('display_num', type=int, default=10)

        recommendations = indexing_recommendation(laptop_index, display_num=display_num)
        results = [{
            'Laptop_Index': df_vectorize.iloc[i[0]]['Laptop_Index'],
            'Laptop_Full_Description': df_vectorize.iloc[i[0]]['Laptop_Full_Description'],
            'Similarity_Score': i[1]
        } for i in recommendations]

        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/text-recommendation', methods=['GET'])
def recommend_by_text():
    try:
        input_text = request.args.get('input_text', type=str)
        num_display = request.args.get('num_display', type=int, default=10)

        recommendations = text_recommendation(input_text, num_display=num_display)
        results = [{
            'Laptop_Index': df_vectorize.iloc[i[0]]['Laptop_Index'],
            'Laptop_Full_Description': df_vectorize.iloc[i[0]]['Laptop_Full_Description'],
            'Similarity_Score': i[1]
        } for i in recommendations]

        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/recommendation-by-price', methods=['GET'])
def filter_price():
    try:
        min_price = request.args.get('min_price', type=float)
        max_price = request.args.get('max_price', type=float)
        show_num = request.args.get('show_num', type=int, default=None)
        reversed_order = request.args.get('reversed', type=bool, default=False)

        filtered_data = filter_by_price(min_price=min_price, max_price=max_price, show_num=show_num, reversed=reversed_order)
        results = filtered_data.to_dict(orient='records')
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
