import os
from flask import Flask
import pandas as pd
from flask_cors import CORS
import faiss
from flask import Flask, request
from sentence_transformers import SentenceTransformer
import numpy as np

courses_df = pd.read_csv("output_data/output_course_data.csv")

json_string = courses_df.to_json(orient='records')
df = courses_df[['College', 'College Program', 'College Course', 'College Course Name', 'High School', 'HS Course Name', 'HS Course Description', 'HS Course Credits', 'Academic Years']]

student_string = df.to_json(orient='records')
data_list = []


# Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
filename = "output.txt"
f = open(filename, "a")


# Process data
course_embeddings = model.encode(courses_df["HS Course Description"].tolist(), convert_to_numpy=True).astype('float32') # Course Descriptions
d = course_embeddings.shape[1]


# Create FAISS similarity search
index = faiss.IndexFlatL2(d)  # L2 distance index (Euclidean)
index.add(course_embeddings)  # Add all course vectors to the index


    


# # Initializing flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


# Route for seeing a data
@app.route('/data', methods=['POST'])
def get_time():
    data = request.get_json()
    input_course_name = data['name']
    input_course_description = data["description"]
    filename = "output.txt"
    f = open(filename, "a")
    # Convert input course to an embedding
    input_embedding = model.encode([input_course_description], convert_to_numpy=True).astype('float32')

    #  Search for the most similar courses
    distances, indices = index.search(input_embedding, k=10)  
    # Get the top matching courses
    similar_courses = courses_df.iloc[indices[0]].copy()
    similar_courses["Similarity_Score"] = 1 / (1 + distances[0])  # Convert distance to similarity score (higher is better)
    similar_course_names = similar_courses["College Course Name"]
    similar_course_descriptions = similar_courses["College Course Description"]
    similar_course_colleges = similar_courses["College"]
    similar_course_numbers = similar_courses["College Course"]
        
   
    return {
      'name1': similar_course_names.iloc[0],
      'description1': similar_course_descriptions.iloc[0],
      'college_course1': similar_course_colleges.iloc[0],
      'course_number1': similar_course_numbers.iloc[0],
      'name2': similar_course_names.iloc[1],
      'description2': similar_course_descriptions.iloc[1],
      'college_course2': similar_course_colleges.iloc[1],
       'course_number2': similar_course_numbers.iloc[1],
      'name3':similar_course_names.iloc[2],
      'description3': similar_course_descriptions.iloc[2],
      'college_course3': similar_course_colleges.iloc[2],
       'course_number3': similar_course_numbers.iloc[2],
        }
    
@app.route('/table')
def get_data():
#    return data
    return json_string
    
@app.route('/search', methods=['POST'])
def get_search():
    search_input = request.get_json()

    texts = courses_df.iloc[0].astype(str).tolist()
    embeddings = model.encode(texts, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # convert input to embedding

    input_embedding = model.encode([search_input], convert_to_numpy=True).astype('float32')

    #search for most similar attribute
    distances, indices = index.search(input_embedding, k=1)
    similar_attributes = courses_df.columns.to_list()
    att_index = int(indices[0][0])
    best_column = similar_attributes[att_index]

    # Process data
    course_embeddings = model.encode(courses_df[best_column].astype(str).tolist(), convert_to_numpy=True).astype('float32') # Course Descriptions
    d = course_embeddings.shape[1]

    index = faiss.IndexFlatL2(d)
    index.add(course_embeddings)
    distances, indices = index.search(input_embedding, k=25)

    top_rows = courses_df.iloc[indices[0]]
    json_string = top_rows.to_json(orient='records')
    return json_string
@app.route('/studentSearch', methods=['POST'])
def get_student_search():
    search_input = request.get_json()

    texts = courses_df.iloc[0].astype(str).tolist()
    embeddings = model.encode(texts, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # convert input to embedding

    input_embedding = model.encode([search_input], convert_to_numpy=True).astype('float32')

    #search for most similar attribute
    distances, indices = index.search(input_embedding, k=1)
    similar_attributes = courses_df.columns.to_list()
    att_index = int(indices[0][0])
    best_column = similar_attributes[att_index]

    # Process data
    course_embeddings = model.encode(courses_df[best_column].astype(str).tolist(), convert_to_numpy=True).astype('float32') # Course Descriptions
    d = course_embeddings.shape[1]

    index = faiss.IndexFlatL2(d)
    index.add(course_embeddings)
    distances, indices = index.search(input_embedding, k=25)


    top_rows = courses_df.iloc[indices[0]]
    df = top_rows[['College', 'College Program', 'College Course', 'High School', 'College Course Name', 'HS Course Name', 'HS Course Description', 'HS Course Credits', 'Academic Years']]
    json_string = df.to_json(orient='records')
    return json_string

@app.route('/filter', methods=['POST'])
def get_filter():
    filters = request.get_json()
    highschool_filter = filters.get("highschool", "").strip()
    college_filter = filters.get("college", "").strip()

    # Start with the full DataFrame
    current_subset_df = courses_df
    print(highschool_filter)
    # Apply filters conditionally
    if highschool_filter:
        current_subset_df = current_subset_df[current_subset_df["High School"].str.lower() == highschool_filter.lower()]
    
    
    if college_filter:
        current_subset_df = current_subset_df[current_subset_df["College"].str.lower() == college_filter.lower()]

    # Convert the filtered DataFrame to JSON
    json_result = current_subset_df.to_json(orient='records')

    return json_result

@app.route('/')
def home():
    return "Flask app is running!"

@app.route('/student')
def student_view():
    return student_string

@app.route('/highschool')
def highschool_filter():
    return student_string

@app.route('/college')
def college_filter():
    return student_string
# Running app
if __name__ == '__main__':
    # app.run(host="0.0.0.0", port=10000)
    # port = int(os.environ.get('PORT', 5000))  # Use the PORT env variable or default to 5000
    app.run(host='0.0.0.0', port=5000)
    

    
