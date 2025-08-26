'''
This file runs all of the backend function calls, including...
    Reading data from CSV and organizing it before printing it to the user
    AI Similarity Search
    Filters
    General Search Bar (calls search_engine.py)
'''

import os
from flask import Flask, jsonify
import pandas as pd
from flask_cors import CORS
import faiss
from flask import Flask, request
from sentence_transformers import SentenceTransformer
import numpy as np
from search_engine import build_general_indices, general_search
from sqlalchemy import create_engine, text, MetaData, Table, Column, Float, String, Text
from sqlalchemy.orm import sessionmaker
from werkzeug.utils import secure_filename
UPLOAD_FOLDER = os.path.join('output_data')
ALLOWED_EXTENSIONS = {'csv'}

# Define your database URL — change this to your actual DB
engine = create_engine('sqlite:///nwesd.db') 
metadata = MetaData()

# Define the table
articulations = Table('articulations', metadata,
    Column('career_cluster', String, key='Career Cluster'),
    Column('school_district', String, key='School District'),
    Column('high_school', String, key='High School'),
    Column('hs_course_name', String, key='HS Course Name'),
    Column('hs_course_credits', Float, key='HS Course Credits'),
    Column('hs_course_description', Text, key='HS Course Description'),
    Column('type_of_credit', String, key='Type of Credit'),
    Column('hs_course_cip_code', String, key='HS Course CIP Code'),
    Column('college', String, key='College'),
    Column('college_course', String, key='College Course'),
    Column('college_course_name', String, key='College Course Name'),
    Column('college_credits', Float, key='College Credits'),
    Column('college_course_description', Text, key='College Course Description'),
    Column('applicable_college_program', String, key='Applicable College Program'),
    Column('college_course_cip_code', String, key='College Course CIP Code'),
    Column('academic_years', String, key='Academic Years'),
    Column('status_of_articulation', String, key='Status of Articulation'),
    Column('articulation', String, key='Articulation'),
    Column('high_school_teacher_name', String, key='High School Teacher Name'),
    Column('consortium_name', String, key='Consortium Name'),
)

# Create the table in the database
metadata.create_all(engine)

# Create session (optional, if you want to query or insert)
Session = sessionmaker(bind=engine)
session = Session()

# Example: fetch data
result = session.execute(text("SELECT * FROM articulations"))


courses_df_unsorted = pd.DataFrame(result.fetchall(), columns=result.keys())
#courses_df_unsorted = pd.DataFrame(result.fetchall(), columns=result.keys())
courses_df_unsorted = pd.read_csv("output_data/output_course_data.csv")
courses_df = courses_df_unsorted.sort_values(by="Career Cluster") # sorting alphabetically for admin view
courses_df = courses_df.drop(['Articulation', 'High School Teacher Name', 'Consortium Name'], axis=1) # Hidden Columns
current_subset_df = courses_df

json_string = courses_df.to_json(orient='records')
df_unsorted_student = courses_df[['School District', 'High School', 'HS Course Name', 'HS Course Credits', 'HS Course Description', 'College',
                  'College Course', 'College Course Name', 'College Credits', 'Applicable College Program', 'Type of Credit', 'Academic Years']] # adjust this to change what columns are visible in student view
# NOTE any changes made to the student view columns here needs to be reflected in the student view search results
# Otherwise the search results could show admin only columns from the student view
df = df_unsorted_student.sort_values(by='School District') # sorting alphabetically for student view
student_string = df.to_json(orient='records')
data_list = []


# Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
filename = "output.txt"
f = open(filename, "a")


# Process data
course_embeddings = model.encode(courses_df["HS Course Description"].tolist(), convert_to_numpy=True).astype('float32') # Course Descriptions
d = course_embeddings.shape[1]
general_indices = build_general_indices(courses_df, model) # used in search


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
    _ = data['name']
    input_course_description = data["description"]
    
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
    
    return { # Three most similar courses for AI Similarity Search
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
    req = request.get_json()
    
    search_input = req.get("searchInput")
    filters = req.get("filters")
    highschool_filter = filters.get("highschool", "").strip()
    college_filter = filters.get("college", "").strip()
    school_district_filter = filters.get("schooldistrict", "").strip()
    career_cluster_filter = filters.get("careercluster", "").strip()
    academic_year_filter = filters.get("academicyear", "").strip()
    status_filter = filters.get("status", "").strip()
    admin_alphabetical_filter = filters.get("adminalphabetical", "").strip()
    
    # Start with the full DataFrame
    current_subset_df = courses_df
   
    # Apply filters conditionally
    if highschool_filter:
        current_subset_df = current_subset_df[current_subset_df["High School"].str.lower() == highschool_filter.lower()]
   
    if college_filter:
        current_subset_df = current_subset_df[current_subset_df["College"].str.lower() == college_filter.lower()]

    if school_district_filter:
        current_subset_df = current_subset_df[current_subset_df["School District"].str.lower() == school_district_filter.lower()]

    if career_cluster_filter:
        current_subset_df = current_subset_df[current_subset_df["Career Cluster"].str.lower() == career_cluster_filter.lower()]
    
    if academic_year_filter:
        current_subset_df = current_subset_df[current_subset_df["Academic Years"].str.lower() == academic_year_filter.lower()]
    
    if status_filter:
        current_subset_df = current_subset_df[current_subset_df["Status of Articulation"].str.lower() == status_filter.lower()]
    
    if admin_alphabetical_filter:
        current_subset_df = current_subset_df.sort_values(by=admin_alphabetical_filter)

    json_string = current_subset_df.to_json(orient='records')

    if len(search_input) != 0:
        results = general_search(
            query=search_input,
            df=courses_df,
            model=model,
            indices=general_indices,
            top_k_per_col=30,   # tweak if you want faster/slower OR if dataset is smaller than 30
            min_results=10,     # guarantee at least 10 results
            rel_threshold=0.60, # keep everything over 60% match
            max_results=None # can cap results if needed
        )
        json_string = results.to_json(orient='records')
    
    return json_string


@app.route('/studentSearch', methods=['POST'])
def get_student_search():
    req = request.get_json()
    
    search_input = req.get("searchInput")
    filters = req.get("filters")
    highschool_filter = filters.get("highschool", "").strip()
    college_filter = filters.get("college", "").strip()
    school_district_filter = filters.get("schooldistrict", "").strip()
    career_cluster_filter = filters.get("careercluster", "").strip()
    academic_year_filter = filters.get("academicyear", "").strip()
    status_filter = filters.get("status", "").strip()
    student_alphabetical_filter = filters.get("studentalphabetical", "").strip()
    
    # Start with the full DataFrame
    current_subset_df = courses_df
   
    # Apply filters conditionally
    if highschool_filter:
        current_subset_df = current_subset_df[current_subset_df["High School"].str.lower() == highschool_filter.lower()]
   
    if college_filter:
        current_subset_df = current_subset_df[current_subset_df["College"].str.lower() == college_filter.lower()]

    if school_district_filter:
        current_subset_df = current_subset_df[current_subset_df["School District"].str.lower() == school_district_filter.lower()]

    if career_cluster_filter:
        current_subset_df = current_subset_df[current_subset_df["Career Cluster"].str.lower() == career_cluster_filter.lower()]

    if academic_year_filter:
        current_subset_df = current_subset_df[current_subset_df["Academic Years"].str.lower() == academic_year_filter.lower()]
    
    if status_filter:
        current_subset_df = current_subset_df[current_subset_df["Status of Articulation"].str.lower() == status_filter.lower()]
    
    if student_alphabetical_filter:
        current_subset_df = current_subset_df.sort_values(by=student_alphabetical_filter)

        

    # json_string = current_subset_df.to_json(orient='records')
    if len(search_input) != 0:
        results = general_search(
            query=search_input,
            df=courses_df,
            model=model,
            indices=general_indices,
            top_k_per_col=30,   # tweak if you want faster/slower OR if dataset smaller than 30
            min_results=10,     # guarantee at least 10 results
            rel_threshold=0.60, # keep everything over 60% match
            max_results=None # can cap results if needed
        )
        current_subset_df = results
    
    current_subset_df = current_subset_df[['School District', 'High School', 'HS Course Name', 'HS Course Credits', 'HS Course Description', 'College',
                  'College Course', 'College Course Name', 'College Credits', 'Type of Credit', 'Academic Years']]
    json_string = current_subset_df.to_json(orient='records')
    return json_string

@app.route('/filter', methods=['POST'])
def get_filter():
    filters = request.get_json()
    highschool_filter = filters.get("highschool", "").strip()
    college_filter = filters.get("college", "").strip()
    school_district_filter = filters.get("schooldistrict", "").strip()
    career_cluster_filter = filters.get("careercluster", "").strip()
    academic_year_filter = filters.get("academicyear", "").strip()
    status_filter = filters.get("status", "").strip()
    admin_alphabetical_filter = filters.get("adminalphabetical", "").strip()
    print(admin_alphabetical_filter)

    # Start with the full DataFrame
    current_subset_df = courses_df
   
    # Apply filters conditionally
    if highschool_filter:
        current_subset_df = current_subset_df[current_subset_df["High School"].str.lower() == highschool_filter.lower()]
   
    if college_filter:
        current_subset_df = current_subset_df[current_subset_df["College"].str.lower() == college_filter.lower()]

    if school_district_filter:
        current_subset_df = current_subset_df[current_subset_df["School District"].str.lower() == school_district_filter.lower()]

    if career_cluster_filter:
        current_subset_df = current_subset_df[current_subset_df["Career Cluster"].str.lower() == career_cluster_filter.lower()]

    if academic_year_filter:
        current_subset_df = current_subset_df[current_subset_df["Academic Years"].str.lower() == academic_year_filter.lower()]

    if status_filter:
        current_subset_df = current_subset_df[current_subset_df["Status of Articulation"].str.lower() == status_filter.lower()]
    
    if admin_alphabetical_filter:
        current_subset_df = current_subset_df.sort_values(by=admin_alphabetical_filter)

    # Convert the filtered DataFrame to JSON
    json_result = current_subset_df.to_json(orient='records')

    return json_result

@app.route('/studentfilter', methods=['POST'])
def get_student_filter():
    filters = request.get_json()
    highschool_filter = filters.get("highschool", "").strip()
    college_filter = filters.get("college", "").strip()
    school_district_filter = filters.get("schooldistrict", "").strip()
    career_cluster_filter = filters.get("careercluster", "").strip()
    academic_year_filter = filters.get("academicyear", "").strip()
    status_filter = filters.get("status", "").strip()
    student_alphabetical_filter = filters.get("studentalphabetical", "").strip()

    # Start with the full DataFrame
    current_subset_df = courses_df
   
    # Apply filters conditionally
    if highschool_filter:
        current_subset_df = current_subset_df[current_subset_df["High School"].str.lower() == highschool_filter.lower()]
   
    if college_filter:
        current_subset_df = current_subset_df[current_subset_df["College"].str.lower() == college_filter.lower()]

    if school_district_filter:
        current_subset_df = current_subset_df[current_subset_df["School District"].str.lower() == school_district_filter.lower()]

    if career_cluster_filter:
        current_subset_df = current_subset_df[current_subset_df["Career Cluster"].str.lower() == career_cluster_filter.lower()]
    
    if academic_year_filter:
        current_subset_df = current_subset_df[current_subset_df["Academic Years"].str.lower() == academic_year_filter.lower()]
    
    if status_filter:
        current_subset_df = current_subset_df[current_subset_df["Status of Articulation"].str.lower() == status_filter.lower()]
    
    if student_alphabetical_filter:
        current_subset_df = current_subset_df.sort_values(by=student_alphabetical_filter)
    

    # Convert the filtered DataFrame to JSON
    current_subset_df = current_subset_df[['School District', 'High School', 'HS Course Name', 'HS Course Credits', 'HS Course Description', 'College',
                  'College Course', 'College Course Name', 'College Credits', 'Type of Credit', 'Academic Years']]
    
    json_result = current_subset_df.to_json(orient='records')

    return json_result

@app.route('/')
def home():
    return "Flask app is running!"

@app.route('/student')
def student_view():
    return student_string

@app.route('/highschoolFilter')
def highschool_filter():
    unique_highschools =[]

    for x in courses_df["High School"]: 
        if x not in unique_highschools: 
            unique_highschools.append(x)
    sorted_unique_highschools = sorted(unique_highschools) #Alphabetizes filter
    return sorted_unique_highschools

@app.route('/collegeFilter')
def college_filter():
    #Checks for the unique values all with that name
    unique_colleges =[]

    for x in courses_df["College"]: 
        if x not in unique_colleges: 
            unique_colleges.append(x)
    sorted_unique_colleges = sorted(unique_colleges)
    return sorted_unique_colleges

@app.route('/schooldistrictFilter')
def school_district_filter():
    unique_school_districts =[]

    for x in courses_df["School District"]: 
        if x not in unique_school_districts: 
            unique_school_districts.append(x)
    sorted_unique_school_districts = sorted(unique_school_districts)
    return sorted_unique_school_districts

@app.route('/careerclusterFilter')
def career_cluster_filter():
    unique_career_cluster =[]

    for x in courses_df["Career Cluster"]: 
        if x not in unique_career_cluster: 
            unique_career_cluster.append(x)
    sorted_unique_career_cluster = sorted(unique_career_cluster)
    return sorted_unique_career_cluster

@app.route('/academicyearFilter')
def academic_year_filter():
    unique_academic_years =[]

    for x in courses_df["Academic Years"]: 
        if x not in unique_academic_years: 
            unique_academic_years.append(x)
    sorted_unique_academic_years = sorted(unique_academic_years)
    return sorted_unique_academic_years

@app.route('/statusFilter')
def status_filter():
    unique_status =[]

    for x in courses_df["Status of Articulation"]: 
        if x not in unique_status: 
            unique_status.append(x)
    sorted_unique_status = sorted(unique_status)
    return sorted_unique_status

@app.route('/adminalphabeticalFilter')
def admin_alphabetical_filter():
    admin_column_headers = ["Career Cluster", "School District", "High School", "HS Course Name", 
                            "College", "Applicable College Program", "College Course", "College Course Name"]
    return admin_column_headers

@app.route('/studentalphabeticalFilter')
def student_alphabetical_filter():
    admin_column_headers = ['School District', 'High School', 'HS Course Name', 'College',
                            'College Course', 'College Course Name']
    return admin_column_headers

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = file.filename
        filepath = os.path.join("output_data", filename)
        file.save(filepath)
        return jsonify({ 'filename': filepath}), 200
        # try:
        #     df = pd.read_csv(filepath)
        #     df.to_sql('articulations', con=engine, if_exists='append', index=False)
        # except Exception as e:
        #     return jsonify({'error': f'Failed to process file: {str(e)}'}), 500

        # return jsonify({'message': 'File uploaded and data saved successfully', 'filename': filename}), 200

# Running app
if __name__ == '__main__':
    # app.run(host="0.0.0.0", port=10000)
    # port = int(os.environ.get('PORT', 5000))  # Use the PORT env variable or default to 5000
    app.run(host='0.0.0.0', port=5000)
