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
from search_engine import general_search
from sqlalchemy import create_engine, text, MetaData, Table, Column, Float, String, Text
from sqlalchemy.orm import sessionmaker
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from index_manager import IndexManager
UPLOAD_FOLDER = os.path.join('output_data')
ALLOWED_EXTENSIONS = {'csv'}


def renamingColumnNames(df): 
    rename_map = {
    'Career Cluster': 'career_cluster',
    'School District': 'school_district',
    'High School': 'high_school',
    'HS Course Name': 'hs_course_name',
    'HS Course Credits': 'hs_course_credits',
    'HS Course Description': 'hs_course_description',
    'Type of Credit': 'type_of_credit',
    'HS Course CIP Code': 'hs_course_cip_code',
    'College': 'college',
    'College Course': 'college_course',
    'College Course Name': 'college_course_name',
    'College Credits': 'college_credits',
    'College Course Description': 'college_course_description',
    'Applicable College Program': 'applicable_college_program',
    'College Course CIP Code': 'college_course_cip_code',
    'Academic Years': 'academic_years',
    'Status of Articulation': 'status_of_articulation',
    'Articulation': 'articulation',
    'High School Teacher Name': 'high_school_teacher_name',
    'Consortium Name': 'consortium_name',
    }
    # Create reverse mapping (snake_case -> original)
    reverse_map = {v: k for k, v in rename_map.items()}

    # Toggle between original and snake_case
    if any(col in df.columns for col in rename_map):
        # If original columns are present, convert to snake_case
        df.rename(columns=rename_map, inplace=True)
    elif any(col in df.columns for col in reverse_map):
        # If snake_case columns are present, convert back to original
        df.rename(columns=reverse_map, inplace=True)
    return df

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

df = pd.read_csv("output_data/output_course_data.csv")
df = renamingColumnNames(df)

df.to_sql('articulations', con=engine, if_exists='replace', index=False)

# Example: fetch data
result = session.execute(text("SELECT * FROM articulations"))

courses_df_unsorted = pd.DataFrame(result.fetchall(), columns=result.keys())
courses_df_unsorted = renamingColumnNames(courses_df_unsorted)
courses_df = courses_df_unsorted.sort_values(by="Career Cluster") # sorting alphabetically for admin view


 # Create FAISS similarity search
index_manager = IndexManager.get_instance()
index_manager.initialize(courses_df)

courses_df = courses_df.drop(['Articulation', 'High School Teacher Name', 'Consortium Name'], axis=1) # Hidden Columns
current_subset_df = courses_df

json_string = courses_df.to_json(orient='records')
df_unsorted_student = courses_df[['School District', 'High School', 'HS Course Name', 'HS Course Credits', 'HS Course Description', 'College',
                  'College Course', 'College Course Name', 'College Credits', 'Applicable College Program', 'Type of Credit', 'Academic Years']] # adjust this to change what columns are visible in student view
# NOTE any changes made to the student view columns here needs to be reflected in the student view search results
# Otherwise the search results could show admin only columns from the student view
df = df_unsorted_student.sort_values(by='School District') # sorting alphabetically for student view
student_string = df.to_json(orient='records')

# # Initializing flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


# Route for seeing a data
@app.route('/data', methods=['POST'])
def get_time():
    data = request.get_json()
    _ = data['name']
    input_course_description = data["description"]
    
    index_manager = IndexManager.get_instance()
    distances, indices  = index_manager.search_similar_course(input_course_description)


    #  Search for the most similar courses
    df = index_manager.get_df()
    similar_courses = df.iloc[indices].copy()
    similar_courses["Similarity_Score"] = 1/(1+distances)
    
    return {
        'name1': similar_courses["College Course Name"].iloc[0],
        'description1': similar_courses["College Course Description"].iloc[0],
        'college_course1': similar_courses["College"].iloc[0],
        'course_number1': similar_courses["College Course"].iloc[0],
        'name2': similar_courses["College Course Name"].iloc[1],
        'description2': similar_courses["College Course Description"].iloc[1],
        'college_course2': similar_courses["College"].iloc[1],
        'course_number2': similar_courses["College Course"].iloc[1],
        'name3': similar_courses["College Course Name"].iloc[2],
        'description3': similar_courses["College Course Description"].iloc[2],
        'college_course3': similar_courses["College"].iloc[2],
        'course_number3': similar_courses["College Course"].iloc[2],
    }
    
@app.route('/table')
def get_data():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM articulations"))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
        df = renamingColumnNames(df)
        df = df.sort_values(by="Career Cluster")
        df = df.drop(['Articulation', 'High School Teacher Name', 'Consortium Name'], axis=1)
        return df.to_json(orient='records')
    
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
    
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM articulations"))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
        df = renamingColumnNames(df)
        current_subset_df = df.sort_values(by="Career Cluster")
        
   
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
        
        current_subset_df = current_subset_df.drop(['Articulation', 'High School Teacher Name', 'Consortium Name'], axis=1)
        json_string = current_subset_df.to_json(orient='records')

        if len(search_input) != 0:
            index_manager = IndexManager.get_instance()
            results = general_search(
                query=search_input,
                df=current_subset_df,
                model=index_manager.get_model(),
                indices=index_manager.get_general_indices(),
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
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM articulations"))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
        df = renamingColumnNames(df)
        current_subset_df = df.sort_values(by='School District')
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
                df=current_subset_df,
                model=index_manager.get_model(),
                indices=index_manager.get_general_indices(),
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

    # Start with the full DataFrame
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM articulations"))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
        df = renamingColumnNames(df)
        current_subset_df = df.sort_values(by="Career Cluster")
   
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
        current_subset_df = current_subset_df.drop(['Articulation', 'High School Teacher Name', 'Consortium Name'], axis=1)
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
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM articulations"))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
        df = renamingColumnNames(df)
        current_subset_df = df.sort_values(by='School District')
   
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
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM articulations"))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
        df = renamingColumnNames(df)
        df = df[['School District', 'High School', 'HS Course Name', 'HS Course Credits',
                 'HS Course Description', 'College', 'College Course', 'College Course Name',
                 'College Credits', 'Applicable College Program', 'Type of Credit', 'Academic Years']]
        df = df.sort_values(by='School District')
        return df.to_json(orient='records')

@app.route('/highschoolFilter')
def highschool_filter():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT DISTINCT high_school FROM articulations"))
        values = [row[0] for row in result if row[0]]
        return sorted(values)

@app.route('/collegeFilter')
def college_filter():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT DISTINCT college FROM articulations"))
        values = [row[0] for row in result if row[0]]
        return sorted(values)

@app.route('/schooldistrictFilter')
def school_district_filter():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT DISTINCT school_district FROM articulations"))
        values = [row[0] for row in result if row[0]]
        return sorted(values)

@app.route('/careerclusterFilter')
def career_cluster_filter():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT DISTINCT career_cluster FROM articulations"))
        values = [row[0] for row in result if row[0]]
        return sorted(values)

@app.route('/academicyearFilter')
def academic_year_filter():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT DISTINCT academic_years FROM articulations"))
        values = [row[0] for row in result if row[0]]
        return sorted(values)

@app.route('/statusFilter')
def status_filter():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT DISTINCT status_of_articulation FROM articulations"))
        values = [row[0] for row in result if row[0]]
        return sorted(values)

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
    try:
        filepath = secure_filename(file.filename)
        filepath = os.path.join("output_data", filepath)
        if os.path.exists(filepath):
            os.remove(filepath)
        file.save(filepath)
        # Load new data
        df = pd.read_csv(filepath)
        os.remove(filepath)            
        df = renamingColumnNames(df)

        # Replace data in SQL database
        df.to_sql('articulations', con=engine, if_exists='replace', index=False)
        result = session.execute(text("SELECT * FROM articulations"))
        courses_df_unsorted = pd.DataFrame(result.fetchall(), columns=result.keys())
        courses_df_unsorted = renamingColumnNames(courses_df_unsorted)
        courses_df = courses_df_unsorted.sort_values(by="Career Cluster")

        index_manager = IndexManager.get_instance()
        index_manager.initialize(courses_df_unsorted)
        courses_df = courses_df.drop(['Articulation', 'High School Teacher Name', 'Consortium Name'], axis=1)

        return jsonify({'message': 'File uploaded and data saved successfully', 'filename': file.filename}), 200
    except Exception as e:
            return jsonify({'error': f'Failed to process file: {str(e)}'}), 500


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return jsonify({'error': 'File is too large. Max upload size is 16MB.'}), 413

# Running app
if __name__ == '__main__':
    # app.run(host="0.0.0.0", port=10000)
    # port = int(os.environ.get('PORT', 5000))  # Use the PORT env variable or default to 5000
    app.run(host='0.0.0.0', port=5000)
