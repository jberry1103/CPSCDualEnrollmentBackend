# # Filename - server.py

# # Import flask and datetime module for showing date and time
# from flask import Flask, request, jsonify
# import datetime
# from flask_cors import CORS
# from sentence_transformers import SentenceTransformer
# import openpyxl

# # Load the workbook
# workbook = openpyxl.load_workbook('2024-25CTEArticulations 3.xlsx')
# model = SentenceTransformer("all-MiniLM-L6-v2")
# # Get the sheet by name or index
# sheet = workbook['Sheet1']  # Replace 'Sheet1' with your sheet name
# # sheet = workbook.active # To get the active sheet
# x = datetime.datetime.now()

# # Initializing flask app
# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})


# Filename - server.py

# Import flask and datetime module for showing date and time
from flask import Flask
import datetime
import pandas as pd
from flask_cors import CORS
import faiss
from flask import Flask, request, jsonify
import json
# import openpyxl
from sentence_transformers import SentenceTransformer
courses_df = pd.read_csv("output_data/output_course_data.csv")
json_string = courses_df.to_json(orient='records')
# data = json.loads(courses_df)
print(json_string)
data_list = []
#for row in sheet.iter_rows(values_only=True):
#    data_list.append(list(row))

# Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
filename = "output.txt"
f = open(filename, "a")
# input_course_name = input("Enter HS Course Name: ")
# input_course_description = input("Enter a brief course description: ")

# Process data
course_embeddings = model.encode(courses_df["HS Course Description"].tolist(), convert_to_numpy=True).astype('float32') # Course Descriptions
d = course_embeddings.shape[1]


# Create FAISS similarity search
index = faiss.IndexFlatL2(d)  # L2 distance index (Euclidean)
index.add(course_embeddings)  # Add all course vectors to the index
# workbook = openpyxl.load_workbook('2024-25CTEArticulations 3.xlsx')

# data_list = []
# for row in sheet.iter_rows(values_only=True):
#     data_list.append(list(row))

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
# x = datetime.datetime.now()

# Initializing flask app
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
    
    print(similar_course_numbers.iloc[0])
    
    # instances_list = [] # [course name similarity, course description similarity, course name, course desc]
    # course_name_similarities = []
    # course_desc_similarities = []
    # similar_instances = []
    # similar_instance_course_names = []
    # num_similar = 0
    # test = 0
    # count = [0,0,0,0,0]
    # averageValue = [0, 0, 0, 0, 0]
    # for row in data_list[2:100]:
    #   sentences = [
    #   row[0],
    #   row[1],
    #   row[2],
    #   row[3],
    #   row[4],
    #   input_course_name,
    #   input_course_description
    #   ]

    
    #   embeddings = model.encode(sentences)
    #   print(embeddings.shape)
    #   # [3, 384]

    #   # 3. Calculate the embedding similarities
    #   similarities = model.similarity(embeddings, embeddings)
    #   print(similarities)
    #   for x in range (0, 5): 
    #     new_item = similarities[x]
    #     for y in range (0, 5):
    #         if (y != x):
    #            count[y] = count[y] + 1
    #            test = test + new_item[y].item()
    #            averageValue[y] = averageValue[y] + new_item[y].item() 
    
    #   if sentences[2] not in similar_instance_course_names:
    #     instances_list.append([similarities[0][5], similarities[6][4], sentences[2], sentences[4]])
    #     course_name_similarities.append(similarities[0][5])
    #     course_desc_similarities.append(similarities[6][4])
    #     similar_instance_course_names.append(sentences[2])
    
    # most_similar_courses = []
    # print("Instances list: ")
    # print(instances_list)
    # for i in range(0, 3):
    #   most_similar_name = max(course_name_similarities)
    #   most_similar_desc = max(course_desc_similarities)
    #   print("most similar name: ", most_similar_name)
    #   print("most similar desc: ", most_similar_desc)
    #   instance_index = 0
    #   if most_similar_name > most_similar_desc:
    #     for j in range(len(instances_list)):
    #         print("j: ", j)
    #         #print("name instance: ", instances_list[j][0], "  ideal name instance: ", most_similar_name)
    #         if (instances_list[j][0] == most_similar_name):
    #             most_similar_courses.append(instances_list[j])
    #             instance_index = j
    #             print("POP")
    #             if (instances_list[j][1] == most_similar_desc):
    #                 course_desc_similarities.remove(most_similar_desc)
    #     instances_list.pop(instance_index)
    #     course_name_similarities.remove(most_similar_name)

    #   else:
    #     for j in range(len(instances_list)):
    #         print("j: ", j)
    #         #print("course desc instance: ", instances_list[j][0], "  ideal course desc instance: ", most_similar_desc)
    #         if (instances_list[j][1] == most_similar_desc):
    #             most_similar_courses.append(instances_list[j])
    #             instance_index = j
    #             print("POP")
    #             if (instances_list[j][0] == most_similar_name):
    #                 course_name_similarities.remove(most_similar_name)
    #     instances_list.pop(instance_index)
    #     course_desc_similarities.remove(most_similar_desc)



    # f.close()
            
    # #if ((similarities[0][5] >= 0.6) or (similarities[6][4] >= 0.6)):
    # #    if sentences[2] not in similar_instance_course_names:
    # #        print("SIMILAR!! ---------")
    # #        similar_instances.append(similarities)
    # #        similar_instance_course_names.append(sentences[2])
    # #        num_similar += 1
    # #    else:
    # #        print("Duplicate course (Skipped)")
    
    # print("t", test)
    # # f.write(similarities)
    # # tensor([[1.0000, 0.6660, 0.1046],
    # #         [0.6660, 1.0000, 0.1411],
    # #         [0.1046, 0.1411, 1.0000]])
    # # Returning an api for showing in  reactjs
    # # return {
    # #     'name1': " Computer Application Essentials INFO 101",
    # #   'description1': "Empowering Your Future CTB104 (Computer Application Essentials) introduces students to fundamental computer skills necessary for academic and professional success. The course covers essential software applications, including word processing, spreadsheets, and presentations, while emphasizing digital literacy, problem-solving, and productivity tools. Students will develop practical skills to efficiently use technology in everyday tasks and enhance their ability to communicate and collaborate in a digital world.",
    # #   'name2': " Information Technology, Introduction to IT 101",
    # #   'description2': "Introduces students to the fundamentals of information technology, including computer hardware, software, networks, and basic problem-solving techniques in IT.",
    # #   'name3': " Introduction to Programming SOFT 102",
    # #   'description3': "Introduces students to programming, focusing on basic coding techniques, problem-solving, and the fundamentals of writing and debugging programs.",
    # #     } similar_course_numbers
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
    
    

# Running app
if __name__ == '__main__':
    app.run(debug=True)