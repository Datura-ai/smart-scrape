
import re


# First, run grep "Received images_themes" /root/.pm2/logs/[pm2 process name].log >> image_themes.txt
def process_file(file_path):
    # Read the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Extract elements
    lines = content.split('\n')
    all_elements = []
    for line in lines:
        elements = line.split(',')
        cleaned_elements = [element.strip().strip('"') for element in elements]
        all_elements.extend(cleaned_elements)

    # Remove duplicates
    unique_elements = list(set(all_elements))

    # Filter out elements with more than 4 words
    filtered_elements = [element for element in unique_elements if len(element.split()) <= 4]

    # Clean and format as a Python list
    cleaned_elements_python_list = [element.replace("'", "").replace("[", "").replace("]", "").strip() for element in filtered_elements if element.strip()]
    python_list_str = "image_themes = [\n    '" + "',\n    '".join(cleaned_elements_python_list) + "'\n]"

    return python_list_str


file_path = 'image_themes.txt' 
formatted_list = process_file(file_path)
print(formatted_list)
