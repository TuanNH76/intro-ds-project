import json
from datetime import datetime
from utils.mongodb import insert_many_documents

# Read the JSON file
with open('count_ner_result.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convert string timestamps to datetime objects
for item in data:
    if 'hour' in item:
        # Parse the datetime string format "YYYY-MM-DD HH"
        item['hour'] = datetime.strptime(item['hour'], '%Y-%m-%d %H')

# Sort data by hour in ascending order
sorted_data = sorted(data, key=lambda x: x['hour'])

# Insert sorted data into MongoDB using mongodb.py function
insert_many_documents('frequency', sorted_data)