import json

# Write into file
data = {"1":"Jatin","2":"Shubham","3":"Vidyaa","4":"Vimal"}
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
f.close()

# Read from file
with open('data.json', 'r', encoding='utf-8') as f:
    jadoo = json.load(f)
print(jadoo)