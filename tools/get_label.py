import json

def get_label():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    return cat_to_name