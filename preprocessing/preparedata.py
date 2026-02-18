import os
import xml.etree.ElementTree as ET
import json
import re

"""
    Create combine train and combine test
    Have both en and jp
"""
def parse_articles(text):
    articles = []
    
    # format "Article xxx"
    splits = re.split(r'(^Article [\d\-]+)', text, flags=re.MULTILINE)
    
    for i in range(1, len(splits), 2):
        article_id = splits[i].strip()
        content = splits[i+1].strip()
        
        articles.append({
            "article_id": article_id,
            "article_content": content
        })
    
    return articles


def load_xml_folder(folder_path):
    data = {}
    
    for filename in os.listdir(folder_path):
        if not filename.endswith(".xml"):
            continue
        
        tree = ET.parse(os.path.join(folder_path, filename))
        root = tree.getroot()
        
        for pair in root.findall("pair"):
            pid = pair.get("id")
            label = pair.get("label")
            t1 = pair.find("t1").text.strip()
            t2 = pair.find("t2").text.strip()
            
            data[pid] = {
                "label": label,
                "t1": t1,
                "t2": t2
            }
    
    return data

en_folder = "data/coliee-2025/test-for-task4/en"
jp_folder = "data/coliee-2025/test-for-task4/jp"

en_data = load_xml_folder(en_folder)
jp_data = load_xml_folder(jp_folder)

final_data = []

for pid in en_data:
    if pid not in jp_data:
        continue
    
    en_item = en_data[pid]
    jp_item = jp_data[pid]
    
    articles = parse_articles(en_item["t1"])
    
    final_data.append({
        "id": pid,
        "yn_label": en_item["label"],
        "query": en_item["t2"],
        "relevant_articles": articles,
        "jp_query": jp_item["t2"]
    })

with open("coliee25/data/combine_test.json", "w", encoding="utf-8") as f:
    json.dump(final_data, f, ensure_ascii=False, indent=4)
