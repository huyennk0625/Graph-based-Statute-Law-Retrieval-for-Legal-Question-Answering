import re
import json

"""
    Create corpus of article from civil.txt
    Readlines -> extract (part, chapter, section, subsection, content, meta)
"""
def parse_civil_code(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data = []

    current_part = {"id": None, "content": None}
    current_chapter = {"id": None, "content": None}
    current_section = {"id": None, "content": None}
    current_subsection = {"id": None, "content": None}

    current_article = None
    article_buffer = []
    current_meta = None

    pending_meta = None

    def save_article():
        nonlocal current_article, article_buffer, current_meta

        if current_article and article_buffer:
            content = " ".join(article_buffer).strip()
            content = re.sub(r"\s+", " ", content)
            content = re.sub(r"\.[^.]*Deleted[\.,]*\s*$", "", content, flags=re.IGNORECASE)

            data.append({
                "part_id": current_part["id"],
                "part_content": current_part["content"],
                "chapter_id": current_chapter["id"],
                "chapter_content": current_chapter["content"],
                "section_id": current_section["id"],
                "section_content": current_section["content"],
                "subsection_id": current_subsection["id"],
                "subsection_content": current_subsection["content"],
                "article_id": current_article,
                "article_content_en": content,
                "meta": current_meta
            })

        current_article = None
        article_buffer = []
        current_meta = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if re.match(r"^Article\b.*Deleted[\.,]*$", line):
            if len(line.split()) < 7:
                continue

        m = re.match(r"^(Part\s+[IVXLC]+)\s+(.*)", line)
        if m:
            save_article()
            current_part = {"id": m.group(1), "content": m.group(2)}
            current_chapter = {"id": None, "content": None}
            current_section = {"id": None, "content": None}
            current_subsection = {"id": None, "content": None}
            continue

        m = re.match(r"^(Chapter\s+[IVXLC]+)\s+(.*)", line)
        if m:
            save_article()
            current_chapter = {"id": m.group(1), "content": m.group(2)}
            current_section = {"id": None, "content": None}
            current_subsection = {"id": None, "content": None}
            continue

        m = re.match(r"^(Section\s+\d+)\s+(.*)", line)
        if m:
            save_article()
            current_section = {"id": m.group(1), "content": m.group(2)}
            current_subsection = {"id": None, "content": None}
            continue

        m = re.match(r"^(Subsection\s+\d+)\s+(.*)", line)
        if m:
            save_article()
            current_subsection = {"id": m.group(1), "content": m.group(2)}
            continue

        if re.match(r"^\(.*\)$", line):
            pending_meta = line
            continue

        m = re.match(r"^(Article\s+\d+(?:-\d+)?)", line)
        if m:
            save_article()

            current_article = m.group(1)

            current_meta = pending_meta
            pending_meta = None

            rest = line[len(current_article):].strip()
            if rest:
                article_buffer.append(rest)

            continue

        if current_article:
            article_buffer.append(line)

    save_article()

    return data

if __name__ == "__main__":
    result = parse_civil_code("data/coliee-2023/data/COLIEE2023statute_data-English/text/civil_code_en-1to724-2.txt")

    with open("coliee25/data/civil_code_parsed.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(len(result))


"""if __name__ == "__main__":
    input_file = "data/coliee-2023/data/COLIEE2023statute_data-English/text/civil_code_en-1to724-2.txt"
    output_file = "civil_code_parsed.json"

    result = parse_civil_code(input_file)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(result)} articles to {output_file}")"""