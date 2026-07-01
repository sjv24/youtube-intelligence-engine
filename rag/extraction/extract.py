import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wikipediaapi
from unstructured.partition.pdf import partition_pdf
from utils.general_functions import saveJson, getRawPath, getProcessedPath

def wikiText(topic):
    wiki = wikipediaapi.Wikipedia(
        user_agent="interstellar-rag",
        language="en"
    )
    page = wiki.page(topic)
    if page.exists():
        return {"summary": page.summary, "text": page.text}
    else:
        print(f"Page not found: {topic}")
        return None

def pdfReader(filename):
    filepath = os.path.join(getRawPath(), filename)
    elements = partition_pdf(
        languages=["eng"],
        filename=filepath,
        strategy="hi_res"
    )
    print(f"Extracted {len(elements)} elements from {filename}")

    text = []
    for element in elements:
        text.append({
            "page": element.metadata.page_number,
            "category": element.category,
            "text": element.text
        })

    return {"title": filename.replace(".pdf", ""), "text": text}

if __name__ == "__main__":
    print("Extracting Wikipedia data...")
    wiki_content = wikiText("Interstellar film")
    saveJson(wiki_content, "wiki.json")
    
    print("Extracting screenplay...")
    script_content = pdfReader("interstellar-2014.pdf")
    saveJson(script_content, "pdf.json")

    print("Extracting textbook...")
    script_content = pdfReader("the_science_of_interstellar.pdf")
    saveJson(script_content, "textbook.json")

    print("Done! Files saved to data/processed/")