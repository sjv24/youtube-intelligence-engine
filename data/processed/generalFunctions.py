import json
import pandas as pd
def saveJson(data,path):
    with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

def readJson(path):
    with open(path, encoding="utf-8") as file_:
        return json.load(file_)

def getPath():
    return r"C:\Users\saart\Desktop\370_NLP\W8L7_Stu\W8L7_Stu\\"

def saveList(data,file_path):
     df = pd.DataFrame(data, columns=["chunks"])
     df.to_csv(file_path, index=False)

def getList(file_path,column):
    df = pd.read_csv(file_path)
    texts = df[column].tolist()
    return texts