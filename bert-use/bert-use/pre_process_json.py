import json

def readJsonFile(path):
    with open(path,'r') as f:
        return json.load(f)

def preProcess():
    with open('dev.tsv','w') as f:
        jsonObj=readJsonFile('test.json')
        #print(jsonObj)
        for ele in jsonObj:
            f.write('%s\t%s\n'%(ele['q_type'],ele['question']))

preProcess()
