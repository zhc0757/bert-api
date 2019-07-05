#找出所有标签的列表
#make a list of all labels

def makeLabelList(dataSetPath:str):
    with open(dataSetPath,'r') as f:
        labelSet=set()
        lines=f.readlines()
        for line in lines:
            labelSet.add(line[:line.find('\t')])
        return list(labelSet)
