import xml.dom.minidom as xdm
import random

DOMTree=xdm.parse('fudan_17252.xml')
questionBank = DOMTree.documentElement
#print(questionBank)
docs=questionBank.getElementsByTagName('DOC')
#print(docs)
with open('dev.tsv','w') as dev_f:
    train_f=open('train.tsv','w')
    for doc in docs:
        if random.random()>0.7:
            dev_f.write('%s\t%s\n'%(doc.getElementsByTagName('QuestionStyle')[0].childNodes[0].data.lstrip('\ufeff'),
                                doc.getElementsByTagName('Question')[0].childNodes[0].data))
        else:
            train_f.write('%s\t%s\n'%(doc.getElementsByTagName('QuestionStyle')[0].childNodes[0].data.lstrip('\ufeff'),
                                doc.getElementsByTagName('Question')[0].childNodes[0].data))
    train_f.close()
#
