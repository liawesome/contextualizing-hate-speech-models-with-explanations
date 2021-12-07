import os
import csv
import re
import numpy as np
from sklearn.metrics import confusion_matrix
data_dir = "/content/contextualizing-hate-speech-models-with-explanations/runs/ws_0.15_seed_0/"
f = open(os.path.join(data_dir, 'eval_details_0_test_ws.txt'))
reader = csv.reader(f, delimiter='\t')
#next(reader) #skip header

data=[]
for i, row in enumerate(reader):
  #print(row)
  data.append((row[0],row[1],re.sub(r'\[CLS\]|\[SEP]','', row[2])))
  
df = pd.DataFrame(data, columns=['true_lab', 'pre_lab', 'text'])
#print(df)

# total 
tn, fp, fn, tp = confusion_matrix(df['true_lab'], df['pre_lab']).ravel()

# false positive rate
fpr = fp/(fp+tn) 
print("false positive rate " + np.str(fpr))

# false negative rate
fnr = fn/(fn+tp)
print("false negative rate " + np.str(fnr))