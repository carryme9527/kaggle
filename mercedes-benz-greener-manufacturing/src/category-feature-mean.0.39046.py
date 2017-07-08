import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.set_index('ID', drop=False)
test.set_index('ID', drop=False)

for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train[c].values) + list(test[c].values)) 
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

columns = ['X5', 'X0', 'X6', 'X8', 'X3', 'X1', 'X2', 'X4']
averages = {}

for i in range(len(columns)):
  limit = i + 1
  tagname = 'TAG_' + ''.join(columns[:limit])

  train[tagname] = 0
  test[tagname] = 0
  for df in [train, test]:
    for t in range(limit):
      df[tagname] += df[columns[t]] * 100**t

  averages[tagname] = {}
  tags = set(train[tagname].values)
  for tag in tags:
    mean = np.mean(train[train[tagname] == tag]['y'].values)
    averages[tagname][tag] = mean


tagnames = []
for i in range(len(columns)):
  limit = i + 1
  tagname = 'TAG_' + ''.join(columns[:limit])
  tagnames.append(tagname)
tagnames.reverse()

for index in test.index:
  for tagname in tagnames:
    tag = test.loc[index, tagname]
    mean = averages.get(tagname).get(tag, None)
    if mean:
      test.loc[index, 'y'] = mean
      break
  else:
    test.loc[index, 'y'] = 100
    

test.loc[:, ['ID', 'y']].to_csv('../output/category-feature-mean.csv', index=False)
