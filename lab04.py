#!/opt/userenvs/nikolay.nechaev/prj01/bin/python3
import re
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

pd.set_option("mode.chained_assignment", None)

CLEANR = re.compile('<.*?>')

def cleanhtml(raw_html):
  cleantext = re.sub(CLEANR, '', raw_html)
  return cleantext



path_local_dir = "./data/"



filter = [1025.0, 4.0, 3078.0,522.0]

#prod
path_local_dir = "/data/share/lab04data/"
filter = [1025.0, 4.0, 3078.0, 522.0, 3084.0, 1550.0, 3090.0, 3603.0, 22.0, 3607.0, 1560.0, 1561.0, 2224.0, 859.0, 31.0, 3626.0, 3627.0, 46.0, 1076.0, 2101.0, 3639.0, 1717.0, 3647.0, 3139.0, 2634.0, 1635.0, 1616.0, 3154.0, 3667.0, 783.0, 88.0, 1967.0, 1117.0, 1211.0, 607.0, 2656.0, 1634.0, 1123.0, 2148.0, 1125.0, 1638.0, 1639.0, 1131.0, 2579.0, 3183.0, 3184.0, 2676.0, 1653.0, 630.0, 3703.0, 2682.0, 2173.0, 3203.0, 2692.0, 645.0, 646.0, 2185.0, 3214.0, 1685.0, 623.0, 2199.0, 2200.0, 1690.0, 1691.0, 160.0, 3233.0, 3235.0, 3748.0, 2215.0, 3240.0, 176.0, 3763.0, 3253.0, 2230.0, 695.0, 3256.0, 723.0, 2235.0, 1215.0, 3780.0, 2758.0, 2760.0, 2253.0, 715.0, 1741.0, 3279.0, 2258.0, 3795.0, 1748.0, 2774.0, 219.0, 1244.0, 1759.0, 3808.0, 3299.0, 742.0, 1257.0, 1262.0, 382.0, 3827.0, 468.0, 3317.0, 3831.0, 2812.0, 766.0, 1281.0, 3332.0, 775.0, 3852.0, 2891.0, 3854.0, 3343.0, 274.0, 3349.0, 1302.0, 3352.0, 133.0, 1308.0, 800.0, 290.0, 1827.0, 3292.0, 1829.0, 1831.0, 1323.0, 3886.0, 3378.0, 3892.0, 2869.0, 3904.0, 3906.0, 1347.0, 2372.0, 3398.0, 840.0, 3913.0, 2379.0, 2896.0, 3411.0, 857.0, 3418.0, 3931.0, 352.0, 3426.0, 357.0, 358.0, 1384.0, 2922.0, 2411.0, 3438.0, 880.0, 372.0, 1909.0, 888.0, 1344.0, 3454.0, 1920.0, 386.0, 3223.0, 903.0, 322.0, 1432.0, 2288.0, 414.0, 2374.0, 1441.0, 2067.0, 421.0, 1961.0, 943.0, 1457.0, 3509.0, 3715.0, 2490.0, 955.0, 3744.0, 958.0, 3007.0, 3009.0, 450.0, 1806.0, 3526.0, 967.0, 2504.0, 464.0, 3025.0, 3538.0, 1700.0, 2517.0, 1500.0, 3550.0, 2026.0, 2029.0, 2030.0, 1519.0, 503.0, 3065.0, 1019.0, 1020.0, 510.0]

filter = [int(item) for item in filter]


def create_dataframe(matrix, tokens):
    doc_names = [f'doc_{i+1}' for i, _ in enumerate(matrix)]
    df = pd.DataFrame(data=matrix, index=doc_names, columns=tokens)
    return(df)



cv = CountVectorizer()
cv = TfidfVectorizer()
fulltext = []

for i in range(1,21):
    with open("{}base_{}.txt".format(path_local_dir,i),'r') as base:
        #fulltext+= cleanhtml(base.read())
        fulltext.append(cleanhtml(base.read()))

std_mean = 0

out = []

defined = []
other = []

for i in filter:
    docs = []
    docs = fulltext.copy()
    #file_id = filter[i]
    with open("{}test_{}.txt".format(path_local_dir, i), 'r') as test:
        text = cleanhtml(test.read())
        docs.append(text)
        text_matrix = cv.fit_transform(docs)
        df = text_matrix.todense()
        sc = cosine_similarity(df)[20]
        out.append(sc)
        std_mean = std_mean + sc.sum()


std_mean = (std_mean-(len(filter))) / len(filter)

for idx,d in enumerate(out):
    if (d.sum()-1) >= std_mean:
        defined.append(filter[idx])
    else:
        other.append(filter[idx])


"""
test_text = []

"""
print("{\n\"defined\":",end='')
print(defined,end=',\n')
print("\"other\":",end='')
print(other)
print("}")




