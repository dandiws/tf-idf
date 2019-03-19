import csv
import math
from nltk.tokenize import RegexpTokenizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

#### ---------- Making of dataset structure ---------- ###

# read csv
with open("D:\\Kuliah\\6 - Pengenalan Pola\\tugas3\\diagnosa.csv", encoding='utf-8-sig') as f:
    reader = csv.reader(f,delimiter=";")
    dataset = list(reader)

with_diagnosis = []
no_diagnosis = []
diagnosis = []
for row in dataset:
    if row[1]=='':
        no_diagnosis.append(row[0])
    else:
        with_diagnosis.append(row[0])
        diagnosis.append(row[1])

dataset = with_diagnosis+no_diagnosis

#### ---------- Making of dataset structure (END) ---------- ###

#### ---------- Data preprocessing ---------- ###
tokenizer = RegexpTokenizer(r'\w+')
filterfactory = StopWordRemoverFactory()
stopword = filterfactory.create_stop_word_remover()
stemfactory = StemmerFactory()
stemmer = stemfactory.create_stemmer()

tokenize=[]
for row in dataset:
    # tokenization
    tokens = tokenizer.tokenize(row.lower())
    tokens_string = ' '.join(tokens)

    # filtering (removing stopword)
    filtered = stopword.remove(tokens_string)

    #stemming
    stemed = stemmer.stem(filtered)    
    terms = stemed.split(' ')    
    tokenize.append(terms)

#### ---------- Data preprocessing(END) ---------- ###

def makeTFdict(document):
    tf_dict ={}
    for term in document:
        if term in tf_dict:
            tf_dict[term]+=1
        else:
            tf_dict[term]=1
    
    #computer frequency for each term
    for term in tf_dict:
        tf_dict[term] = tf_dict[term]/len(document)
        # tf_dict[term] = 1+math.log(tf_dict[term])
    return tf_dict

TF_dicts = [makeTFdict(document) for document in tokenize]

# fungsi untuk menghitung jumlah dokumen yang memuat sebuat term
def makeCountDict(dataset):
    count_dict={}
    for document in dataset:
        terms_distinct = list(set(document))
        for term in terms_distinct:
            if term in count_dict:
                count_dict[term] += 1
            else:
                count_dict[term] = 1

    return count_dict

COUNT_dict = makeCountDict(tokenize)

# fungsi untuk menghitung nilai idf untuk setiap term didalam korpus
def makeIDFdict(count_dict,N):
    idf_dict={}
    for term in count_dict:
        idf_dict[term] =math.log(N/count_dict[term])

    return idf_dict

IDF_dict = makeIDFdict(COUNT_dict,len(tokenize))

def makeTFIDFdict(tf_dict,idf_dict):
    tfidf_dict = {}
    for term in tf_dict:
        tfidf_dict[term] = tf_dict[term]*idf_dict[term]

    return tfidf_dict

TFIDF_dicts = [makeTFIDFdict(tf_dict,IDF_dict) for tf_dict in TF_dicts]

WORD_dict = sorted(COUNT_dict.keys())

#fungsi untuk membuat matriks/vektor tfidf dokumen
def makeTFIDFVector(tfidf_dict):
    tfidf_vector = [0.0] * len(WORD_dict)
    i=0
    for i,term in enumerate(WORD_dict):
        if term in tfidf_dict:        
            tfidf_vector[i] = tfidf_dict[term]

    return tfidf_vector      

TFIDF_vector = [makeTFIDFVector(dict) for dict in TFIDF_dicts]

def dot_product(vector_x, vector_y):
    dot = 0.0
    for e_x, e_y in zip(vector_x, vector_y):
        dot += e_x * e_y
    return dot

def magnitude(vector):
    mag = 0.0
    for index in vector:
        mag += math.pow(index, 2)
    return math.sqrt(mag)

def similiarity(vector_x,vector_y):
    return dot_product(vector_x, vector_y)/ (magnitude(vector_x) * magnitude(vector_y))

n = len(with_diagnosis)
m = len(no_diagnosis)
diagnosa_result = [0]*m
similiarities=[0]*m
for i in range(m):
    max=0  
    max_index = None
    for j in range(n):
        sim = similiarity(TFIDF_vector[i+n],TFIDF_vector[j])                
        if sim>max:
            max=sim
            max_index=j
    similiarities[i]=max
    diagnosa_result[i]= max_index

# simpan diagnosa dalam file csv
with open('hasil_diagnosa.csv','w',encoding='utf-8',newline="") as file:
    writer = csv.writer(file,delimiter=";",quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(m):
        diagnosa_index = diagnosa_result[i]
        if diagnosa_index is not None:
            # header : gejala yang akan didiagnosa, gejala termirip, diagnosa penyakit, nilai cosine similiarity
            row = [no_diagnosis[i],with_diagnosis[diagnosa_index],diagnosis[diagnosa_index],round(similiarities[i],2)]
        else:
            row = [no_diagnosis[i],"-","-",0.0]
        
        writer.writerow(row)




