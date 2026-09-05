from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise  import cosine_similarity
import numpy as np 

load_dotenv()

embedding  =  OpenAIEmbeddings(
    model= "text-embedding-3-large",
    dimensions=300 
)

documents = [
    "In the world full of chaos & wars, the sports still servers a medium of dialog between war nations.",
    "The Little master Sunil Gavskar's batting is treat to watch cricket. He's innings against reak windies pacers without helmet is wellknown.",
    "Former Indian captain Mr.Rahul Dravid is known as The Wall, He tested the patience of odd team's bowlers every single time.",
    "Virat Kolhi is worlds best white ball batsman. Scored highest centuries in ODI",
    "JAsprit bhumhra is a world class pace bowler with high accuracy and most wickets in death overs.",
    "M S dhoni is former indian captain. Won all of the ICC major trophies."
]

query =  "Tell me about sunil gavaskar"

doc_embedding  =  embedding.embed_documents(documents)

query_embedding  =  embedding.embed_query(query)

scores =  cosine_similarity([query_embedding],  doc_embedding) # All parameters must be 2D list  

index, score = sorted(list(enumerate(scores[0])), lambda x:x[1])[-1]   

print(documents[index])
print("Similarity Score: ", score)





