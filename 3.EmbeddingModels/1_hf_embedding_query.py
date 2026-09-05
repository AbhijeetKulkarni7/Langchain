from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

embedding  =  HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
)

text  = [ "delhi is the capital of India",
         "Mumbai Is the capital of maharashtra",
          "Chennai is the capital of Tamilnadu" ]

# vector =  embedding.embed_query(text)   <- use this for single line of text
vector =  embedding.embed_documents(text)  # use this for Document embedding 
print(str(vector))





