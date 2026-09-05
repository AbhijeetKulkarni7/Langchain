from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    max_output_tokens= 10  #restricting output to 10 token only
)

result = model.invoke("What is the capital of USSR in one word?")


print(result.content[0]["text"])
