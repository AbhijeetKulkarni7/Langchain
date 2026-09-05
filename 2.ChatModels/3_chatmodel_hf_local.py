from langchain_huggingface import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100,
        "temperature": 1.0
    }
)

result = llm.invoke("What is the result of expression 2 + 2 = ? ")

print(result)