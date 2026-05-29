from dotenv import load_dotenv
import os

from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from PIL import Image


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def load_llm():

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.5
    )

    return llm


CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly, No small talk please.
"""


def set_custom_prompt(custom_prompt_template):

    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

    return prompt


# load database

DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)


# create qa chain

qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={
        'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
    }
)


# now invoke with a single query

user_query = input("Write Query Here: ")

response = qa_chain.invoke({'query': user_query})

result = response["result"]
source_documents = response["source_documents"]


print("\nRESULT:\n")
print(result)

print("\nSOURCE DOCUMENTS:\n")

for i, doc in enumerate(source_documents, 1):

    print(f"\nSource Document {i}:\n")

    print(doc.page_content)

    page_number = doc.metadata.get("page", 0)

    image_path = f"page_images/page_{page_number + 1}.png"

    print(f"\nReference Image: {image_path}")

    image = Image.open(image_path)

    image.show()

    print("\n" + "=" * 80)