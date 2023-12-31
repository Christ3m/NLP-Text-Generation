from langchain.llms import GPT4All
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings

#If you don't know the answer, just say that you don't know, don't try to make up an answer.
prompt_template = """
Answer to greatings and use only the following pieces of context to answer the users question accurately.
Do not use any information not provided in the context.

Context: {context}

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(
    template = prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

documents = TextLoader("data.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)

texts = text_splitter.split_documents(documents)

instructor_embeddings = HuggingFaceInstructEmbeddings(
  model_name="hkunlp/instructor-large",
  model_kwargs={"device": "cpu"}
)

vectorstore = FAISS.from_documents(texts, instructor_embeddings)

llm = GPT4All(
    model="orca-mini-3b-gguf2-q4_0.gguf",
    max_tokens=2048,
    allow_download=False,
    backend="gptj",
    verbose=False,
)

qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),
                                 chain_type_kwargs=chain_type_kwargs,
                                 verbose=True,
                                 #return_source_documents=True
                                )

while True:
    answer = qa.run(input("You: "))
    print("chatbot: ", answer)