import openai
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter


def chunk_text(filing_path):
    # load the document and split it into chunks
    loader = TextLoader(filing_path)
    documents = loader.load()

    # split it into chunks
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    document_chunks = text_splitter.split_documents(documents)

    return document_chunks


def create_vector_store(document_chunks, openai_api_key):
    # Create a vector store from document chunks using OpenAI embeddings
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings(openai_api_key=openai_api_key))
    return vector_store

def create_retriever(vector_store, openai_api_key):

    llm = ChatOpenAI(openai_api_key=openai_api_key)

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, what should I focus on to find relevant information?")
    ])
    retriever_chain = create_history_aware_retriever(llm, vector_store.as_retriever(), prompt)
    return retriever_chain

def parse_query(question, openai_api_key):

    client = OpenAI(api_key=openai_api_key)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are an assistant trained to extract company ticker, filing year, type, and section from a query, like as an example if the user asks the risks and uncertainties that could materially have affected Apple company in 2022, it is expected to output an answer in this format: 'AAPL, 2022, 10-K, Item 1'"},
            {"role": "user", "content": question}
        ],
        max_tokens=100
    )
    # Example output parsing needs to be adapted based on expected LLM output
    return response.choices[0].message.content.split(", ")


def generate_answer(context, question, openai_api_key):

    client = OpenAI(api_key=openai_api_key)

    context_content = "\n".join([context[i].page_content for i in range(len(context))])

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",  # Updated model name
        messages=[
            {"role": "system", "content": "You are a knowledgeable assistant trained to answer questions based on given context."},
            {"role": "user", "content": context_content},
            {"role": "user", "content": question}
        ],
        max_tokens=300,
        temperature=0.5
    )

    return response.choices[0].message.content

