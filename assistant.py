import chromadb
import ollama
import psycopg
from psycopg.rows import dict_row

# First pull llama
# ollama.pull("llama3")


# Test code to see if bot is running
# output = ollama.generate(model="llama3", prompt="hello world!")
# response = output["response"]

# print(response)

client = chromadb.Client()

system_prompt = (
    "You are an AI assistant that has memory of every conversation you have ever had with this user."
    "On every prompt from the user, the system has checked for any relevant messages you have had with the user."
    "If any embedded previous conversations are attached, use them for context to responding to the user,"
    "if the context is relevant and useful to responding. If the recalled conversations are irrelevant,"
    "disregard speaking about them and respond normally as an AI assistant. Do not talk about recalling conversations."
    "Just use any useful data from the previous conversations and respond normally as an intelligent AI assistant."
)
convo = [{"role": "system", "content": system_prompt}]
DB_PARAMS = {
    "dbname": "memory_agent",
    "user": "ai",
    "password": "password",
    "host": "localhost",
    "port": "5432",
}


def connect_db():
    conn = psycopg.connect(**DB_PARAMS)
    return conn


def fetch_conversations():
    conn = connect_db()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute("SELECT * FROM conversations")
        conversations = cursor.fetchall()
    conn.close()
    return conversations


def store_conversations(prompt, response):
    conn = connect_db()
    with conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO conversations (timestamp, prompt, response) VALUES (CURRENT_TIMESTAMP, %s, %s)",
            (prompt, response),
        )
        conn.commit()
    conn.close()


def stream_response(prompt):
    convo.append({"role": "user", "content": prompt})
    response = ""
    stream = ollama.chat(model="llama3", messages=convo, stream=True)
    print("\nASSISTANT:")

    for chunk in stream:
        content = chunk["message"]["content"]
        response += content
        print(content, end="", flush=True)

    print("\n")
    store_conversations(prompt=prompt, response=response)
    convo.append({"role": "assistant", "content": prompt})


def create_vector_db(conversations):
    vector_db_name = "conversations"
    try:
        client.delete_collection(name=vector_db_name)
    except ValueError:
        pass
    vector_db = client.create_collection(name=vector_db_name)
    for c in conversations:
        serialized_convo = f"prompt: {c['prompt']} response: {c['response']}"
        response = ollama.embeddings(model="nomic-embed-text", prompt=serialized_convo)
        embedding = response["embedding"]
        vector_db.add(
            ids=[str(c["id"])], embeddings=[embedding], documents=[serialized_convo]
        )


def retrieve_embeddings(prompt):
    # Generate the embedding for the input prompt
    response = ollama.embeddings(model="nomic-embed-text", prompt=prompt)
    prompt_embedding = response["embedding"]

    # Retrieve the collection
    vector_db = client.get_collection(name="conversations")

    # Query the vector database for the closest matching conversation
    results = vector_db.query(query_embeddings=[prompt_embedding], n_results=1)

    # Get the best matching document (conversation context)
    best_embedding = results["documents"][0][0]
    return best_embedding


conversations = fetch_conversations()
create_vector_db(conversations=conversations)
print(fetch_conversations())
while True:
    prompt = input("USER: \n")
    context = retrieve_embeddings(prompt=prompt)
    prompt = f"USER PROMPT: {prompt} \nCONTEXT FROM EMBEDDINGS: {context}"
    stream_response(prompt=prompt)
