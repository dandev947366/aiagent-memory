import chromadb
import ollama

# First pull llama
# ollama.pull("llama3")


# Test code to see if bot is running
# output = ollama.generate(model="llama3", prompt="hello world!")
# response = output["response"]

# print(response)

client = chromadb.Client()

message_history = [
    {"id": 1, "prompt": "What is my name?", "response": "Your name is Dan Le."},
    {"id": 2, "prompt": "What is square root of 9876?", "response": "99.3780659904"},
    {
        "id": 3,
        "prompt": "What kind of dog do I have?",
        "response": "Your dog's name is Roxy.",
    },
]
convo = []


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


create_vector_db(conversations=message_history)
while True:
    prompt = input("USER: \n")
    context = retrieve_embeddings(prompt=prompt)
    prompt = f"USER PROMPT: {prompt} \nCONTEXT FROM EMBEDDINGS: {context}"
    stream_response(prompt=prompt)
