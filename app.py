import os
from flask import Flask, request, render_template, jsonify
from google import genai
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv 

load_dotenv()

app = Flask(__name__)

# --- Initialisering af RAG elementer tilpas til jeres db_croma mappe ---
VECTOR_DB_PATH = r"C:\mydev\ragai\chroma_db" 

# Initialiser embedding-modellen (oversætteren)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Indlæs Vektorlageret (Bibliotekaren) fra disken
try:
    vector_db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
    # Opret Retrieveren: Værktøjet der henter relevant info
    retriever = vector_db.as_retriever(search_kwargs={"k": 3}) # Hent de 3 bedste bidder
    print("✅ Chroma DB og Retriever indlæst.")
except Exception as e:
    print(f"FEJL: Kunne ikke indlæse Chroma DB. Har du kørt setup_rag.py? Fejl: {e}")
    retriever = None

# Initialiser Gemini Klienten
# Nøglen aflæses fra miljøvariablen GEMINI_API_KEY på serveren
# opret .env til local test
try:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") # Skal sættes som systemvariabel på Render/Railway!
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY miljøvariabel er ikke sat.")
        
    client = genai.Client(api_key=GEMINI_API_KEY)
    MODEL_NAME = "gemini-2.5-flash" # Hurtig og effektiv model
    print("✅ Gemini Klient klar.")
except ValueError as e:
    print(f"FEJL: {e}")
    client = None

# --- Funktionen der laver RAG magien ---
def get_rag_answer(question):
    if not client or not retriever:
        return "Systemfejl: AI-systemet er ikke korrekt indlæst. Tjek logs."

    # 1. Opslag (Retrieval)
    # Find de mest relevante tekstbidder fra studieordningen
    docs = retriever.invoke(question)
    context = "\n---\n".join([doc.page_content for doc in docs])

    # 2. Prompt Konstruktion (Byg spørgsmålet til Gemini)
    # Fortæl Gemini at den SKAL svare baseret på den tekst, vi giver den
    SYSTEM_PROMPT = (
        "Du er en statistikbank og besvarer spørgsmål omkring Premier League"
    #    "Brug kun den kontekst, der er angivet nedenfor, til at "
    #    "besvare spørgsmålet. Svar på dansk. Hvis konteksten ikke indeholder svaret, "
    #    "skal du sige, at du ikke kan finde informationen."
    )
    
    
    full_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Kontekst fra studieordning:\n{context}\n\n"
        f"Spørgsmål: {question}"
    )

    # 3. Generering (Inference)
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=full_prompt
        )
        return response.text
    except Exception as e:
        return f"Fejl under kommunikation med Gemini: {e}"

# --- Flask Routes ---

@app.route('/')
# Her skal du implementere brugerstyring/login, så kun tilladte kan tilgå 
# (simpel hardcoded login i Flask er fint til demo)
# I koden herunder, antager vi at brugeren er logget ind

def index():
    return render_template('index.html') # Du skal lave en simpel HTML side med en formular

@app.route('/ask', methods=['POST'])
def ask_question():
    user_prompt = request.json.get('prompt')
    if not user_prompt:
        return jsonify({"answer": "Skriv venligst et spørgsmål."}), 400

    # Få svaret fra RAG-systemet
    ai_answer = get_rag_answer(user_prompt)
    
    return jsonify({"answer": ai_answer})

if __name__ == '__main__':
    # Husk: I cloud deployment skal der bruges en WSGI server som Gunicorn
    # Til lokal test:
    app.run(debug=True)