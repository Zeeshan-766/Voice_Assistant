import cohere
import pyttsx3
import speech_recognition as sr
import chromadb  

class VoiceBot:
    def __init__(self, cohere_api_key, chroma_db_path):
        """Initialize Cohere, ChromaDB, and text-to-speech engine."""
        self.engine = pyttsx3.init()
        self.co = cohere.Client(cohere_api_key)
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.client.get_or_create_collection(name="my_collection")

    def recognize_speech(self):
        """Capture and convert user speech to text."""
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            print("🎤 Speak your query...")
            try:
                audio = recognizer.listen(source)
                return recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                print("❌ Could not understand. Please repeat.")
            except sr.RequestError:
                print("❌ Speech recognition service unavailable.")
                return None

    def embed_text(self, text):
        """Generate embeddings using Cohere."""
        return self.co.embed(texts=[text], model="large").embeddings[0]

    def query_chromadb(self, user_query):
        """Retrieve relevant information from ChromaDB and generate a response."""
        query_embedding = self.embed_text(user_query)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=3)

        if results["documents"]:
            best_match = results["documents"][0][0]
            llm_prompt = f"Answer the following query based on the provided information:\n\nQuery: {user_query}\n\nInformation: {best_match}\n\nResponse:"
            return self.generate_response(llm_prompt)
        return "❌ No relevant information found."

    def generate_response(self, prompt):
        """Generate a response using Cohere LLM."""
        response = self.co.generate(model="command", prompt=prompt, max_tokens=300, temperature=0.7)
        return response.generations[0].text.strip()

    def start_conversation(self):
        """Main function to handle user queries via voice or text."""
        while True:
            user_input = input("\n🎤 Speak or type your query (type 'stop' to exit): ").strip()

            if user_input.lower() == "stop":
                print("🛑 Conversation ended.")
                self.engine.say("Conversation ended.")
                self.engine.runAndWait()
                break  

            query_text = self.recognize_speech() if user_input == "" else user_input

            if query_text:
                if query_text.lower() == "stop":
                    print("🛑 Conversation ended.")
                    self.engine.say("Conversation ended.")
                    self.engine.runAndWait()
                    break

                response_text = self.query_chromadb(query_text)
                print(f"\n🤖 AI Response: {response_text}")
                self.engine.say(response_text)
                self.engine.runAndWait()

# 🔥 Run the Voice Bot 🔥
if __name__ == "__main__":
    bot = VoiceBot(
        cohere_api_key="BYPRiIVwmWkIvrs3Hku4kxnfCQGVDoEdDkXc4Fdo",
        chroma_db_path="chromadb_store"
    )
    bot.start_conversation()
