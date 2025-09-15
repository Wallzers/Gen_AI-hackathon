import os
from flask import Flask, request, render_template, session, redirect, url_for
from flask_session import Session
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# ---------------------------
# Flask setup
# ---------------------------
app = Flask(__name__)
app.secret_key = "your_secret_key"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

VECTOR_DIR = "vectorstore"
vectorstore = None


# ---------------------------
# Index route
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    global vectorstore
    if "chat_history" not in session:
        session["chat_history"] = []

    answer = ""

    if request.method == "POST":
        # ----------------- Upload handling -----------------
        uploaded_files = request.files.getlist("file")  # get list of files
        all_chunks = []

        for file in uploaded_files:
            if file.filename:
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)

                ext = os.path.splitext(filepath)[-1].lower()

                if ext == ".txt":
                    loader = TextLoader(filepath)
                elif ext == ".pdf":
                    loader = PyMuPDFLoader(filepath)
                else:
                    return "Unsupported file type", 400

                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)  # collect chunks from all files

        if all_chunks:
            # Embeddings
            embedding = HuggingFaceEmbeddings(
                model_name="local_models/all-MiniLM-L6-v2",
                model_kwargs={"local_files_only": True}
            )

            # Build or rebuild index
            vectorstore = FAISS.from_documents(all_chunks, embedding)
            vectorstore.save_local(VECTOR_DIR)

    # ----------------- Query handling -----------------
    if "query" in request.form and vectorstore:
        user_input = request.form["query"]
        session["chat_history"].append({"role": "user", "content": user_input})
        session["typing"] = True  # 🚀 flag for typing
        session.modified = True
        llm = Ollama(model="mistral")
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

        result = qa({"query": user_input})
        answer = result["result"]

        # Optional: show sources
        sources = [doc.metadata.get("source", "N/A") for doc in result["source_documents"]]
        if sources:
            answer += f"\n\n📚 Sources: {', '.join(set(sources))}"

        # Save chat
        session["chat_history"].append({"role": "user", "content": user_input})
        session["chat_history"].append({"role": "assistant", "content": answer})
        session.modified = True


    return render_template("new_index.html", chat_history=session["chat_history"], answer=answer)


# ---------------------------
# Reset chat
# ---------------------------
@app.route("/reset")
def reset():
    session.pop("chat_history", None)
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
