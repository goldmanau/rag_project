import gradio as gr
from utils.embeddings import EmbeddingStore
from llama_cpp import Llama

# Load vector DB + embeddings
store = EmbeddingStore()

# Load local LLM (place your GGUF model in models/)
#llm = Llama(model_path="models/llama-3-8b.gguf")

llm = Llama(
    model_path="models/llama-3-8b.gguf",
    n_ctx=4096,          # or 8192 if your RAM allows
    n_threads=4,         # adjust to your CPU
    n_batch=512,
    temperature=0.2,
    max_tokens=512
)

def answer_question(question):
    """Retrieve context and generate an answer."""
    context_chunks = store.query(question)
    context = "\n".join(context_chunks)

    prompt = f"""
You are an assistant answering questions based on the provided context.

Context:
{context}

Question: {question}
Answer:
"""

    response = llm(prompt)
    return response["choices"][0]["text"]

# Gradio UI
def ui_answer(question):
    return answer_question(question)

app = gr.Interface(
    fn=ui_answer,
    inputs=gr.Textbox(label="Ask a question about your documents"),
    outputs="text",
    title="Document Q&A System (RAG)"
)

if __name__ == "__main__":
    app.launch()
