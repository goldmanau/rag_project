import gradio as gr
from utils.embeddings import EmbeddingStore
from ctransformers import AutoModelForCausalLM

# Load vector DB + embeddings
store = EmbeddingStore()

# Load local GGUF model
llm = AutoModelForCausalLM.from_pretrained(
    "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",   # change filename to match your model
    model_type="mistral",
    gpu_layers=0,               # CPU only; set >0 if you have GPU
    max_new_tokens=512,
    temperature=0.2
)

def answer_question(question):
    """Retrieve context and generate an answer."""
    context_chunks = store.query(question)
    context = "\n".join(context_chunks)

    prompt = f"""
You are an assistant answering questions based ONLY on the provided context.

Context:
{context}

Question: {question}

Answer:
"""

    response = llm(prompt)
    return response

# Gradio UI
def ui_answer(question):
    return answer_question(question)

app = gr.Interface(
    fn=ui_answer,
    inputs=gr.Textbox(label="Ask a question about your documents"),
    outputs="text",
    title="Document Q&A System (Local RAG)"
)

if __name__ == "__main__":
    app.launch()
