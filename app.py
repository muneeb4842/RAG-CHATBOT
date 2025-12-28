import os
import re
import contextlib
import numpy as np
import torch
import gradio as gr
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# HF stability
_ = np.__version__
_ = torch.__version__

# Groq (set GROQ_API_KEY in HF Secrets)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Embeddings (Enhancement #1)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# Fixed settings (because you want to remove sliders)
TOP_K = 5
TEMPERATURE = 0.2

STATE = {
    "chunks": None,   # list of {text, source, page}
    "embeds": None,   # np.array (n, d)
    "file_text": {},  # filename -> preview text
}

# -----------------------------
# PDF extraction
# -----------------------------
def extract_pdf_pages(pdf_file):
    reader = PdfReader(pdf_file)
    name = os.path.basename(getattr(pdf_file, "name", "uploaded.pdf"))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append({
                "text": re.sub(r"\s+", " ", text).strip(),
                "source": name,
                "page": i + 1
            })
    return pages

# -----------------------------
# Chunking
# -----------------------------
def chunk_text(text, size=900, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = max(0, end - overlap)
    return chunks

def build_chunks(pages):
    records = []
    for p in pages:
        for c in chunk_text(p["text"]):
            c = c.strip()
            if c:
                records.append({"text": c, "source": p["source"], "page": p["page"]})
    return records

# -----------------------------
# Retrieval
# -----------------------------
def retrieve(query, chunks, embeds, k=TOP_K):
    q = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = np.dot(embeds, q)
    top = sims.argsort()[-k:][::-1]
    return [{
        "text": chunks[i]["text"],
        "source": chunks[i]["source"],
        "page": chunks[i]["page"]
    } for i in top]

# -----------------------------
# Groq calls
# -----------------------------
def ask_groq_answer(question, retrieved):
    context = "\n\n".join(
        f"[{r['source']} | page {r['page']}]\n{r['text']}"
        for r in retrieved
    )
    prompt = f"""
Answer ONLY using the context.
If missing, say: "I couldn't find that in the uploaded documents."

Context:
{context}

Question:
{question}

Answer (include sources like file.pdf page X):
""".strip()

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=float(TEMPERATURE),
    )
    return res.choices[0].message.content

def ask_groq_summary(text):
    text = (text or "")[:6000]
    prompt = f"""
Summarize the document in 6–10 bullet points.
Add a short "Key Topics" list.

Text:
{text}
""".strip()

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return res.choices[0].message.content

def format_sources(retrieved):
    seen = set()
    out = []
    for r in retrieved:
        key = (r["source"], r["page"])
        if key not in seen:
            out.append(f"- **{r['source']}** (page {r['page']})")
            seen.add(key)
    return "### 📌 Sources\n" + ("\n".join(out) if out else "_No sources._")

# -----------------------------
# Actions
# -----------------------------
def process_pdfs(files):
    try:
        if not files:
            STATE["chunks"] = None
            STATE["embeds"] = None
            STATE["file_text"] = {}
            return ("⚠️ Upload PDFs first.", gr.update(choices=[], value=None), "")

        pages = []
        file_text = {}

        for f in files:
            p = extract_pdf_pages(f)
            pages.extend(p)
            fname = os.path.basename(getattr(f, "name", "uploaded.pdf"))
            file_text[fname] = " ".join(x["text"] for x in p[:6])

        chunks = build_chunks(pages)
        if not chunks:
            STATE["chunks"] = None
            STATE["embeds"] = None
            STATE["file_text"] = file_text
            names = list(file_text.keys())
            first = names[0] if names else None
            return ("⚠️ No readable text found (maybe scanned PDFs).",
                    gr.update(choices=names, value=first),
                    (file_text.get(first, "") if first else ""))

        texts = [c["text"] for c in chunks]
        embeds = embedder.encode(
            texts,
            batch_size=16,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        STATE["chunks"] = chunks
        STATE["embeds"] = embeds
        STATE["file_text"] = file_text

        names = list(file_text.keys())
        first = names[0] if names else None
        return (f"✅ Indexed {len(chunks)} chunks.", gr.update(choices=names, value=first),
                (file_text.get(first, "") if first else ""))

    except Exception as e:
        STATE["chunks"] = None
        STATE["embeds"] = None
        return (f"❌ Error: {type(e).__name__}: {e}", gr.update(choices=[], value=None), "")

def update_preview(name):
    return STATE["file_text"].get(name, "")

def summarize(name):
    if not name:
        return "⚠️ Select a PDF first."
    txt = STATE["file_text"].get(name, "")
    if not txt:
        return "⚠️ No text to summarize."
    return ask_groq_summary(txt)

# ✅ CHAT: messages-format history (dicts)
def chat(question, history):
    history = history or []

    if not question or not question.strip():
        return history, "### 📌 Sources\n_Type a question first._"

    if STATE["chunks"] is None or STATE["embeds"] is None:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": "Please upload PDFs and click **Build Knowledge Base** first."})
        return history, "### 📌 Sources\n_No sources._"

    retrieved = retrieve(question, STATE["chunks"], STATE["embeds"], TOP_K)
    answer = ask_groq_answer(question, retrieved)

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    return history, format_sources(retrieved)

def clear():
    return [], "### 📌 Sources\n_Cleared._"

# -----------------------------
# UI (Top-K + Temperature removed)
# -----------------------------
CUSTOM_CSS = """
:root{--bgA:#070a1a;--bgB:#0b1430;--card: rgba(15, 23, 42, 0.72);--stroke: rgba(148, 163, 184, 0.20);--shadow: 0 16px 50px rgba(0,0,0,0.45);}
.gradio-container{
  background:
    radial-gradient(1200px 700px at 12% 12%, rgba(124,58,237,0.30), transparent 55%),
    radial-gradient(1000px 650px at 88% 18%, rgba(6,182,212,0.22), transparent 58%),
    radial-gradient(900px 700px at 20% 92%, rgba(34,197,94,0.18), transparent 58%),
    radial-gradient(900px 650px at 90% 88%, rgba(249,115,22,0.14), transparent 58%),
    linear-gradient(180deg, var(--bgA), var(--bgB));
}
.panel{border:1px solid var(--stroke);border-radius:18px;background:var(--card);box-shadow:var(--shadow);}
.gr-button{
  border-radius:14px !important;
  border:1px solid rgba(255,255,255,0.12) !important;
  background:linear-gradient(90deg, rgba(124,58,237,0.92), rgba(6,182,212,0.78)) !important;
  color:white !important;
  font-weight:850 !important;
}
"""

with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"""
<div style="border:1px solid rgba(148,163,184,0.20); border-radius:18px; padding:14px 16px;
background:linear-gradient(90deg, rgba(124,58,237,0.28), rgba(6,182,212,0.18), rgba(34,197,94,0.14));
box-shadow:0 16px 50px rgba(0,0,0,0.45);">
  <h1 style="margin:0;">✨ RAG PDF Chatbot</h1>
  <p style="margin:6px 0 0 0; opacity:0.9;">
    Sentence-Transformers (<b>{EMBED_MODEL_NAME}</b>) • Preview + Summary • Fixed Top-K={TOP_K} • Temp={TEMPERATURE}
  </p>
</div>
        """
    )

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Group(elem_classes=["panel"]):
                gr.Markdown("## 📂 Upload & Preview")
                files = gr.File(file_count="multiple", file_types=[".pdf"], label="Upload PDFs")
                btn_index = gr.Button("📚 Build Knowledge Base")
                status = gr.Textbox(label="Status", interactive=False)

                picker = gr.Dropdown(label="Preview PDF", choices=[], value=None)
                preview = gr.Textbox(label="Preview (first pages)", lines=10)

                btn_summary = gr.Button("🧾 Generate Summary")
                summary = gr.Markdown()

        with gr.Column(scale=6):
            with gr.Group(elem_classes=["panel"]):
                gr.Markdown("## 💬 Chat")
                chatbot = gr.Chatbot(height=520)  # expects messages on your Space
                question = gr.Textbox(label="Ask a question")
                with gr.Row():
                    ask_btn = gr.Button("🚀 Ask")
                    clear_btn = gr.Button("🧹 Clear")

            sources = gr.Markdown("### 📌 Sources\nAsk a question to see sources.")

    btn_index.click(process_pdfs, inputs=files, outputs=[status, picker, preview])
    picker.change(update_preview, inputs=picker, outputs=preview)
    btn_summary.click(summarize, inputs=picker, outputs=summary)

    ask_btn.click(chat, inputs=[question, chatbot], outputs=[chatbot, sources])
    question.submit(chat, inputs=[question, chatbot], outputs=[chatbot, sources])
    clear_btn.click(clear, outputs=[chatbot, sources])

# HF stable launch + hide fd=-1 cleanup spam
if __name__ == "__main__":
    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        demo.queue().launch(ssr_mode=False)
