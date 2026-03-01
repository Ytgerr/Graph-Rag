import gradio as gr
import requests
import json
from typing import Optional

TITLE = "RAG System"
BACKEND_URL = "http://localhost:8000"

def chat_send(message, history, rag_state):
    history = history or []
    message = (message or "").strip()
    if not message:
        return history, ""

    try:
        response = requests.post(
            f"{BACKEND_URL}/query",
            json={
                "query": message,
                "top_k": rag_state.get("top_k", 5),
                "temperature": rag_state.get("temperature", 0.2),
                "model": rag_state.get("model", "openai/gpt-4o-mini"),
                "similarity_type": rag_state.get("similarity_type", "enhanced")
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data["answer"]
            
            # Format sources with better styling
            sources = data.get("sources", [])
            scores = data.get("similarity_scores", [])
            
            if sources:
                sources_md = "##  Sources\n\n"
                for i, (src, score) in enumerate(zip(sources, scores), 1):
                    truncated = src[:150] + "..." if len(src) > 150 else src
                    sources_md += f"**[{i}]** ({score:.1%} match) {truncated}\n\n"
            else:
                sources_md = ""
        else:
            answer = f"❌ Backend error: {response.status_code}"
            sources_md = ""
    
    except requests.exceptions.ConnectionError:
        answer = "❌ Cannot connect to backend at http://localhost:8000\n\nMake sure backend is running: `uv run backend`"
        sources_md = ""
    except Exception as e:
        answer = f"❌ Error: {str(e)}"
        sources_md = ""
    
    # Add messages to history in the new format
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    
    return history, sources_md

def clear_chat():
    return [], ""

CSS = r"""
* { box-sizing: border-box; }

.gradio-container { 
    max-width: 100% !important;
    padding: 0 !important;
}

body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }

#main_col { 
    padding: 20px;
    max-width: 1000px;
    margin: 0 auto;
}

#title_md { 
    padding-left: 64px;
    color: white;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    margin-bottom: 20px;
}

/* Chatbot styling */
#chatbot { 
    height: calc(100vh - 320px) !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1) !important;
}

#chatbot .message {
    border-radius: 12px;
    padding: 12px 16px;
}

#chatbot .message.user {
    background: #667eea;
    color: white;
    margin-left: 40px;
    border-radius: 12px 0 12px 12px;
}

#chatbot .message.assistant {
    background: white;
    color: #333;
    margin-right: 40px;
    border-radius: 0 12px 12px 12px;
    border-left: 4px solid #667eea;
}

/* Sources panel */
#sources {
    background: rgba(255,255,255,0.95);
    border-radius: 8px;
    padding: 16px;
    margin-top: 12px;
    border-left: 4px solid #48bb78;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    color: #333 !important;
}

#sources * {
    color: #333 !important;
}

/* Input area */
#msg textarea {
    font-size: 15px !important;
    min-height: 60px !important;
    line-height: 1.4 !important;
    border-radius: 8px !important;
}

/* Buttons */
#send_btn, #clear_btn {
    height: 48px !important;
    font-size: 15px !important;
    font-weight: 600;
    border-radius: 8px !important;
    transition: all 0.3s ease;
}

#send_btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
}

#send_btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4) !important;
}

#clear_btn {
    background: white !important;
    border: 2px solid #ff6b6b !important;
    color: #ff6b6b !important;
    font-weight: 600;
}

#clear_btn:hover {
    background: #ff6b6b !important;
    color: white !important;
}

/* Settings button */
#settings_btn {
    position: fixed;
    left: 14px;
    top: 18px;
    z-index: 10010;
    width: 48px;
    height: 48px;
    min-width: 48px !important;
    padding: 0 !important;
    border-radius: 50% !important;
    font-size: 20px !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    transition: all 0.3s ease;
}

#settings_btn:hover {
    transform: scale(1.1) rotate(15deg);
    box-shadow: 0 8px 16px rgba(102, 126, 234, 0.6);
}

/* Settings panel */
#settings_panel {
    position: fixed;
    left: 14px;
    top: 80px;
    z-index: 10020;
    width: 320px;
    max-width: calc(100vw - 28px);
    max-height: calc(100vh - 100px);
    overflow-y: auto;
    padding: 16px;
    background: white;
    border-radius: 12px;
    box-shadow: 0 12px 48px rgba(0,0,0,0.2);
    border: 1px solid #e2e8f0;
}

#close_settings_btn { 
    height: 40px !important;
    width: 100% !important;
}

/* Sliders and inputs in settings */
#settings_panel input,
#settings_panel .gr-slider {
    border-radius: 6px !important;
}
"""

with gr.Blocks(title=TITLE) as demo:
    rag_state = gr.State({"top_k": 5, "temperature": 0.2, "model": "openai/gpt-4o-mini", "similarity_type": "enhanced"})
    settings_open = gr.State(False)

    settings_btn = gr.Button("⚙️", elem_id="settings_btn", scale=0)

    with gr.Column(visible=False, elem_id="settings_panel") as settings_panel:
        gr.Markdown("### ⚙️ Settings")
        top_k = gr.Slider(1, 20, value=5, step=1, label=" Context (Top-K)")
        temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label=" Temperature")
        model_name = gr.Textbox(value="openai/gpt-4o-mini", label=" Model")
        similarity_type = gr.Radio(
            choices=["enhanced", "cosine"],
            value="enhanced",
            label=" Similarity Type"
        )
        close_settings_btn = gr.Button("✕ Close", elem_id="close_settings_btn")

    with gr.Column(elem_id="main_col"):
        gr.Markdown(f"#  {TITLE}", elem_id="title_md")

        chatbot = gr.Chatbot(
            label="Conversation",
            elem_id="chatbot"
        )
        
        sources = gr.Markdown("", label="Sources", elem_id="sources")

        msg = gr.Textbox(
            label=" Your Question",
            placeholder="Ask anything... (Enter to send, Shift+Enter for new line)",
            elem_id="msg",
            lines=2,
            max_lines=5,
        )

        with gr.Row():
            send_btn = gr.Button(" Send", variant="primary", elem_id="send_btn", scale=2)
            clear_btn = gr.Button(" Clear", elem_id="clear_btn", scale=1)

    def toggle_settings(is_open):
        is_open = not bool(is_open)
        return gr.update(visible=is_open), is_open

    settings_btn.click(toggle_settings, inputs=[settings_open], outputs=[settings_panel, settings_open])
    close_settings_btn.click(lambda: (gr.update(visible=False), False), outputs=[settings_panel, settings_open])

    def update_state(top_k, temperature, model, sim_type, state):
        state = dict(state or {})
        state["top_k"] = int(top_k)
        state["temperature"] = float(temperature)
        state["model"] = model
        state["similarity_type"] = sim_type
        return state

    top_k.change(update_state, inputs=[top_k, temperature, model_name, similarity_type, rag_state], outputs=[rag_state])
    temperature.change(update_state, inputs=[top_k, temperature, model_name, similarity_type, rag_state], outputs=[rag_state])
    model_name.change(update_state, inputs=[top_k, temperature, model_name, similarity_type, rag_state], outputs=[rag_state])
    similarity_type.change(update_state, inputs=[top_k, temperature, model_name, similarity_type, rag_state], outputs=[rag_state])

    send_btn.click(chat_send, inputs=[msg, chatbot, rag_state], outputs=[chatbot, sources]).then(lambda: "", None, msg)
    msg.submit(chat_send, inputs=[msg, chatbot, rag_state], outputs=[chatbot, sources]).then(lambda: "", None, msg)
    clear_btn.click(clear_chat, outputs=[chatbot, sources])

def main():
    demo.launch(server_name="0.0.0.0", server_port=7860, css=CSS, theme=gr.themes.Soft())