import gradio as gr
import requests
import json
from typing import Optional

TITLE = " Graph RAG System"
BACKEND_URL = "http://localhost:8000"

def chat_send(message, history, rag_state):
    history = history or []
    message = (message or "").strip()
    if not message:
        return history, "", ""

    try:
        response = requests.post(
            f"{BACKEND_URL}/query",
            json={
                "query": message,
                "top_k": rag_state.get("top_k", 5),
                "temperature": rag_state.get("temperature", 0.2),
                "model": rag_state.get("model", "openai/gpt-4o-mini"),
                "retrieval_mode": rag_state.get("retrieval_mode", "graph_rag"),
                "use_entity_context": rag_state.get("use_entity_context", True)
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data["answer"]
            
            # Format sources with better styling
            sources = data.get("sources", [])
            scores = data.get("similarity_scores", [])
            metadata = data.get("metadata", {})
            
            if sources:
                sources_md = "##  Retrieved Sources\n\n"
                for i, (src, score) in enumerate(zip(sources, scores), 1):
                    truncated = src[:200] + "..." if len(src) > 200 else src
                    sources_md += f"**[{i}]** ({score:.1%} relevance)\n> {truncated}\n\n"
            else:
                sources_md = ""
            
            # Format metadata
            if metadata:
                metadata_md = "##  Retrieval Metadata\n\n"
                metadata_md += f"- **Method**: {metadata.get('method', 'N/A')}\n"
                
                if 'query_entities' in metadata:
                    metadata_md += f"- **Query Entities**: {metadata['query_entities']}\n"
                
                if 'graph_stats' in metadata:
                    stats = metadata['graph_stats']
                    metadata_md += f"- **Graph Entities**: {stats.get('num_entities', 0)}\n"
                    metadata_md += f"- **Graph Relations**: {stats.get('num_relations', 0)}\n"
                
                if 'vector_weight' in metadata:
                    metadata_md += f"- **Vector Weight**: {metadata['vector_weight']:.2f}\n"
                    metadata_md += f"- **Graph Weight**: {metadata['graph_weight']:.2f}\n"
            else:
                metadata_md = ""
        else:
            answer = f" Backend error: {response.status_code}"
            sources_md = ""
            metadata_md = ""
    
    except requests.exceptions.ConnectionError:
        answer = " Cannot connect to backend at http://localhost:8000\n\nMake sure backend is running: `uv run backend`"
        sources_md = ""
        metadata_md = ""
    except Exception as e:
        answer = f" Error: {str(e)}"
        sources_md = ""
        metadata_md = ""
    
    # Add messages to history
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    
    return history, sources_md, metadata_md

def clear_chat():
    return [], "", ""

def get_system_stats():
    """Fetch system statistics"""
    try:
        response = requests.get(f"{BACKEND_URL}/stats", timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            stats_md = "##  System Statistics\n\n"
            stats_md += f"**Status**: {data.get('status', 'unknown')}\n\n"
            stats_md += f"**Documents**: {data.get('num_documents', 0)}\n\n"
            
            if data.get('graph_stats'):
                gs = data['graph_stats']
                stats_md += "### Knowledge Graph\n"
                stats_md += f"- Entities: {gs.get('num_entities', 0)}\n"
                stats_md += f"- Relations: {gs.get('num_relations', 0)}\n"
                stats_md += f"- Graph Nodes: {gs.get('num_nodes', 0)}\n"
                stats_md += f"- Graph Edges: {gs.get('num_edges', 0)}\n"
                stats_md += f"- Avg Degree: {gs.get('avg_degree', 0):.2f}\n\n"
            
            if data.get('vector_stats'):
                vs = data['vector_stats']
                stats_md += "### Vector Store\n"
                stats_md += f"- Documents: {vs.get('num_documents', 0)}\n"
                stats_md += f"- Embedding Dim: {vs.get('embedding_dimension', 0)}\n"
                stats_md += f"- Size: {vs.get('total_size_mb', 0):.2f} MB\n"
            
            return stats_md
        else:
            return f" Failed to fetch stats: {response.status_code}"
    except Exception as e:
        return f" Error: {str(e)}"

CSS = r"""
* { box-sizing: border-box; }

.gradio-container { 
    max-width: 100% !important;
    padding: 0 !important;
}

body { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
}

#main_col { 
    padding: 20px;
    max-width: 1200px;
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
    height: calc(100vh - 400px) !important;
    min-height: 400px !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1) !important;
}

#chatbot .message {
    border-radius: 12px;
    padding: 12px 16px;
}

#chatbot .message.user {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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

/* Info panels */
#sources, #metadata {
    background: rgba(255,255,255,0.95);
    border-radius: 8px;
    padding: 16px;
    margin-top: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    color: #333 !important;
    max-height: 300px;
    overflow-y: auto;
}

#sources {
    border-left: 4px solid #48bb78;
}

#metadata {
    border-left: 4px solid #4299e1;
}

#sources *, #metadata * {
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
#send_btn, #clear_btn, #stats_btn {
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
}

#clear_btn:hover {
    background: #ff6b6b !important;
    color: white !important;
}

#stats_btn {
    background: white !important;
    border: 2px solid #4299e1 !important;
    color: #4299e1 !important;
}

#stats_btn:hover {
    background: #4299e1 !important;
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
    width: 340px;
    max-width: calc(100vw - 28px);
    max-height: calc(100vh - 100px);
    overflow-y: auto;
    padding: 20px;
    background: white;
    border-radius: 12px;
    box-shadow: 0 12px 48px rgba(0,0,0,0.2);
    border: 1px solid #e2e8f0;
}

#close_settings_btn { 
    height: 40px !important;
    width: 100% !important;
    margin-top: 12px;
}

/* Sliders and inputs in settings */
#settings_panel input,
#settings_panel .gr-slider {
    border-radius: 6px !important;
}

/* Stats modal */
#stats_modal {
    background: rgba(255,255,255,0.98);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
}
"""

with gr.Blocks(title=TITLE, css=CSS) as demo:
    rag_state = gr.State({
        "top_k": 5,
        "temperature": 0.2,
        "model": "openai/gpt-4o-mini",
        "retrieval_mode": "graph_rag",
        "use_entity_context": True
    })
    settings_open = gr.State(False)

    settings_btn = gr.Button("⚙️", elem_id="settings_btn", scale=0)

    with gr.Column(visible=False, elem_id="settings_panel") as settings_panel:
        gr.Markdown("### ⚙️ Graph RAG Settings")
        
        top_k = gr.Slider(1, 20, value=5, step=1, label="📊 Context Size (Top-K)")
        temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="🌡️ Temperature")
        model_name = gr.Textbox(value="openai/gpt-4o-mini", label="🤖 Model")
        
        retrieval_mode = gr.Radio(
            choices=["graph_rag", "vector", "hybrid"],
            value="graph_rag",
            label=" Retrieval Mode",
            info="graph_rag: Graph+Vector, vector: Vector only, hybrid: Full hybrid"
        )
        
        use_entity_context = gr.Checkbox(
            value=True,
            label=" Use Entity Context",
            info="Include knowledge graph entity relationships"
        )
        
        close_settings_btn = gr.Button("✕ Close", elem_id="close_settings_btn")

    with gr.Column(elem_id="main_col"):
        gr.Markdown(f"# {TITLE}", elem_id="title_md")
        gr.Markdown(
            "Advanced RAG system with **Knowledge Graph** integration. "
            "Combines vector similarity with entity relationships for better retrieval.",
            elem_id="subtitle"
        )

        chatbot = gr.Chatbot(
            label="Conversation",
            elem_id="chatbot",
            height=500
        )
        
        with gr.Row():
            sources = gr.Markdown("", label="Sources", elem_id="sources")
            metadata = gr.Markdown("", label="Metadata", elem_id="metadata")

        msg = gr.Textbox(
            label=" Your Question",
            placeholder="Ask about RAG, Graph RAG, or AI... (Enter to send, Shift+Enter for new line)",
            elem_id="msg",
            lines=2,
            max_lines=5,
        )

        with gr.Row():
            send_btn = gr.Button(" Send", variant="primary", elem_id="send_btn", scale=3)
            clear_btn = gr.Button(" Clear", elem_id="clear_btn", scale=1)
            stats_btn = gr.Button(" Stats", elem_id="stats_btn", scale=1)

    # Settings toggle
    def toggle_settings(is_open):
        is_open = not bool(is_open)
        return gr.update(visible=is_open), is_open

    settings_btn.click(toggle_settings, inputs=[settings_open], outputs=[settings_panel, settings_open])
    close_settings_btn.click(lambda: (gr.update(visible=False), False), outputs=[settings_panel, settings_open])

    # Update state
    def update_state(top_k, temperature, model, mode, use_entities, state):
        state = dict(state or {})
        state["top_k"] = int(top_k)
        state["temperature"] = float(temperature)
        state["model"] = model
        state["retrieval_mode"] = mode
        state["use_entity_context"] = use_entities
        return state

    for component in [top_k, temperature, model_name, retrieval_mode, use_entity_context]:
        component.change(
            update_state,
            inputs=[top_k, temperature, model_name, retrieval_mode, use_entity_context, rag_state],
            outputs=[rag_state]
        )

    # Chat interactions
    send_btn.click(
        chat_send,
        inputs=[msg, chatbot, rag_state],
        outputs=[chatbot, sources, metadata]
    ).then(lambda: "", None, msg)
    
    msg.submit(
        chat_send,
        inputs=[msg, chatbot, rag_state],
        outputs=[chatbot, sources, metadata]
    ).then(lambda: "", None, msg)
    
    clear_btn.click(clear_chat, outputs=[chatbot, sources, metadata])
    
    # Stats button
    stats_btn.click(
        get_system_stats,
        outputs=[metadata]
    )

def main():
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft()
    )

if __name__ == "__main__":
    main()
