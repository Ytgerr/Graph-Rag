import gradio as gr
import requests

TITLE = "RAG System: Graph RAG vs Vector RAG"
BACKEND_URL = "http://localhost:8002"


def chat_send(message, graph_history, vector_history, rag_state):
    graph_history = graph_history or []
    vector_history = vector_history or []
    message = (message or "").strip()
    if not message:
        return graph_history, vector_history, "", "", ""

    graph_history.append({"role": "user", "content": message})
    vector_history.append({"role": "user", "content": message})

    try:
        response = requests.post(
            f"{BACKEND_URL}/query/dual",
            json={
                "query": message,
                "top_k": rag_state.get("top_k", 5),
                "temperature": rag_state.get("temperature", 0.2),
                "model": rag_state.get("model", "openai/gpt-4o-mini"),
            },
            timeout=120,
        )

        if response.status_code == 200:
            data = response.json()

            graph_data = data["graph"]
            vector_data = data["vector"]

            graph_answer = graph_data["answer"]
            vector_answer = vector_data["answer"]

            graph_sources = graph_data.get("sources", [])
            graph_scores = graph_data.get("similarity_scores", [])
            vector_sources = vector_data.get("sources", [])
            vector_scores = vector_data.get("similarity_scores", [])

            graph_sources_md = _format_sources(graph_sources, graph_scores)
            vector_sources_md = _format_sources(vector_sources, vector_scores)
        else:
            graph_answer = f"Backend error: {response.status_code}"
            vector_answer = f"Backend error: {response.status_code}"
            graph_sources_md = ""
            vector_sources_md = ""

    except requests.exceptions.ConnectionError:
        err = "Cannot connect to backend. Make sure it is running: `uv run backend`"
        graph_answer = err
        vector_answer = err
        graph_sources_md = ""
        vector_sources_md = ""
    except Exception as e:
        graph_answer = f"Error: {str(e)}"
        vector_answer = f"Error: {str(e)}"
        graph_sources_md = ""
        vector_sources_md = ""

    graph_history.append({"role": "assistant", "content": graph_answer})
    vector_history.append({"role": "assistant", "content": vector_answer})

    return graph_history, vector_history, graph_sources_md, vector_sources_md, ""


def _format_sources(sources, scores):
    if not sources:
        return ""
    md = ""
    for i, (src, score) in enumerate(zip(sources, scores), 1):
        truncated = src[:200] + "..." if len(src) > 200 else src
        md += f"**[{i}]** ({score:.1%})\n> {truncated}\n\n"
    return md


def clear_chat():
    return [], [], "", ""


def get_system_stats():
    try:
        response = requests.get(f"{BACKEND_URL}/stats", timeout=10)
        if response.status_code == 200:
            data = response.json()
            md = "## System Statistics\n\n"
            md += f"**Status**: {data.get('status', 'unknown')}\n\n"
            md += f"**Documents**: {data.get('num_documents', 0)}\n\n"

            if data.get("graph_stats"):
                gs = data["graph_stats"]
                md += "### Graph RAG\n"
                md += f"- Entities: {gs.get('num_entities', 0)}\n"
                md += f"- Relations: {gs.get('num_relations', 0)}\n"
                md += f"- Documents: {gs.get('num_documents', 0)}\n\n"

            if data.get("vector_stats"):
                vs = data["vector_stats"]
                md += "### Vector RAG\n"
                md += f"- Embedding Model: {vs.get('embedding_model', 'N/A')}\n"
                md += f"- Chunks: {vs.get('num_documents', 0)}\n"

            return md
        else:
            return f"Failed to fetch stats: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"


def start_wiki_collection(topic, lang, limit, expand):
    topic = (topic or "").strip()
    if not topic:
        return "Enter a topic to collect"

    try:
        resp = requests.post(
            f"{BACKEND_URL}/collect-wiki",
            json={
                "topic": topic,
                "lang": lang,
                "limit": int(limit),
                "expand": bool(expand),
            },
            timeout=10,
        )
        if resp.status_code == 200:
            expand_label = " + LLM expansion" if expand else ""
            return f"Started: **{topic}** ({lang}, limit {int(limit)}{expand_label})"
        elif resp.status_code == 409:
            return "Collection already in progress, please wait"
        else:
            return f"Error: {resp.text}"
    except requests.exceptions.ConnectionError:
        return "Backend unavailable"
    except Exception as e:
        return f"Error: {e}"


def poll_wiki_status():
    try:
        resp = requests.get(f"{BACKEND_URL}/collect-wiki/status", timeout=5)
        if resp.status_code != 200:
            return "Failed to get status"

        d = resp.json()
        status = d.get("status", "idle")
        topic = d.get("topic", "")
        total = d.get("total", 0)
        current = d.get("current", 0)
        collected = d.get("collected", 0)
        error = d.get("error", "")
        subtopics = d.get("subtopics", [])

        if status == "idle":
            return "No active collection"
        elif status == "starting":
            return f"Starting: **{topic}**..."
        elif status == "expanding":
            return f"LLM generating subtopics for **{topic}**..."
        elif status == "discovering":
            msg = f"Discovering pages for **{topic}**..."
            if subtopics:
                msg += f"\n\nSubtopics ({len(subtopics)}): " + ", ".join(subtopics[:8])
                if len(subtopics) > 8:
                    msg += f" ... (+{len(subtopics) - 8})"
            return msg
        elif status == "collecting":
            pct = (current / total * 100) if total > 0 else 0
            msg = f"Collecting: **{topic}** -- {current}/{total} ({pct:.0f}%), articles: {collected}"
            if subtopics:
                msg += f"\n\nSubtopics: {len(subtopics)}"
            return msg
        elif status == "reloading":
            return f"Reloading RAG system ({collected} articles)..."
        elif status == "done":
            msg = f"Done: **{topic}** -- {collected} articles collected, RAG reloaded"
            if subtopics:
                msg += f"\n\nSubtopics used: {len(subtopics)}"
            return msg
        elif status == "error":
            return f"Error: {error}"
        else:
            return f"Status: {status}"

    except Exception as e:
        return f"Error: {e}"


CSS = r"""
* { box-sizing: border-box; }
.gradio-container { max-width: 100% !important; padding: 0 !important; }
body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
}
#main_col { padding: 20px; max-width: 1400px; margin: 0 auto; }
#title_md { color: white; text-shadow: 0 2px 4px rgba(0,0,0,0.2); margin-bottom: 8px; }
#subtitle { color: rgba(255,255,255,0.85); margin-bottom: 16px; }

/* --- Chat columns --- */
.chat-col-header {
    text-align: center; font-weight: 700; font-size: 16px;
    padding: 10px 0; margin-bottom: 4px;
    border-radius: 8px 8px 0 0;
    color: white; text-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.graph-header { background: linear-gradient(135deg, #38b2ac 0%, #319795 100%); }
.vector-header { background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%); }

#graph_chatbot, #vector_chatbot {
    height: calc(100vh - 460px) !important; min-height: 350px !important;
    border-radius: 0 0 12px 12px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1) !important;
}
#graph_chatbot .message.user, #vector_chatbot .message.user {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; margin-left: 20px; border-radius: 12px 0 12px 12px;
}
#graph_chatbot .message.assistant {
    background: white; color: #333; margin-right: 20px;
    border-radius: 0 12px 12px 12px; border-left: 4px solid #38b2ac;
}
#vector_chatbot .message.assistant {
    background: white; color: #333; margin-right: 20px;
    border-radius: 0 12px 12px 12px; border-left: 4px solid #ed8936;
}

/* --- Sources --- */
#graph_sources, #vector_sources {
    background: rgba(255,255,255,0.95); border-radius: 8px; padding: 12px 16px;
    margin-top: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    color: #333 !important; max-height: 220px; overflow-y: auto;
}
#graph_sources { border-left: 4px solid #38b2ac; }
#vector_sources { border-left: 4px solid #ed8936; }
#graph_sources *, #vector_sources * { color: #333 !important; }

/* --- Input & buttons --- */
#msg textarea {
    font-size: 15px !important; min-height: 60px !important;
    line-height: 1.4 !important; border-radius: 8px !important;
}
#send_btn, #clear_btn, #stats_btn {
    height: 48px !important; font-size: 15px !important; font-weight: 600;
    border-radius: 8px !important; transition: all 0.3s ease;
}
#send_btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
}
#send_btn:hover { transform: translateY(-2px); box-shadow: 0 8px 16px rgba(102,126,234,0.4) !important; }
#clear_btn { background: white !important; border: 2px solid #ff6b6b !important; color: #ff6b6b !important; }
#clear_btn:hover { background: #ff6b6b !important; color: white !important; }
#stats_btn { background: white !important; border: 2px solid #4299e1 !important; color: #4299e1 !important; }
#stats_btn:hover { background: #4299e1 !important; color: white !important; }

/* --- Stats --- */
#stats_output {
    background: rgba(255,255,255,0.95); border-radius: 8px; padding: 16px;
    margin-top: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    color: #333 !important; border-left: 4px solid #4299e1;
}
#stats_output * { color: #333 !important; }

/* --- Wiki status --- */
#wiki_status {
    background: rgba(255,255,255,0.95); border-radius: 8px; padding: 12px 16px;
    border-left: 4px solid #ed8936; color: #333 !important; margin-top: 8px;
}
#wiki_status * { color: #333 !important; }

/* --- Settings tab buttons --- */
#collect_btn {
    height: 44px !important; font-size: 14px !important; font-weight: 600;
    border-radius: 8px !important;
    background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%) !important;
    border: none !important; color: white !important;
}
#collect_btn:hover { transform: translateY(-2px); box-shadow: 0 8px 16px rgba(237,137,54,0.4) !important; }
#refresh_status_btn {
    height: 44px !important; font-size: 14px !important;
    background: white !important; border: 2px solid #ed8936 !important; color: #ed8936 !important;
    border-radius: 8px !important;
}
#refresh_status_btn:hover { background: #ed8936 !important; color: white !important; }
"""

with gr.Blocks(title=TITLE, css=CSS) as demo:
    rag_state = gr.State(
        {
            "top_k": 5,
            "temperature": 0.2,
            "model": "openai/gpt-4o-mini",
        }
    )

    with gr.Column(elem_id="main_col"):
        gr.Markdown(f"# {TITLE}", elem_id="title_md")
        gr.Markdown(
            "Both **Graph RAG** and **Vector RAG** answer simultaneously -- compare results side by side.",
            elem_id="subtitle",
        )

        with gr.Row(equal_height=True):
            # -- Graph RAG column --
            with gr.Column(scale=1):
                gr.HTML('<div class="chat-col-header graph-header">Graph RAG</div>')
                graph_chatbot = gr.Chatbot(
                    elem_id="graph_chatbot",
                    height=450,
                    show_label=False,
                )
                graph_sources = gr.Markdown("", elem_id="graph_sources")

            # -- Vector RAG column --
            with gr.Column(scale=1):
                gr.HTML('<div class="chat-col-header vector-header">Vector RAG</div>')
                vector_chatbot = gr.Chatbot(
                    elem_id="vector_chatbot",
                    height=450,
                    show_label=False,
                )
                vector_sources = gr.Markdown("", elem_id="vector_sources")

        # -- Input area --
        msg = gr.Textbox(
            label="Your Question",
            placeholder="Ask anything -- both RAG systems will answer simultaneously (Enter to send)",
            elem_id="msg",
            lines=2,
            max_lines=5,
        )

        with gr.Row():
            send_btn = gr.Button("Send", variant="primary", elem_id="send_btn", scale=3)
            clear_btn = gr.Button("Clear", elem_id="clear_btn", scale=1)
            stats_btn = gr.Button("Stats", elem_id="stats_btn", scale=1)

        stats_output = gr.Markdown("", elem_id="stats_output", visible=False)

        # -- Settings as Accordion (always accessible, no broken overlay) --
        with gr.Accordion("Settings", open=False):
            with gr.Group():
                gr.Markdown("### Model")
                model_name = gr.Textbox(
                    value="openai/gpt-4o-mini",
                    label="LLM Model (OpenRouter)",
                )
                temperature = gr.Slider(
                    0.0, 1.0, value=0.2, step=0.05, label="Temperature"
                )

            with gr.Group():
                gr.Markdown("### Retrieval")
                top_k = gr.Slider(1, 20, value=5, step=1, label="Top-K Results")

            with gr.Group():
                gr.Markdown("### Wiki Dataset Collection")
                wiki_topic = gr.Textbox(
                    value="",
                    label="Topic",
                    placeholder="Moscow, AI, Physics...",
                )
                wiki_lang = gr.Radio(
                    choices=["en", "ru"], value="en", label="Wikipedia Language"
                )
                wiki_limit = gr.Slider(
                    10, 500, value=100, step=10, label="Max Documents"
                )
                wiki_expand = gr.Checkbox(
                    value=True,
                    label="LLM Topic Expansion",
                    info="LLM generates sub-topics for broader coverage",
                )
                with gr.Row():
                    collect_btn = gr.Button("Collect Dataset", elem_id="collect_btn", scale=2)
                    refresh_status_btn = gr.Button("Refresh Status", elem_id="refresh_status_btn", scale=1)
                wiki_status = gr.Markdown("No active collection", elem_id="wiki_status")

    # -- Event handlers --

    def update_state(top_k_val, temp_val, model_val, state):
        state = dict(state or {})
        state["top_k"] = int(top_k_val)
        state["temperature"] = float(temp_val)
        state["model"] = model_val
        return state

    for component in [top_k, temperature, model_name]:
        component.change(
            update_state,
            inputs=[top_k, temperature, model_name, rag_state],
            outputs=[rag_state],
        )

    send_btn.click(
        chat_send,
        inputs=[msg, graph_chatbot, vector_chatbot, rag_state],
        outputs=[graph_chatbot, vector_chatbot, graph_sources, vector_sources, msg],
    )
    msg.submit(
        chat_send,
        inputs=[msg, graph_chatbot, vector_chatbot, rag_state],
        outputs=[graph_chatbot, vector_chatbot, graph_sources, vector_sources, msg],
    )
    clear_btn.click(
        clear_chat,
        outputs=[graph_chatbot, vector_chatbot, graph_sources, vector_sources],
    )

    def show_stats():
        md = get_system_stats()
        return gr.update(value=md, visible=True)

    stats_btn.click(show_stats, outputs=[stats_output])

    collect_btn.click(
        start_wiki_collection,
        inputs=[wiki_topic, wiki_lang, wiki_limit, wiki_expand],
        outputs=[wiki_status],
    )
    refresh_status_btn.click(poll_wiki_status, outputs=[wiki_status])


def main():
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
