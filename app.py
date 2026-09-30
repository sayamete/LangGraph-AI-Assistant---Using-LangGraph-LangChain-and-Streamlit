import streamlit as st
import json
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver


# =========================================================
# STREAMLIT CONFIG
# =========================================================

st.set_page_config(
    page_title="LangGraph AI Assistant",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# LOAD NOTEBOOK AS BACKEND
# =========================================================

@st.cache_resource(show_spinner="Loading LangGraph backend...")
def load_backend():

    notebook_path = (
        Path(__file__).parent
        / "langgraph_project.ipynb"
    )

    if not notebook_path.exists():

        raise FileNotFoundError(
            f"""
Notebook not found:

{notebook_path}

Keep these two files in the same folder:

app.py
langgraph_project(4).ipynb
"""
        )

    # -----------------------------------------------------
    # Read notebook
    # -----------------------------------------------------

    with open(
        notebook_path,
        "r",
        encoding="utf-8"
    ) as f:

        notebook = json.load(f)


    # -----------------------------------------------------
    # Notebook namespace
    # -----------------------------------------------------

    namespace = {

        "__name__":
        "__langgraph_backend__",

        "__file__":
        str(notebook_path)

    }


    # -----------------------------------------------------
    # Execute notebook cells
    # -----------------------------------------------------

    for index, cell in enumerate(
        notebook.get("cells", [])
    ):

        if cell.get("cell_type") != "code":
            continue


        source = "".join(
            cell.get("source", [])
        )


        if not source.strip():
            continue


        # -------------------------------------------------
        # Skip notebook test cell
        # -------------------------------------------------

        if index == 18:
            continue


        exec(
            compile(
                source,
                f"{notebook_path}::cell_{index}",
                "exec"
            ),
            namespace
        )


    # -----------------------------------------------------
    # Get graph from notebook
    # -----------------------------------------------------

    graph = namespace.get("graph")


    if graph is None:

        raise RuntimeError(
            "The notebook did not create the LangGraph object `graph`."
        )


    # =====================================================
    # MEMORY
    # =====================================================

    memory = MemorySaver()


    # Compile notebook graph WITH MEMORY

    build = graph.compile(
        checkpointer=memory
    )


    return build


# =========================================================
# LOAD BACKEND
# =========================================================

try:

    build = load_backend()

except Exception as e:

    st.error(
        "❌ LangGraph backend could not be loaded."
    )

    st.exception(e)

    st.stop()


# =========================================================
# STREAMLIT SESSION
# =========================================================

# One thread_id = one conversation

if "thread_id" not in st.session_state:

    st.session_state.thread_id = str(
        uuid.uuid4()
    )


if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Controls")


    st.write(
        f"**Conversation ID:** "
        f"`{st.session_state.thread_id}`"
    )


    st.markdown("---")


    # -----------------------------------------------------
    # Clear Chat
    # -----------------------------------------------------

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        # Clear Streamlit UI

        st.session_state.chat_history = []


        # Create new conversation

        st.session_state.thread_id = str(
            uuid.uuid4()
        )


        st.rerun()


    st.markdown("---")


    st.markdown(
        """
### Backend

`langgraph_project(4).ipynb`

### Pipeline

- Hybrid Retrieval
- Chroma
- BM25
- RRF
- Groq LLM
- Tool Calling
- LangGraph
- Structured Output
- Conversation Memory
"""
    )


# =========================================================
# TITLE
# =========================================================

st.title(
    "🤖 LangGraph AI Assistant"
)

st.caption(
    "RAG + Tools + Structured Output + Conversation Memory"
)


# =========================================================
# SHOW OLD CHAT
# =========================================================

for item in st.session_state.chat_history:

    with st.chat_message(
        item["role"]
    ):

        # -------------------------------------------------
        # USER MESSAGE
        # -------------------------------------------------

        if item["role"] == "user":

            st.markdown(
                item["content"]
            )


        # -------------------------------------------------
        # ASSISTANT MESSAGE
        # -------------------------------------------------

        else:

            data = item.get(
                "data",
                {}
            )


            if isinstance(
                data,
                dict
            ):

                # -----------------------------
                # Answer
                # -----------------------------

                answer = data.get(
                    "answer",
                    ""
                )


                if answer:

                    st.markdown(
                        answer
                    )


                # -----------------------------
                # Summary
                # -----------------------------

                summary = data.get(
                    "summary",
                    ""
                )


                if summary:

                    st.caption(
                        f"Summary: {summary}"
                    )


                # -----------------------------
                # Sources
                # -----------------------------

                sources = data.get(
                    "sources",
                    []
                )


                if sources:

                    st.markdown(
                        "**Sources:**"
                    )


                    for source in sources:

                        st.write(
                            f"- {source}"
                        )


            else:

                st.markdown(
                    str(data)
                )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask your question..."
)


if question:

    # =====================================================
    # SHOW USER QUESTION
    # =====================================================

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )


    with st.chat_message("user"):

        st.markdown(
            question
        )


    # =====================================================
    # RUN LANGGRAPH
    # =====================================================

    with st.chat_message("assistant"):

        with st.spinner(
            "Thinking..."
        ):

            try:

                # -------------------------------------------------
                # IMPORTANT
                #
                # Your current State uses:
                #
                # messages
                #
                # not:
                #
                # message
                # -------------------------------------------------

                input_state = {

                    "messages": [

                        HumanMessage(
                            content=question
                        )

                    ]

                }


                # =================================================
                # MEMORY CONFIG
                # =================================================

                config = {

                    "configurable": {

                        "thread_id":
                        st.session_state.thread_id

                    }

                }


                # =================================================
                # INVOKE GRAPH
                # =================================================

                response = build.invoke(
                    input_state,
                    config=config
                )


                # =================================================
                # GET FINAL ANSWER
                # =================================================

                final_answer = response.get(
                    "final_answer"
                )


                if final_answer is None:

                    st.error(
                        "❌ Graph did not return final_answer."
                    )


                else:

                    # -------------------------------------------------
                    # Pydantic StructuredOutput
                    # -------------------------------------------------

                    if hasattr(
                        final_answer,
                        "model_dump"
                    ):

                        data = final_answer.model_dump()


                    elif isinstance(
                        final_answer,
                        dict
                    ):

                        data = final_answer


                    else:

                        data = {

                            "answer":
                            str(final_answer),

                            "sources": [],

                            "summary": ""

                        }


                    # =================================================
                    # ANSWER
                    # =================================================

                    answer = data.get(
                        "answer",
                        ""
                    )


                    if answer:

                        st.markdown(
                            answer
                        )


                    # =================================================
                    # SUMMARY
                    # =================================================

                    summary = data.get(
                        "summary",
                        ""
                    )


                    if summary:

                        st.caption(
                            f"Summary: {summary}"
                        )


                    # =================================================
                    # SOURCES
                    # =================================================

                    sources = data.get(
                        "sources",
                        []
                    )


                    if sources:

                        st.markdown(
                            "**Sources:**"
                        )


                        for source in sources:

                            st.write(
                                f"- {source}"
                            )


                    # =================================================
                    # SAVE UI HISTORY
                    # =================================================

                    st.session_state.chat_history.append(
                        {

                            "role":
                            "assistant",

                            "data":
                            data

                        }
                    )


            except Exception as e:

                st.error(
                    "❌ Something went wrong while running LangGraph."
                )

                st.exception(e)