import os
from typing import List, Dict, Any, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from IPython.display import display, Image  # Für Jupyter-Notebooks

class FinanceState(TypedDict, total=False):
    """Gemeinsamer Zustand für alle Finanzagenten."""
    frage: str
    antwort: str

class FinanzScrapAgent:
    
    def __init__(self) -> None:

        load_dotenv()

        self._llm = self._init_llm()

        self._graph = self._init_graph()

    def _init_llm(self) -> ChatOllama:

        provider = (os.getenv("LLM_PROVIDER") or "ollama").lower()
        base_url = os.getenv("LLM_BASE_URL") or "http://localhost:11434"
        model = os.getenv("LLM_MODEL") or "llama3.2:3b"

        return ChatOllama(model=model, base_url=base_url, temperature= 0.7)
    
    def ask_llm(self, state: FinanceState = {}) -> FinanceState:

        question = (state.get("frage") or "").strip()
        if not question:
            raise ValueError("Die Frage darf nicht leer sein.")
        print(state)
        messages = [
            ("system", "Du bist ein knapper Finance-Assistant. Antworte kurz und klar."),
            ("human", question),
        ]
        ai_msg = self._llm.invoke(messages)

        return {"antwort": ai_msg.content.strip()}
    
    
    def _init_graph(self):

        graph = StateGraph(FinanceState)

        graph.add_node("ask_llm", self.ask_llm)
        graph.set_entry_point("ask_llm")
        graph.set_finish_point("ask_llm")

        app = graph.compile()

        png_data = app.get_graph().draw_mermaid()

        with open("graph.mmd", "w") as f:
            f.write(png_data)

        return app
    
    def invoke(self, frage: str) -> str:

        return self._graph.invoke({"frage":frage})
    
if __name__ == "__main__":
    agent = FinanzScrapAgent()
    result = agent.invoke("Was sind die aktuellen Trends im Finanzsektor?")
    print(result["antwort"])
    
    

        

        


