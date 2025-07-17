import threading
import yaml
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableMap
from operator import itemgetter
import re

from dotenv import load_dotenv

load_dotenv()

class LLMRunner:
    def __init__(self, prompt_path, model, command_queue=None):
        self.command_queue = command_queue
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_str = yaml.safe_load(f)["template"]
        self.prompt = PromptTemplate.from_template(prompt_str)
        self.llm = ChatOpenAI(model=model, streaming=False, temperature=0)
        self.chat_store = {}
        self.chat_history_limit = 10

        chain = (
            RunnableMap({
                "chat_history": itemgetter("chat_history"),
                "question": itemgetter("question"),
                "observation": itemgetter("observation")
            })
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        self.rag_with_history = RunnableWithMessageHistory(
            chain,
            self.get_session_history,  
            input_messages_key="question", 
            history_messages_key="chat_history", 
        )
        self.thread = None
        self.stop_event = threading.Event()

    def get_session_history(self, session_id):
        session_id = "default_session"
        if session_id not in self.chat_store:
            from langchain_community.chat_message_histories import ChatMessageHistory
            self.chat_store[session_id] = ChatMessageHistory()
        history = self.chat_store[session_id]
        if len(history.messages) > self.chat_history_limit:
            history.messages = history.messages[-self.chat_history_limit:]
        return history

    def process_query(self, query, context=None):
        input_dict = {"question": query}
        input_dict["observation"] = context
        try:
            response = self.rag_with_history.invoke(
                input_dict, config={"configurable": {"session_id": "rag123"}}
            )
            response_text = response.content if hasattr(response, "content") else str(response)
            return response_text
        except Exception as e:
            print(e)
            return "Error!"
                
    def talk(self, sim, yolo):
        while not self.stop_event.is_set():
            try:
                query = input("💬 Human: ")
                frame = sim.latest_frame
                observation = yolo.detect(frame)
                response = self.process_query(query, observation)
                self.command_queue.put(response)
                
                ans_match = re.search(r"RESPONSE:\s*([^\n]+)", response, re.IGNORECASE)
                answer = ans_match.group(1).strip() if ans_match else ""
                print(f"\n🤖 Pinky: \n{answer} \n")
            except EOFError:
                break

    def start_thread(self, target, args=()):
        self.thread = threading.Thread(target=target, args=args, daemon=True)
        self.thread.start()
