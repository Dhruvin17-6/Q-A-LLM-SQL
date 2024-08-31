import os
from typing import List, Tuple
from openai import AzureOpenAI
from utils.load_config import LoadConfig
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sqlalchemy import create_engine
from langchain_community.agent_toolkits import create_sql_agent
import langchain
from langchain.memory import ConversationBufferMemory
from operator import itemgetter
langchain.debug = True

APPCFG = LoadConfig()

class ChatBot:
    """
    A ChatBot class capable of responding to messages using different modes of operation.
    It can interact with SQL databases, leverage language chain agents for Q&A,
    and use embeddings for Retrieval-Augmented Generation (RAG) with ChromaDB.
    """
    def __init__(self):
        self.memory = ConversationBufferMemory()
    
    def respond(self, chatbot: List, message: str, chat_type: str, app_functionality: str) -> Tuple[str, List]:
        """
        Respond to a message based on the given chat and application functionality types.

        Args:
            chatbot (List): A list representing the chatbot's conversation history.
            message (str): The user's input message to the chatbot.
            chat_type (str): Describes the type of the chat (interaction with SQL DB or RAG).
            app_functionality (str): Identifies the functionality for which the chatbot is being used (e.g., 'Chat').

        Returns:
            Tuple[str, List]: A tuple containing a string (response) and the updated chatbot conversation list.
        """
        response_text = ""  # Placeholder for the response text

        if app_functionality == "Chat":
            # Update conversation history
            conversation_history = self.memory.load_memory_variables({}).get("conversation_history", [])
            conversation_history.append(f"User: {message}")

            if chat_type == "Q&A with stored SQL-DB":
                mysql_uri = APPCFG.sqldb_directory
                try:
                    db = SQLDatabase.from_uri(mysql_uri)
                    execute_query = QuerySQLDataBaseTool(db=db)
                    write_query = create_sql_query_chain(APPCFG.model1, db)
                    answer_prompt = PromptTemplate.from_template(APPCFG.agent_llm_system_role)
                    answer = answer_prompt | APPCFG.model | StrOutputParser()

                    # Combine history and current message
                    combined_input = "\n".join(conversation_history) + f"\nUser: {message}"

                    # Create and invoke the chain
                    chain = (
                        RunnablePassthrough.assign(query=write_query).assign(
                            result=itemgetter("query") | execute_query
                        )
                        | answer
                    )
                    response = chain.invoke({"question": combined_input})

                    # Update conversation history with response
                    conversation_history.append(f"Bot: {response}")
                    response_text = response  # Set the response text

                    # Save updated history
                    self.memory.save_context({"conversation_history": conversation_history}, {})

                    # Append to chatbot
                    chatbot.append((message, response))

                except Exception as e:
                    error_message = f"Error connecting to SQL DB: {str(e)}"
                    chatbot.append((message, error_message))
                    response_text = error_message

            elif chat_type == "Q&A with Uploaded CSV/XLSX SQL-DB" or chat_type == "Q&A with stored CSV/XLSX SQL-DB":
                if chat_type == "Q&A with Uploaded CSV/XLSX SQL-DB":
                    if os.path.exists(APPCFG.uploaded_files_sqldb_directory):
                        engine = create_engine(f"sqlite:///{APPCFG.uploaded_files_sqldb_directory}")
                        db = SQLDatabase(engine=engine)
                    else:
                        error_message = "SQL DB from the uploaded csv/xlsx files does not exist. Please upload the files."
                        chatbot.append((message, error_message))
                        response_text = error_message
                        return response_text, chatbot

                elif chat_type == "Q&A with stored CSV/XLSX SQL-DB":
                    if os.path.exists(APPCFG.stored_csv_xlsx_sqldb_directory):
                        engine = create_engine(f"sqlite:///{APPCFG.stored_csv_xlsx_sqldb_directory}")
                        db = SQLDatabase(engine=engine)
                    else:
                        error_message = "SQL DB from the stored csv/xlsx files does not exist. Please prepare the SQL DB."
                        chatbot.append((message, error_message))
                        response_text = error_message
                        return response_text, chatbot

                agent_executor = create_sql_agent(
                    APPCFG.model, db=db, agent_type="openai-tools", verbose=True
                )
                response = agent_executor.invoke({"input": message})
                response = response["output"]

                # Update conversation history with response
                conversation_history.append(f"Bot: {response}")
                response_text = response  # Set the response text

                # Save updated history
                self.memory.save_context({"conversation_history": conversation_history}, {})

                # Append to chatbot
                chatbot.append((message, response))

            elif chat_type == "RAG with stored CSV/XLSX ChromaDB":
                client = AzureOpenAI(
                    api_version="2023-03-15-preview",
                    azure_endpoint="https://fabric-poc.openai.azure.com/",
                    api_key="5a3e842aa1b14dc7b092553422349c8d",
                )

                response = client.embeddings.create(
                    input=message,
                    model="gpt-text-ada"
                )
                query_embeddings = response.data[0].embedding
                vectordb = APPCFG.chroma_client.get_collection(
                    name=APPCFG.collection_name)
                results = vectordb.query(
                    query_embeddings=query_embeddings,
                    n_results=APPCFG.top_k
                )
                prompt = f"User's question: {message} \n\n Search results:\n {results}"

                messages = [
                    {"role": "system", "content": str(APPCFG.rag_llm_system_role)},
                    {"role": "user", "content": prompt}
                ]
                llm_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages
                )
                response = llm_response.choices[0].message.content

                # Update conversation history with response
                conversation_history.append(f"Bot: {response}")
                response_text = response  # Set the response text

                # Save updated history
                self.memory.save_context({"conversation_history": conversation_history}, {})

                # Append to chatbot
                chatbot.append((message, response))

        return response_text, chatbot

