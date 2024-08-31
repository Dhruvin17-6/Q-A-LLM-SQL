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
# from operator import itemgetter
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
    @staticmethod
    def respond(chatbot: List, message: str, chat_type: str, app_functionality: str) -> Tuple:
        """
        Respond to a message based on the given chat and application functionality types.

        Args:
            chatbot (List): A list representing the chatbot's conversation history.
            message (str): The user's input message to the chatbot.
            chat_type (str): Describes the type of the chat (interaction with SQL DB or RAG).
            app_functionality (str): Identifies the functionality for which the chatbot is being used (e.g., 'Chat').

        Returns:
            Tuple[str, List, Optional[Any]]: A tuple containing an empty string, the updated chatbot conversation list,
                                             and an optional 'None' value. The empty string and 'None' are placeholder
                                             values to match the required return type and may be updated for further functionality.
                                             Currently, the function primarily updates the chatbot conversation list.
        """
        if app_functionality == "Chat":
            memory = ConversationBufferMemory()
            # If we want to use langchain agents for Q&A with our SQL DBs that were created from .sql files.
            if chat_type == "Q&A with stored SQL-DB":
                mysql_uri = APPCFG.sqldb_directory
                try:
                    db = SQLDatabase.from_uri(mysql_uri)
                    execute_query = QuerySQLDataBaseTool(db=db)
                    write_query = create_sql_query_chain(APPCFG.model1, db)
                    answer_prompt = PromptTemplate.from_template(APPCFG.agent_llm_system_role)
                    answer = answer_prompt | APPCFG.model | StrOutputParser()

                    # Load existing question history (if any)
                    question_history = memory.load_memory_variables({}).get("questions", "")

                    # Append the current question to the history
                    if question_history:
                        question_history += f"\n{len(question_history.splitlines()) + 1}. {message}"
                    else:
                        question_history = f"1. {message}"

                    # Combine the question and history to generate a more informed query
                    combined_input = f"{question_history}\n{len(question_history.splitlines()) + 1}. {message}"

                    # Create and invoke the chain
                    chain = (
                        RunnablePassthrough.assign(query=write_query).assign(
                            result=itemgetter("query") | execute_query
                        )
                        | answer
                    )
                    response = chain.invoke({"question": combined_input, "history": question_history})

                    # Save the updated question history
                    memory.save_context({"questions": question_history}, {"response": response})

                    # Print the complete question history (old and new)
                    print("Question History:")
                    print(question_history)

                except Exception as e:
                    chatbot.append((message, f"Error connecting to SQL DB: {str(e)}"))
                    return "", chatbot, None

            # If we want to use langchain agents for Q&A with our SQL DBs that were created from CSV/XLSX files.
            elif chat_type == "Q&A with Uploaded CSV/XLSX SQL-DB" or chat_type == "Q&A with stored CSV/XLSX SQL-DB":
                if chat_type == "Q&A with Uploaded CSV/XLSX SQL-DB":
                    if os.path.exists(APPCFG.uploaded_files_sqldb_directory):
                        engine = create_engine(
                            f"sqlite:///{APPCFG.uploaded_files_sqldb_directory}")
                        db = SQLDatabase(engine=engine)
                        print(db.dialect)
                    else:
                        chatbot.append(
                            (message, f"SQL DB from the uploaded csv/xlsx files does not exist. Please first upload the csv files from the chatbot."))
                        return "", chatbot, None
                elif chat_type == "Q&A with stored CSV/XLSX SQL-DB":
                    if os.path.exists(APPCFG.stored_csv_xlsx_sqldb_directory):
                        engine = create_engine(
                            f"sqlite:///{APPCFG.stored_csv_xlsx_sqldb_directory}")
                        db = SQLDatabase(engine=engine)
                    else:
                        chatbot.append(
                            (message, f"SQL DB from the stored csv/xlsx files does not exist. Please first execute `src/prepare_csv_xlsx_sqlitedb.py` module."))
                        return "", chatbot, None
                print(db.dialect)
                print(db.get_usable_table_names())
                agent_executor = create_sql_agent(
                    APPCFG.model, db=db, agent_type="openai-tools", verbose=True)
                response = agent_executor.invoke({"input": message})
                response = response["output"]

            elif chat_type == "RAG with stored CSV/XLSX ChromaDB":

                client = AzureOpenAI(
    api_version="2023-03-15-preview",
    azure_endpoint="https://fabric-poc.openai.azure.com/",
    api_key="5a3e842aa1b14dc7b092553422349c8d",
)
                
                response = client.embeddings.create(
        input = message,
        model= "gpt-text-ada"
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
                    {"role": "system", "content": str(
                        APPCFG.rag_llm_system_role
                    )},
                    {"role": "user", "content": prompt}
                ]
                llm_response =client.chat.completions.create(
                    model="gpt3-5",
                    messages=messages
                )
                response = llm_response.choices[0].message.content

            # Get the `response` variable from any of the selected scenarios and pass it to the user.
            chatbot.append(
                (message, response))
            return "", chatbot
        else:
            pass
