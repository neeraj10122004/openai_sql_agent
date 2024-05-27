from langchain.utilities import SQLDatabase
from langchain.llms import OpenAI
from langchain.agents.agent_types import AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.utilities import SQLDatabase
from langchain_openai import OpenAI
from langchain.agents import create_sql_agent
import os
from dotenv import load_dotenv

st.title("SQL Agent Using Openai")

# Enter your OpenAI API private access key here. IMPORTANT - don't share your code online if it contains your access key or anyone will be able to access your openai account
if api_key := st.text_input("Enter the API Key for OPENAI"):
    os.environ['OPENAI_API_KEY'] = api_key
    load_dotenv()
    # define the database we want to use for our test
    db = SQLDatabase.from_uri('sqlite:///sql_lite_database.db')

    # choose llm model, in this case the default OpenAI model
    llm = OpenAI(
            temperature=0,
            verbose=True,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            )
    # setup agent
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response=str(agent_executor.invoke(prompt)["output"])
            st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})