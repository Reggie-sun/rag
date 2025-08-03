#主要为了实现（QA),复杂的机器人聊天模型，或者回答特定的聊天问题"""
#1索引（加载文本，分割文本，贮存在向量数据库）
#2检索和生成
import os
import streamlit as st
import tempfile
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_react_agent,AgentExecutor
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()
hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
st.title("文档回答")
upload_files=st.sidebar.file_uploader(label="上传pdf文件",type=["pdf"],accept_multiple_files=True)
if not upload_files:
    st.info("请先上传文件")
    st.stop()
#直接读取上传文件不行吗，为毛还要写个临时路径再把内容写进去？

#能直接读取上传的文件内容，没问题！
#但是——如果你用的工具（比如 TextLoader)只支持“文件路径”作为输入,那就必须写入一个临时文件再读。

#因为要在stremlit云端部署,所以不能直接从电脑上读取,但是你可以上传到github来直接读取,或者像这样把它写入临时路径再读取。
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
instructions ="""
您是一个设计用于查询文档来回答问题的代理。
您可以使用文档检索工具，查询文档的次数只能是一次，并基于检索内容来回答问题.
如果您从文档中找不到任何信息用于回答问题，则可以自己思考作为答案。"""

base_prompt_template = '''{instructions}

TOOLS:

You have access to the following tools:
{tools}
To use a tool, please use the following format:

Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: {input}
Observation: the result of the action

When you have a response to say to the Human, or if you do not need to use a tool, you MusT use the format:

Thought: Do I need to use a tool? No
Final Answer: [your response here]

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}'''

base_prompt = PromptTemplate.from_template(template=base_prompt_template)

prompt=base_prompt.partial(instructions=instructions)

from sentence_transformers import SentenceTransformer

from langchain_core.embeddings import Embeddings

class SBertEmbeddings(Embeddings):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2',use_auth_token=hf_token):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()
    
@st.cache_resource(ttl=3600)
def configure_retriever(uploaded_files):
    docs=[]
    temp_dir=tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath=os.path.join(temp_dir.name,file.name)
        with open(temp_filepath,"wb") as f:
            f.write(file.getvalue())#file.getvalue()得到的是bytes字节流数据，本质是二进制数据，所以用wb
        loader=PyPDFLoader(temp_filepath)
        docs.extend(loader.load())
    
    split_docs=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
    embedding_model = SBertEmbeddings()
    vector=Chroma.from_documents(split_docs,embedding_model)
    retriever=vector.as_retriever()
    return retriever



retriever=configure_retriever(upload_files)

if 'messages' not in st.session_state or st.sidebar.button('清空聊天记录'):
    st.session_state['messages']=[{'role':'assistant','content':'您好，我是文档问答助手'}]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])
 
from langchain.tools.retriever import create_retriever_tool

tool=create_retriever_tool(retriever,
                           name='文档检索',
                           description="检索用户问题，并且给出回答"
                           )

tools=[tool]

from langchain_core.messages import AIMessage, ToolMessage

msgs=StreamlitChatMessageHistory()

from langchain.memory import ConversationBufferWindowMemory

memory=ConversationBufferWindowMemory(
    chat_memory=msgs,return_messages=True,memory_key='chat_history',output_key='output')

api_key = os.getenv('DEEPSEEK_API_KEY')
if not api_key:
    st.error("请先在 .env 文件中配置 DEEPSEEK_API_KEY")
    st.stop()
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url="https://api.deepseek.com",
    temperature=0.0,
    max_tokens=10000,
)

agent=create_react_agent(llm,tools,prompt)

agent_executor=AgentExecutor(agent=agent,tools=tools,
                             memory=memory,
                             verbose=True,
                             handle_parsing_errors="没有检索到相似内容")

user_query=st.chat_input(placeholder="请输入你的问题")

if user_query:
    st.session_state.messages.append({'role':'user','content':user_query})
    st.chat_message('user').write(user_query)
    
    st_cb=StreamlitCallbackHandler(st.container())

    with st.chat_message('assistant'):
          out=agent_executor.invoke({'input':user_query},config={'callbacks':[st_cb]})
          st.session_state.messages.append({'role':'assistant','content':out['output']})

          st.write(out['output'])



