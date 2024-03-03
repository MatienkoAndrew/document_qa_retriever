import logging
import os
import time

from omegaconf import DictConfig, OmegaConf
import hydra

# LangChain и связанные с ним библиотеки
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Расширения LangChain и дополнительные инструменты
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings, HuggingFaceEmbeddings
from langchain_community.llms import GigaChat

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

########################################################################################################################
# I was getting following error while trying to run chromadb example code using my python3.10.8 venv3.10:
# File "~/venv3.10/lib/python3.10/site-packages/chromadb/__init__.py", line 36, in <module> raise RuntimeError( RuntimeError: Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0.
#
# I executed following steps to resolve this error:
#
# Inside my python3.10.8's virtual environment i.e. venv3.10, installed pysqlite3-binary using command: pip install pysqlite3-binary
# Added these 3 lines in venv3.10/lib/python3.10/site-packages/chromadb/__init__.py at the beginning:
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
########################################################################################################################


def log_function_start_end(logger):
    """Декоратор для логирования начала и конца выполнения функции."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.info(f"Starting {func.__name__}...")
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            hours, rem = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(rem, 60)
            logger.info(f"Finished {func.__name__}. in {int(hours)}h {int(minutes)}min {int(seconds)}s")
            return result
        return wrapper
    return decorator


class DocumentQuestionAnswering:
    def __init__(self, config: DictConfig):
        self.config = config
        self.logger = self.setup_logging()
        self.setup_credentials()
        self.texts = self.load_and_process_documents()
        self.vectordb = self.setup_vector_database()
        self.qa_chain = self.create_retrieval_qa_chain()

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)

    def setup_credentials(self):
        credentials = OmegaConf.to_container(self.config['credentials'], resolve=True)
        os.environ["GIGACHAT_CREDENTIALS"] = credentials['GIGACHAT_CREDENTIALS']

    @log_function_start_end(logger)
    def load_and_process_documents(self):
        path = self.config['app']['filepath']
        loader = DirectoryLoader(path, glob="./*.txt", loader_cls=TextLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        return texts

    @log_function_start_end(logger)
    def setup_vector_database(self):
        app_config = OmegaConf.to_container(self.config['app'], resolve=True)
        embedding = HuggingFaceEmbeddings(model_name=app_config['embedding_model'])
        persist_directory = app_config['persist_directory_name']

        if os.path.exists(persist_directory):
            vectordb = Chroma(persist_directory=persist_directory,
                              embedding_function=embedding)
        else:
            vectordb = Chroma.from_documents(documents=self.texts,
                                             embedding=embedding,
                                             persist_directory=persist_directory)
            vectordb.persist()
        return vectordb

    @log_function_start_end(logger)
    def create_retrieval_qa_chain(self):
        gigachat = GigaChat(verify_ssl_certs=False)
        qa_chain = RetrievalQA.from_chain_type(llm=gigachat,
                                               chain_type="stuff",
                                               retriever=self.vectordb.as_retriever(),
                                               return_source_documents=True)
        return qa_chain

    @log_function_start_end(logger)
    def run_queries(self):
        app_config = OmegaConf.to_container(self.config['app'], resolve=True)
        if app_config['queries']:
            for query in app_config['queries']:
                print(f"Query: {query}")
                llm_response = self.qa_chain(query)
                self.process_llm_response(llm_response)

    @staticmethod
    def process_llm_response(llm_response):
        print(llm_response['result'])
        print('\n\nSources:')
        for source in llm_response["source_documents"]:
            print(source.metadata['source'])


@hydra.main(version_base=None, config_path="config/", config_name="config")
def main(config: DictConfig):
    qa_app = DocumentQuestionAnswering(config)
    qa_app.run_queries()


if __name__ == "__main__":
    main()
