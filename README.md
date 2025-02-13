# Text Summarization using LangChain

This repository demonstrates various text summarization techniques implemented using LangChain and Groq's language models. The project showcases different approaches to summarize both speeches and documents using various chain types and prompts.

## Features

- Speech summarization using ChatGroq with Qwen 2.5 32B model
- Multiple summarization techniques:
  - Simple LLM Chain
  - Stuff Documents Chain
  - Map Reduce Chain
  - Refine Chain
- Support for multilingual summarization
- PDF document processing capabilities
- Customizable prompt templates

## Prerequisites

```bash
pip install langchain-groq python-dotenv langchain
```

## Environment Setup

1. Create a `.env` file in your project root
2. Add your Groq API key:
```
GROQ_API_KEY=your_api_key_here
```

## Usage

### Basic Speech Summarization

```python
from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage, SystemMessage

llm = ChatGroq(groq_api_key=api_key, model_name="qwen-2.5-32b")

chat_message = [
    SystemMessage(content="You are an expert with expertise with summarizing speeches"),
    HumanMessage(content=f"Please provide a concise short summary for the following speech: \n Text:{speech}")
]
```

### Multilingual Summarization

```python
from langchain.chains import LLMChain
from langchain import PromptTemplate

template = """You are a helpful assistant. Write a summary of the following speech:
Speech:{speech}
translate the precise summary into {language}"""

prompt = PromptTemplate(
    template=template,
    input_variables=["speech", "language"],
)

llm_chain = LLMChain(llm=llm, prompt=prompt)
```

### Document Summarization

#### Using Stuff Documents Chain

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain

template = """write a concise summary and short summary of the following document:
document: {text}
summary:
"""

prompt = PromptTemplate(template=template, input_variables=["text"])
chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
```

#### Using Map Reduce Chain

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

summary_chain = load_summarize_chain(
    llm=llm,
    chain_type='map_reduce',
    verbose=True,
    map_prompt=map_prompt_template,
    combine_prompt=final_prompt_template
)
```

## Supported Chain Types

1. **Stuff Chain**: Best for smaller documents that can be processed in a single API call
2. **Map Reduce Chain**: Ideal for large documents, splits the text and processes chunks in parallel
3. **Refine Chain**: Processes the document iteratively, refining the summary with each step

## Contributing

Feel free to open issues and pull requests to improve the project.

## License

This project is open source and available under the MIT License.
