from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from retrieval_agent_fireworks import agent_executor as retrieval_agent_fireworks_chain

import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"belajar_llm"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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" 

app = FastAPI()

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(app, retrieval_agent_fireworks_chain, path="/retrieval-agent-fireworks")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
