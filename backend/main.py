import os
import json
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import uvicorn


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    init_resources()
    yield
    # Shutdown logic (if any)

app = FastAPI(title="AI Travel Assistant API", lifespan=lifespan)


# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the directory where main.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration (Relative to BASE_DIR)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_ARTIFACTS_PATH = os.path.join(BASE_DIR, "models", "logistic_model_outputs.pkl")
LOCATIONS_CSV = os.path.join(BASE_DIR, "data", "locations.csv")
PRODUCTS_CSV = os.path.join(BASE_DIR, "data", "products.csv")

# Global variables for models and data
embedder = None
ml_model = None
ml_label_encoder = None
ml_attribute_names = None
location_vs = None
attr_vs = None
df_products = None
df_locations = None
llm = None

def init_resources():
    global embedder, ml_model, ml_label_encoder, ml_attribute_names
    global location_vs, attr_vs, df_products, df_locations, llm

    print("🔄 Initializing resources...")
    
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not found in .env")
        return

    # 1. Embedder (Cloud-based)
    embedder = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    # 2. ML Model (Local)
    if os.path.exists(MODEL_ARTIFACTS_PATH):
        artifacts = joblib.load(MODEL_ARTIFACTS_PATH)
        ml_model = artifacts["model"]
        ml_label_encoder = artifacts["label_encoder"]
        ml_attribute_names = artifacts["attribute_names"]
        
        attr_docs = [Document(page_content=attr, metadata={"name": attr}) for attr in ml_attribute_names]
        attr_vs = FAISS.from_documents(attr_docs, embedder)
        print("✅ Local ML Model loaded.")
    else:
        print(f"⚠️ Local ML Model not found at {MODEL_ARTIFACTS_PATH}. Run train.py first.")

    # 3. Data & Vector Stores
    if os.path.exists(LOCATIONS_CSV):
        df_locations = pd.read_csv(LOCATIONS_CSV)
        docs = [Document(page_content=row["text_data"], metadata={"name": row["location_name"]})
                for _, row in df_locations.iterrows()]
        location_vs = FAISS.from_documents(docs, embedder)
        print("✅ Locations vector store initialized.")
    
    if os.path.exists(PRODUCTS_CSV):
        df_products = pd.read_csv(PRODUCTS_CSV)
        print("✅ Products data loaded.")

    # 4. LLM (Cloud-based OpenAI)
    llm = ChatOpenAI(model_name="gpt-4o-mini", api_key=OPENAI_API_KEY)
    print("✅ OpenAI LLM initialized.")

# Pydantic models for API
class ChatRequest(BaseModel):
    query: str
    history: List[dict] = []

class ChatResponse(BaseModel):
    answer: str
    followups: List[str]
    context: Optional[str] = None
    products: Optional[str] = None
    predicted_product: Optional[str] = None




@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not llm:
        raise HTTPException(status_code=500, detail="LLM not initialized")

    query = request.query
    history = request.history
    
    # Simple Routing Logic
    # 1. Check for attributes match first if it's a preference list
    context = ""
    products_info = ""
    predicted_product = ""
    
    # Determine if we should search for attributes or locations
    # For simplicity, we'll do semantic search on both and see which matches better or just combine
    
    if attr_vs:
        attr_matches = attr_vs.similarity_search(query, k=5)
        matched_attrs = [doc.page_content for doc in attr_matches]
        
        # If we have many matches, try ML prediction
        vec = [1 if attr in matched_attrs else 0 for attr in ml_attribute_names]
        df_input = pd.DataFrame([vec], columns=ml_attribute_names)
        pred_index = ml_model.predict(df_input)[0]
        predicted_product = ml_label_encoder.inverse_transform([pred_index])[0]
        
        # Get product info
        matched = df_products[df_products["product_name"] == predicted_product]
        if not matched.empty:
            row = matched.iloc[0]
            products_info = f"**{predicted_product}**\n{row.get('description', '')}\nPrice: ${row.get('price_usd', 0)}"
            
            # Get locations for this product
            locs_str = row.get("locations", "")
            loc_names = [l.strip() for l in locs_str.split(",") if l.strip()]
            matched_locs = df_locations[df_locations["location_name"].isin(loc_names)]
            context = "\n\n".join([f"{r['location_name']}: {r['text_data']}" for _, r in matched_locs.iterrows()])

    # If no predicted product or low confidence (default to normal RAG)
    if not context and location_vs:
        docs = location_vs.similarity_search(query, k=3)
        context = "\n\n".join([f"{doc.metadata['name']}: {doc.page_content}" for doc in docs])
        
        # Search for products semantically
        if df_products is not None:
            # Simple keyword match or logic for products
            # (In a real app we'd use embedding search for products too)
            products_info = "Recommended products based on search: " + ", ".join(df_products.head(2)["product_name"].tolist())

    # Generate Answer
    prompt = [
        SystemMessage(content=(
            "You are a helpful AI travel assistant. "
            "Answer the user's travel question using the provided context and products. "
            "Include recommended products in **bold**. "
            "Suggest 3 follow-up questions at the end in the format: "
            "**Suggested Follow-Ups:**\n- Q1\n- Q2\n- Q3"
        ))
    ]
    
    for msg in history[-5:]:
        if msg["role"] == "user":
            prompt.append(HumanMessage(content=msg["content"]))
        else:
            prompt.append(AIMessage(content=msg["content"]))
            
    prompt.append(HumanMessage(content=f"Context: {context}\n\nProducts: {products_info}\n\nUser Question: {query}"))
    
    response = llm.invoke(prompt)
    output = response.content
    
    answer = output
    followups = []
    
    if "**Suggested Follow-Ups:**" in output:
        parts = output.split("**Suggested Follow-Ups:**")
        answer = parts[0].strip()
        f_text = parts[1].strip()
        followups = [line.strip("- ").strip() for line in f_text.splitlines() if line.strip()]

    return ChatResponse(
        answer=answer,
        followups=followups[:3],
        context=context,
        products=products_info,
        predicted_product=predicted_product
    )

# Serve frontend files (Must be after API routes)
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"🔗 Server running at: http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


