from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import os
import uuid
import json
from pathlib import Path
from datetime import datetime

from utils.pdf_utils import extract_text_from_pdfs
from utils.chunk_utils import get_text_chunks
from utils.qa_utils import get_vectorstore, get_qa_chain
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware to allow access from all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,    
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Path to the React build
build_path = os.path.join(os.path.dirname(__file__), "dist")

# Mount static files
app.mount(
    "/assets", StaticFiles(directory=os.path.join(build_path, "assets")), name="assets"
)

# Storage paths
STORAGE_DIR = "storage"
SESSION_DATA_FILE = os.path.join(STORAGE_DIR, "session_data.json")

# Ensure storage directory exists
os.makedirs(STORAGE_DIR, exist_ok=True)

class QAChainStorage:
    def __init__(self):
        self.qa_chains = {}  # Keep QA chains in memory
        self.session_data = {}  # Store raw text and metadata
        self.load_from_disk()
    
    def load_from_disk(self):
        """Load session data and recreate QA chains"""
        try:
            if os.path.exists(SESSION_DATA_FILE):
                with open(SESSION_DATA_FILE, 'r', encoding='utf-8') as f:
                    self.session_data = json.load(f)
                
                print(f"Loaded {len(self.session_data)} sessions from disk")
                
                # Recreate QA chains for all sessions
                for session_id, data in self.session_data.items():
                    try:
                        self._recreate_qa_chain(session_id, data)
                        print(f"Recreated QA chain for session: {session_id}")
                    except Exception as e:
                        print(f"Failed to recreate QA chain for session {session_id}: {e}")
                        # Mark session as inactive but keep the data
                        self.session_data[session_id]["status"] = "inactive"
                
        except Exception as e:
            print(f"Error loading from disk: {e}")
            self.session_data = {}
            self.qa_chains = {}
    
    def _recreate_qa_chain(self, session_id, data):
        """Recreate QA chain from stored raw text"""
        if data.get("status") != "active":
            return
            
        raw_text = data.get("raw_text", "")
        filename = data.get("filename", "unknown")
        
        if not raw_text:
            raise ValueError("No raw text found for session")
        
        # Recreate the full pipeline
        chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(chunks, filename)
        qa_chain = get_qa_chain(vectorstore)
        
        self.qa_chains[session_id] = qa_chain
    
    def save_to_disk(self):
        """Save session data to disk"""
        try:
            with open(SESSION_DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.session_data, f, ensure_ascii=False, indent=2)
            
            print(f"Saved {len(self.session_data)} sessions to disk")
            
        except Exception as e:
            print(f"Error saving to disk: {e}")
    
    def add_session(self, session_id, raw_text, filename=None):
        """Add a new session with raw text and create QA chain"""
        try:
            if session_id in self.session_data:
                # Auto-generate new session_id by appending a suffix
                original_id = session_id
                counter = 1
                while session_id in self.session_data:
                    session_id = f"{original_id}-{counter}"
                    counter += 1
                print(f"[INFO] Session ID already exists. New session ID used: {session_id}")

            # Store raw text and metadata
            self.session_data[session_id] = {
                "raw_text": raw_text,
                "filename": filename,
                "created_at": datetime.now().isoformat(),
                "status": "active"
            }

            # Create QA chain
            chunks = get_text_chunks(raw_text)
            vectorstore = get_vectorstore(chunks, filename)
            qa_chain = get_qa_chain(vectorstore)
            self.qa_chains[session_id] = qa_chain

            # Save to disk
            self.save_to_disk()

            print(f"[INFO] Added session: {session_id} with filename: {filename}")

        except Exception as e:
            print(f"[ERROR] Adding session: {e}")
            if session_id in self.session_data:
                del self.session_data[session_id]
            raise

    
    def get_chain(self, session_id):
        """Get QA chain for a session"""
        # If not in memory, try to recreate from stored data
        if session_id not in self.qa_chains and session_id in self.session_data:
            try:
                data = self.session_data[session_id]
                if data.get("status") == "active":
                    self._recreate_qa_chain(session_id, data)
                    print(f"Recreated QA chain for session: {session_id}")
            except Exception as e:
                print(f"Failed to recreate QA chain for session {session_id}: {e}")
                return None
        
        return self.qa_chains.get(session_id)
    
    def remove_session(self, session_id):
        """Remove a session and its data"""
        try:
            # Remove from memory
            if session_id in self.qa_chains:
                del self.qa_chains[session_id]
            
            # Remove from persistent storage
            if session_id in self.session_data:
                del self.session_data[session_id]
            
            self.save_to_disk()
            print(f"Removed session: {session_id}")
            
        except Exception as e:
            print(f"Error removing session: {e}")
    
    def list_sessions(self):
        """List all sessions with their metadata"""
        sessions_info = {}
        for session_id, data in self.session_data.items():
            sessions_info[session_id] = {
                "filename": data.get("filename", "unknown"),
                "created_at": data.get("created_at", "unknown"),
                "status": data.get("status", "unknown"),
                "has_active_chain": session_id in self.qa_chains
            }
        return sessions_info
    
    def cleanup_old_sessions(self, max_sessions=50):
        """Remove old sessions if we have too many"""
        if len(self.session_data) > max_sessions:
            # Sort by creation date and remove oldest
            sorted_sessions = sorted(
                self.session_data.items(),
                key=lambda x: x[1].get("created_at", ""),
                reverse=True
            )
            
            sessions_to_keep = dict(sorted_sessions[:max_sessions])
            sessions_to_remove = [sid for sid in self.session_data.keys() 
                                if sid not in sessions_to_keep]
            
            for session_id in sessions_to_remove:
                self.remove_session(session_id)
            
            print(f"Cleaned up {len(sessions_to_remove)} old sessions")

# Initialize storage
storage = QAChainStorage()

class QueryRequest(BaseModel):
    query: str
    session_id: str

# Serve index.html for root or any unknown route (for SPA)
@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    index_file = os.path.join(build_path, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"error": "React build not found"}

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List


MAX_FILE_SIZE_MB = 500

@app.post("/upload/")
async def upload_pdfs(files: List[UploadFile] = File(...), session_id: str = Form(...)):
    try:
        # Validate file sizes
        for file in files:
            content = await file.read()
            if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
                raise HTTPException(
                    status_code=400,
                    detail=f"File '{file.filename}' exceeds the 500MB size limit."
                )
            # Reset the file pointer after reading
            file.file.seek(0)

        # Extract raw text and filename
        raw_text, base_filename = extract_text_from_pdfs(files)
        
        # Save session and create the QA chain
        storage.add_session(session_id, raw_text, base_filename)
        
        # Optional: cleanup old sessions
        #storage.cleanup_old_sessions(max_sessions=150)
        
        return {"status": "PDFs processed successfully", "filename": base_filename}
    
    except Exception as e:
        return {"error": f"Failed to process PDFs: {str(e)}"}# @app.post("/query/")
# async def query_docs(req: QueryRequest):
#     qa_chain = storage.get_chain(req.session_id)
#     if not qa_chain:
#         return {"error": "Session not found. Please upload PDFs first."}
    
#     try:
#         response = qa_chain.run(req.query)
#         return {"response": response}
#     except Exception as e:
#         return {"error": f"Query failed: {str(e)}"}


@app.post("/query/")
async def query_docs(req: QueryRequest):
    qa_chain = storage.get_chain(req.session_id)
    if not qa_chain:
        return {"error": "Session not found. Please upload PDFs first."}
    
    try:
        response = qa_chain.run(req.query)
        
        # Save to file
        save_query_response_to_file(req.session_id, req.query, response)

        return {"response": response}
    except Exception as e:
        return {"error": f"Query failed: {str(e)}"}


@app.get("/sessions/")
async def list_sessions():
    """API endpoint to list all sessions"""
    return {"sessions": storage.list_sessions()}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """API endpoint to delete a specific session"""
    if session_id in storage.session_data:
        storage.remove_session(session_id)
        return {"status": f"Session {session_id} deleted successfully"}
    return {"error": "Session not found"}

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_sessions": len(storage.qa_chains),
        "total_sessions": len(storage.session_data)
    }

# ========================== TERMINAL MODE ==========================

def main(query, session_id):
    qa_chain = storage.get_chain(session_id)
    if not qa_chain:
        raise ValueError("Session ID not initialized. Upload PDFs first.")
    return qa_chain.run(query)

print(f"Loaded sessions: {list(storage.session_data.keys())}")
print(f"Active QA chains: {list(storage.qa_chains.keys())}")


def get_unique_session_id(self, base_id):
    session_id = base_id
    counter = 1
    while session_id in self.session_data:
        session_id = f"{base_id}-{counter}"
        counter += 1
    return session_id



def save_query_response_to_file(session_id, query, response):
    # Get current date in YYYY-MM-DD format
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # File name includes session_id and date
    log_file = os.path.join(STORAGE_DIR, f"queries_{session_id}_{current_date}.txt")

    entry = f"---\nQuery: {query}\nResponse: {response}\n"

    # Prepend new entry if file exists
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            existing = f.read()
    else:
        existing = ""

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(entry + existing)







if __name__ == "__main__":
    import uvicorn

    # For terminal testing only
    session_id = "console-session"
    pdf_path = input("Enter path to PDF folder: ").strip()

    # Load PDFs manually
    from pathlib import Path

    class FakeUpload:
        def __init__(self, path):
            self.filename = os.path.basename(path)
            self.file = open(path, "rb")

    pdf_files = [FakeUpload(str(p)) for p in Path(pdf_path).glob("*.pdf")]

    if pdf_files:
       
        raw_text, base_filename = extract_text_from_pdfs(pdf_files)
        # Generate a unique session ID based on base_filename (e.g., filename as prefix)
        base_session_id = os.path.splitext(base_filename)[0].replace(" ", "_").lower()
        session_id = storage.get_unique_session_id(base_session_id)

        # Save to persistent storage
        storage.add_session(session_id, raw_text, base_filename)

        while True:
            user_query = input("QUERY: ")
            if user_query.lower() == "quit":
                break
            response = main(user_query, session_id)
            print("\nRESPONSE:\n", response)

        # Optional: Close files
        for f in pdf_files:
            f.file.close()
    else:
        print("No PDF files found in the specified directory.")