# server/app.py
# Required by openenv validate as the server entry point
from supportsphere.server.app import app  

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
