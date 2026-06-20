import uvicorn
from app.core.config import settings

if __name__ == "__main__":
    print(f"Starting Server at: http://{settings.HOST}:{settings.PORT}")
    print(f"API Docs available at: http://{settings.HOST}:{settings.PORT}/docs")
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
