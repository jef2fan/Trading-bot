import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import finall  # فایل اصلی تو

app = FastAPI(title="Trading Bot API", version="1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/run")
def run_once():
    try:
        results = finall.main_once()
        out = {sym: vars(sig) for sym, sig in results.items()}
        return JSONResponse(out)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
