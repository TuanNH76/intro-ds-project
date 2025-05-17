# from api.grpc_server import serve_grpc
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from main.backend import (
    get_ner_count_by_time_interval,
    get_scraped_urls_by_time_interval,
    get_total_scraped_urls,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or use ["*"] to allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Model Service is running"}


@app.get("/scraped_urls")
async def get_scraped_urls(start_time: str, end_time: str):
    """
    Get scraped URLs from MongoDB within a specified time interval.
    """
    try:
        res = await get_scraped_urls_by_time_interval(start_time, end_time)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid start or end time format. Please use ISO format.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while fetching scraped URLs: {str(e)}",
        )
    return res


@app.get("/total_scraped_urls")
async def get_total_urls():
    """
    Get all scraped URLs from MongoDB.
    """
    try:
        res = await get_total_scraped_urls()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while fetching total scraped URLs: {str(e)}",
        )
    return {
        "total_scraped_urls": res,
    }


@app.get("/ner_count")
async def get_ner_count(start_time: str, end_time: str):
    """
    Get NER counts from MongoDB within a specified time interval.
    """
    try:
        res = await get_ner_count_by_time_interval(start_time, end_time)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid start or end time format. Please use ISO format.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while fetching NER counts: {str(e)}",
        )
    return res
