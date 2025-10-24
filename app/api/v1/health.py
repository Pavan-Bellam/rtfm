from fastapi import APIRouter

router = APIRouter()

@router.get(
    "/",
    description="Check if the app is live"
)
def health():
    return {"status" : "ok"}

