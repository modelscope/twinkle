from fastapi import Request
from fastapi.responses import JSONResponse


async def verify_request_token(request: Request, call_next):
    authorization = request.headers.get("Authorization")
    token = authorization[7:] if authorization and authorization.startswith("Bearer ") else authorization
    if not is_token_valid(token):
        return JSONResponse(status_code=403, content={"detail": "Invalid token"})

    request_id = request.headers.get("X-Ray-Serve-Request-Id")
    if not request_id:
        return JSONResponse(
            status_code=400,
            content={"detail": "Missing X-Ray-Serve-Request-Id header, required for sticky session"}
        )
    request.state.request_id = request_id
    request.state.token = token
    response = await call_next(request)
    return response


def is_token_valid(token: str) -> bool:
    return True