import fastapi
import fastapi.security


AUTHORIZATION = fastapi.security.APIKeyHeader(name="Authorization", auto_error=True)


async def get_token(authorization: str = fastapi.Depends(AUTHORIZATION)) -> None:
    """
    Require a token to call this endpoint.

    It'll already be injected by the token_middleware, but _enforce_ it
    """
    return ...
