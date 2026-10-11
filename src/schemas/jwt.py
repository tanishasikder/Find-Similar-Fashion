import jwt
from jwt import PyJWKClient
from jwt.exceptions import InvalidTokenError, PyJWKClientError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from dotenv import load_dotenv
import os

load_dotenv()

jwt_url = os.environ.get('JWT_URL')

JWKS_URL = jwt_url
EXPECTED_AUDIENCE = "authenticated"
EXPECTED_ISSUER = os.environ.get('EXPECTED_ISSUER')

# Without these jwt.decode would silently skip the issuer check
if not JWKS_URL or not EXPECTED_ISSUER:
    raise RuntimeError('JWT_URL and EXPECTED_ISSUER must be set in the environment')

jwks_client = PyJWKClient(JWKS_URL)

# Reads the token from the "Authorization: Bearer <token>" header
bearer_scheme = HTTPBearer(auto_error=False)

def unauthorized(detail: str):
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={'WWW-Authenticate': 'Bearer'}
    )

def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials is None:
        raise unauthorized('Authorization is missing')

    token = credentials.credentials
    try:
        key = jwks_client.get_signing_key_from_jwt(token)

        payload = jwt.decode(
            token,
            key.key,
            algorithms=['RS256'],
            audience=EXPECTED_AUDIENCE,
            issuer=EXPECTED_ISSUER
        )

        return payload

    except (InvalidTokenError, PyJWKClientError) as e:
        print(f"Token validation failed: {e}")
        raise unauthorized('Invalid or expired token')
