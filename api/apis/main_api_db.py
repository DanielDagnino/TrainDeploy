import io
import logging
import os
import traceback
from datetime import datetime, timedelta
from enum import Enum as PyEnum
from time import sleep
from typing import Union

import jwt
import torchaudio
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.logger import logger as fastapi_logger
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from passlib.hash import sha256_crypt
from path import Path
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, func, Column, Integer, Float, String, DateTime, Enum
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import QueuePool
from typing_extensions import Annotated

from apis.clf_ai.model.model import predict_is_ai, extractor_ai_voice
from apis.utils.logger.logger import setup_logging

# Setup logging
log_dir = Path('./logger')
log_dir.makedirs_p()
setup_logging(log_dir=log_dir, logger_level=logging.WARNING, production=True, rank=None)
logger = logging.getLogger(__name__ + ": " + __file__)

logger.info("torchaudio.set_audio_backend")
torchaudio.set_audio_backend("soundfile")

logger.info("load_dotenv")
load_dotenv()

logger.info("variables")
# App limits
MAX_FILE_SIZE_BYTES = 25 * (1024 ** 2)
APPLY_EXPIRATION_TIME = False

# JWT config
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# MySQL config
MYSQL_DAGNINO_PASSWORD = os.getenv("MYSQL_DAGNINO_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
PATH_SSL_CA_CERTIFICATE = "./DigiCertGlobalRootCA.pem"
DB_CONNECTION_MAX_RETRIES = 3

# Define the database URL for connecting to the MySQL database.
DATABASE_URL = 'mysql+pymysql://' \
               f'dagnino:{MYSQL_DAGNINO_PASSWORD}@{DB_HOST}:3306/TrainDeploy_API_DB?ssl_ca={PATH_SSL_CA_CERTIFICATE}'

# Create an instance of the FastAPI application.
app = FastAPI()

# Set the logging level for the FastAPI logger to WARNING.
fastapi_logger.setLevel(logging.WARNING)


@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    """
    Handle unhandled exceptions and return a JSON response with a 500 status code.

    Args:
        request (Request): The incoming request that caused the exception.
        exc (Exception): The exception that was raised.

    Returns:
        JSONResponse: A JSON response with a 500 status code and an error message.
    """
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"message": "Internal Server Error"})


class User(BaseModel):
    email: str
    organization: str


class ErrorResponse(BaseModel):
    error: str


class HealthResponse(BaseModel):
    status: str = "OK"


class Token(BaseModel):
    access_token: str
    token_type: str


class AuthenticationIn(BaseModel):
    email: str
    password: str


class ProtectedOut(BaseModel):
    logged_in_as: User


class PredictionProbOut(BaseModel):
    score: Annotated[float, Field(strict=True, ge=0, le=1)]


class FileSizeOut(BaseModel):
    size_in_mb: Annotated[float, Field()]


@app.get("/")
async def health_check() -> HealthResponse:
    return HealthResponse()


# Create a base class for declarative class definitions.
Base = declarative_base()


class PricingType(PyEnum):
    """
    Enumeration for different pricing types.

    Attributes:
        PerSec (str): Pricing based on the number of seconds used.
        PerRequest (str): Pricing based on the number of requests made.
        NoLimit (str): No limit on usage.
    """
    PerSec = 'PerSec'
    PerRequest = 'PerRequest'
    NoLimit = 'NoLimit'


class UserAccessDB(Base):
    """
    SQLAlchemy ORM model for the 'user_access' table.

    Attributes:
        id (int): Primary key, unique identifier for each user access record.
        email (str): User's email address, must be unique and not nullable.
        password (str): User's password, stored as a hashed string, not nullable.
        organization (str): Name of the user's organization, not nullable.
        usage_deadline_utc (datetime): The deadline for API usage in UTC.
        pricing_type (PricingType): Enum indicating the pricing type for the user.
        n_requests_max (int): Maximum number of API requests allowed for the user.
        requests_count (int): Current count of API requests made by the user.
        n_secs_max (int): Maximum number of processing seconds allowed for the user.
        secs_count (float): Current count of processing seconds used by the user.
    """
    __tablename__ = 'user_access'
    id = Column(Integer, primary_key=True)
    email = Column(String(128), unique=True, nullable=False)
    password = Column(String(128), nullable=False)
    organization = Column(String(128), nullable=False)
    usage_deadline_utc = Column(DateTime)

    pricing_type = Column(Enum(PricingType), nullable=False)
    n_requests_max = Column(Integer)
    requests_count = Column(Integer)
    n_secs_max = Column(Integer)
    secs_count = Column(Float)


def create_database_engine_with_retry(max_retries=3):
    """
    pool_size: It determines the number of database connections to be pooled by SQLAlchemy. It represents the maximum
        number of connections that can be open at the same time
    max_overflow: It is the number of additional connections that can be created by the pool, above the pool_size,
        to handle transient spikes in database activity.
    pool_timeout: This parameter represents the maximum number of seconds that a connection pool will wait to
        get a connection before timing out.
    Args:
        max_retries: Max number of tries to create a DB.

    Returns:
        Engine created.
    """
    retries = 0
    while retries < max_retries:
        try:
            _engine = create_engine(DATABASE_URL, poolclass=QueuePool, pool_size=5, max_overflow=10, pool_timeout=10,
                                    pool_pre_ping=True, pool_use_lifo=True)
            return _engine
        except OperationalError as e:
            logger.error(f"Database connection error (retry will be tried): {e}")
            retries += 1
            sleep(1)
        except NotImplementedError as e:
            logger.error(f"NotImplementedError: {e}")

    logger.error("Failed to establish a database connection.")
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error")


# Create the database engine with retry logic.
engine = create_database_engine_with_retry(DB_CONNECTION_MAX_RETRIES)

# Create all tables in the database using the metadata from the Base class.
Base.metadata.create_all(engine)

# Create a sessionmaker factory bound to the engine.
# SessionLocal = sessionmaker(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """
    Provide a database session for the duration of a request.

    This function yields a SQLAlchemy session that can be used to interact with the database.
    The session is automatically closed after the request is completed.

    Yields:
        Session: A SQLAlchemy session object.
    """
    _db_sesion = SessionLocal()
    try:
        yield _db_sesion
    finally:
        _db_sesion.close()


def verify_password(email: str, password: str, db_session: Session):
    """
    Verify the user's password against the stored hash in the database.

    Args:
        email (str): The email of the user to verify.
        password (str): The plaintext password provided by the user.
        db_session (Session): The database session to use for querying user information.

    Returns:
        bool: True if the password is valid, False otherwise.

    Raises:
        HTTPException: If there is a database error or if the email/password combination is invalid.
    """
    try:
        user_db = db_session.query(UserAccessDB).filter_by(email=email).first()
    except Exception as e:
        db_session.rollback()
        logger.error(f"Rollback the transaction cause by the exception {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error")

    if not user_db or not sha256_crypt.verify(password, user_db.password):
        logger.error(f"Invalid email ({email}) or password ({password})")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    return True


def verify_user_time_limit(email: str, db_session: Session):
    """
    Verify if the user's usage deadline has been reached.

    Args:
        email (str): The email of the user to check.
        db_session (Session): The database session to use for querying user information.

    Raises:
        HTTPException: If there is a database error or if the user's usage deadline has been reached.
    """
    try:
        user_db = db_session.query(UserAccessDB).filter_by(email=email).first()
        usage_deadline_utc = user_db.usage_deadline_utc
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error")

    if usage_deadline_utc is not None and datetime.utcnow() > usage_deadline_utc:
        msg = "Date limit for the API usage has been reached."
        logger.error(msg)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=msg)


def verify_user_usage_limits(email: str, db_session: Session):
    """
    Verify the user's usage limits based on their pricing type.

    Args:
        email (str): The email of the user to check.
        db_session (Session): The database session to use for querying user information.

    Raises:
        HTTPException: If there is a database error or if the user's usage limits have been exceeded.
    """
    try:
        user_db = db_session.query(UserAccessDB).filter_by(email=email).first()
        pricing_type = user_db.pricing_type
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error")

    if pricing_type == PricingType.PerRequest:
        try:
            requests_count = db_session.query(func.sum(UserAccessDB.requests_count)).filter_by(
                organization=user_db.organization).scalar()
            n_requests_max = user_db.n_requests_max
        except Exception as e:
            logger.error(f"Database error: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error")

        if n_requests_max is not None and (requests_count + 1) > n_requests_max:
            msg = "Maximum number of requests to the API has been exceeded."
            logger.error(msg)
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=msg)

    elif pricing_type == PricingType.PerSec:
        try:
            secss_count = db_session.query(func.sum(UserAccessDB.secs_count)).filter_by(
                organization=user_db.organization).scalar()
            n_secs_max = user_db.n_secs_max
        except Exception as e:
            logger.error(f"Database error: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error")

        if n_secs_max is not None and secss_count > n_secs_max:
            msg = "Maximum processing seconds allowed by the API have been exceeded."
            logger.error(msg)
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=msg)

    elif pricing_type == PricingType.NoLimit:
        return

    else:
        msg = f"Unsupported pricing_type encountered: {pricing_type}."
        logger.error(msg)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error")

def create_access_token(email: str, db_session: Session):
    """
    Create a JWT access token for the given user.

    Args:
        email (str): The email of the user for whom the token is being created.
        db_session (Session): The database session to use for querying user information.

    Returns:
        str: The generated JWT access token.

    Raises:
        HTTPException: If there is a database error during the user query.
    """
    try:
        user_db = db_session.query(UserAccessDB).filter_by(email=email).first()
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error")

    if APPLY_EXPIRATION_TIME:
        token_expiration_time = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        token = jwt.encode({'email': email, 'organization': user_db.organization,
                            'token_expiration_time': token_expiration_time.isoformat()}, JWT_SECRET_KEY,
                           algorithm=JWT_ALGORITHM)
    else:
        token = jwt.encode({'email': email, 'organization': user_db.organization}, JWT_SECRET_KEY,
                           algorithm=JWT_ALGORITHM)
    return token


def add_requests_count(current_user: User, duration: float, db_session: Session):
    """
    Increment the request count and processing time for the current user.

    Args:
        current_user (User): The user making the request.
        duration (float): The duration of the request in seconds.
        db_session (Session): The database session to use for the operation.

    Raises:
        HTTPException: If there is an error during the database operation.
    """
    try:
        user_db = (db_session.query(UserAccessDB).filter_by(email=current_user.email).first())
        user_db.requests_count += 1
        user_db.secs_count += duration
        db_session.commit()
    except Exception as e:
        db_session.rollback()
        logger.error(f"Rollback the transaction cause by the exception {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error")


# OAuth2PasswordBearer is a class provided by FastAPI to handle OAuth2 authentication.
# It retrieves the token from the request and verifies it.
# The tokenUrl parameter specifies the URL where the client can get the token.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login_for_access_token")


def get_and_check_user(access_token: str = Depends(oauth2_scheme), db_session: Session = Depends(get_db)) -> User:
    """
    Retrieve and validate the current user based on the provided access token.

    Args:
        access_token (str): The OAuth2 access token provided by the client.
        db_session (Session): The database session to use for the operation.

    Returns:
        User: The authenticated user.

    Raises:
        HTTPException: If the token is invalid, expired, or if there is a database error.
    """
    try:
        decoded_token = jwt.decode(access_token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.DecodeError:
        msg = "Invalid token"
        logger.error(msg)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=msg)

    token_expiration_time = decoded_token.get('token_expiration_time', None)
    if token_expiration_time is not None and datetime.utcnow() > datetime.fromisoformat(token_expiration_time):
        msg = "Token has expired"
        logger.error(msg)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=msg)

    email = decoded_token['email']
    verify_user_usage_limits(email, db_session)
    verify_user_time_limit(email, db_session)

    organization = decoded_token['organization']
    return User(email=email, organization=organization)


async def impose_file_size_limit(file: UploadFile = File(...)):
    """
    Check if the uploaded file exceeds the maximum allowed size or is empty.

    Args:
        file (UploadFile): The uploaded file to check.

    Raises:
        HTTPException: If the file size exceeds the allowed limit or if the file is empty.
    """
    current_position = file.file.tell()
    file.file.seek(0, 2)  # Move to the end of the file
    total_bytes = file.file.tell()  # Get total bytes
    file.file.seek(current_position)  # Reset the file position to the original

    if total_bytes > MAX_FILE_SIZE_BYTES:
        msg = f"File size exceeds the allowed limit of {MAX_FILE_SIZE_BYTES} bytes"
        logger.error(msg)
        raise HTTPException(status_code=413, detail=msg)

    if total_bytes == 0:
        msg = "Uploaded file is empty"
        logger.error(msg)
        raise HTTPException(status_code=400, detail=msg)


@app.post("/login_for_access_token")
async def login_for_access_token(auth_in: AuthenticationIn, db_session: Session = Depends(get_db)) -> Union[
    Token, ErrorResponse]:
    """
    Authenticate the user and generate an access token.

    Args:
        auth_in (AuthenticationIn): The authentication input containing email and password.
        db_session (Session): The database session to use for the operation.

    Returns:
        Union[Token, ErrorResponse]: The access token if authentication is successful, or an error response.

    Raises:
        HTTPException: If there is an error during the authentication process or token generation.
    """
    email = auth_in.email
    password = auth_in.password

    verify_password(email, password, db_session)
    verify_user_time_limit(email, db_session)
    verify_user_usage_limits(email, db_session)
    try:
        access_token = create_access_token(email, db_session)
        return Token(access_token=access_token, token_type="bearer")
    except HTTPException as http_exc:
        logger.error(f"Login failed for user '{email}': {http_exc.detail}")
        logger.error(f"HTTPException: {http_exc.detail}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error")
    except Exception as e:
        logger.error(f"Login failed for user {email}")
        logger.error(f"Exception: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error")


@app.get("/protected_route")
async def protected_route(current_user: User = Depends(get_and_check_user)) -> ProtectedOut:
    """
    Handle the protected route request.

    Args:
        current_user (User): The current authenticated user.

    Returns:
        ProtectedOut: The response containing the logged-in user information.
    """
    return ProtectedOut(logged_in_as=current_user)


@app.post("/is_voice_ai_gen/")
async def predict_is_voice_ai_gen(audio_file: UploadFile = File(...), current_user: User = Depends(get_and_check_user),
                                  db_session: Session = Depends(get_db)) -> Union[PredictionProbOut, ErrorResponse]:
    """
    Predict if the uploaded audio file is AI-generated.

    Args:
        audio_file (UploadFile): The uploaded audio file to analyze.
        current_user (User): The current authenticated user.
        db_session (Session): The database session to use for the operation.

    Returns:
        Union[PredictionProbOut, ErrorResponse]: The prediction probability if successful, or an error response.

    Raises:
        HTTPException: If there is an error during the file size check or prediction process.
    """
    try:
        await impose_file_size_limit(audio_file)
        audio_bytes = await audio_file.read()
        score, duration = predict_is_ai(io_file=io.BytesIO(audio_bytes), extractor=extractor_ai_voice)
        response = PredictionProbOut(score=score)
        add_requests_count(current_user, duration, db_session)
        return response
    except HTTPException as http_exc:
        logger.error(http_exc.detail)
        return ErrorResponse(error="Internal error")
    except Exception:
        error_trace = traceback.format_exc()
        logger.error(error_trace)
        return JSONResponse(status_code=500, content=str(error_trace))


@app.post("/size_audio/")
async def size_audio(audio_file: UploadFile = File(...), current_user: User = Depends(get_and_check_user),
                     db_session: Session = Depends(get_db)) -> Union[FileSizeOut, ErrorResponse]:
    """
    Calculate the size of the uploaded audio file in megabytes.

    Args:
        audio_file (UploadFile): The uploaded audio file to analyze.
        current_user (User): The current authenticated user.
        db_session (Session): The database session to use for the operation.

    Returns:
        Union[FileSizeOut, ErrorResponse]: The size of the audio file in megabytes if successful, or an error response.

    Raises:
        HTTPException: If there is an error during the file size check or file reading process.
    """
    try:
        await impose_file_size_limit(audio_file)
        audio_bytes = await audio_file.read()
        size_in_bytes = io.BytesIO(audio_bytes).getbuffer().nbytes
        size_in_mb = size_in_bytes / (1024 ** 2)
        add_requests_count(current_user, 0, db_session)
        return FileSizeOut(size_in_mb=size_in_mb)
    except HTTPException as http_exc:
        logger.error(http_exc.detail)
        return ErrorResponse(error="Internal error")


def main():
    """
    Start the FastAPI application using Uvicorn.

    This function retrieves the host and port configuration from environment variables
    and runs the FastAPI application using Uvicorn.

    Environment Variables:
        APP_HOST (str): The host address to bind the server to. Defaults to "0.0.0.0".
        APP_PORT (int): The port number to bind the server to. Defaults to 8000.

    """
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = os.getenv("APP_PORT", 8000)
    uvicorn.run("apis.main_api_db:app", host=host, port=int(port), workers=1, reload=False)


if __name__ == "__main__":
    main()
