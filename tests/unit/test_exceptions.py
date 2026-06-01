"""Unit tests for custom exception hierarchy."""

from http import HTTPStatus

import pytest

from src.shared.exceptions import (
    AuthenticationError,
    AuthorizationError,
    DevalignException,
    ExternalServiceError,
    FileTooLargeError,
    MLPipelineError,
    NotFoundError,
    RAGPipelineError,
)


def test_base_exception_default_detail() -> None:
    exc = DevalignException()
    assert exc.detail == "An unexpected error occurred"
    assert exc.status_code == HTTPStatus.INTERNAL_SERVER_ERROR


def test_base_exception_custom_detail() -> None:
    exc = DevalignException("Custom message")
    assert exc.detail == "Custom message"


def test_not_found_error() -> None:
    exc = NotFoundError("User not found")
    assert exc.status_code == HTTPStatus.NOT_FOUND
    assert exc.detail == "User not found"


def test_auth_error() -> None:
    exc = AuthenticationError()
    assert exc.status_code == HTTPStatus.UNAUTHORIZED


def test_authorization_error() -> None:
    exc = AuthorizationError()
    assert exc.status_code == HTTPStatus.FORBIDDEN


def test_file_too_large() -> None:
    exc = FileTooLargeError()
    assert exc.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE


def test_ml_pipeline_error() -> None:
    exc = MLPipelineError("Pipeline failed")
    assert exc.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert "Pipeline failed" in exc.detail


def test_external_service_error() -> None:
    exc = ExternalServiceError()
    assert exc.status_code == HTTPStatus.BAD_GATEWAY


def test_exception_is_raised() -> None:
    with pytest.raises(NotFoundError, match="Resource not found"):
        raise NotFoundError()


def test_rag_pipeline_error() -> None:
    exc = RAGPipelineError("LLM failed")
    assert exc.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
