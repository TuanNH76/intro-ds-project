from fastapi import HTTPException


class _ApiError(HTTPException):
    def __init__(self, detail, status_code):
        super().__init__(detail=detail, status_code=status_code)


class _InternalError(Exception):
    def __init__(self, detail):
        super().__init__(detail)


class ApiBadRequest(_ApiError):
    def __init__(self, detail):
        status_code = 400
        detail = "Bad Request: " + detail
        super().__init__(detail=detail, status_code=status_code)


class ApiUnauthorized(_ApiError):
    def __init__(self, detail):
        status_code = 401
        detail = "Unauthorized: " + detail
        super().__init__(detail=detail, status_code=status_code)


class ApiNotFound(_ApiError):
    def __init__(self, detail):
        status_code = 404
        detail = "Not Found: " + detail
        super().__init__(detail=detail, status_code=status_code)


class ApiCannotHandle(_ApiError):
    def __init__(self, detail):
        status_code = 405
        detail = "Cannot Handle: " + detail
        super().__init__(detail=detail, status_code=status_code)


class ApiOverload(_ApiError):
    def __init__(self, detail):
        status_code = 429
        detail = "Overload: " + detail
        super().__init__(detail=detail, status_code=status_code)


class Overload(_InternalError):
    def __init__(self, detail):
        super().__init__(detail)


class ApiInternalError(_ApiError):
    def __init__(self, detail):
        status_code = 500
        detail = "Internal Error: " + detail
        super().__init__(detail=detail, status_code=status_code)


class FailedExternalAPI(_InternalError):
    """
    The error for the failed external API call.

    Args:
        detail (str): inherited from the API error message.
    """

    def __init__(self, detail):
        super().__init__(detail)


class LimitedRequest(_InternalError):
    """
    The error for the limited request.

    Args:
        detail (str): the message for user.
    """

    def __init__(self, detail):
        super().__init__(detail)


class MissingInformationError(_InternalError):
    """
    Custom exception for missing information.

    Args:
        message (str): The error message describing the missing information.
    """

    def __init__(self, message: str):
        super().__init__(message)


class ResourceNotFound(_InternalError):
    """
    Custom exception for resource not found.

    Args:
        message (str): The error message indicating the resource does not exist.
    """

    def __init__(self, message: str):
        super().__init__(message)


class CanNotAnswer(_InternalError):
    """
    Custom exception for errors related to OpenAI's function calling.

    This exception is raised when OpenAI's function calling mechanism
    is unable to process the request or fails to generate a valid response.
    It indicates that the system cannot proceed due to issues within
    OpenAI's function call handling, such as incorrect parameters,
    unsupported operations, or internal API errors.

    Args:
        message (str): A descriptive error message indicating the reason
                       for the failure in OpenAI's function calling process.
    """

    def __init__(self, message: str):
        super().__init__(message)


class CannotProcessOutput(_InternalError):
    """
    Custom exception for errors related to OpenAI's function calling.

    This exception is raised when OpenAI's function calling mechanism
    is unable to process the request or fails to generate a valid response.
    It indicates that the system cannot proceed due to issues within
    OpenAI's function call handling, such as incorrect parameters,
    unsupported operations, or internal API errors.
    Args:

        message (str): A descriptive error message indicating the reason
                        for the failure in OpenAI's function calling process.
    """

    def __init__(self, message: str):
        super().__init__(message)


class HandledError(_InternalError):
    def __init__(self, message):
        super().__init__(message)