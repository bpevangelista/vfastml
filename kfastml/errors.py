class UserError(Exception):
    status_code = 400

    def __init__(self, message: str):
        super().__init__(message)


class InvalidRequestError(UserError):
    status_code = 400
    pass


class UnsupportedError(UserError):
    status_code = 400
    pass


class AuthenticationError(UserError):
    status_code = 401
    pass


class RateLimitError(UserError):
    status_code = 429
    pass


class QuotaLimitError(UserError):
    status_code = 429
    pass


class InternalServerError(Exception):
    status_code = 500

    def __init__(self, message="Something bad happened"):
        super().__init__(message)


class InternalServerTooBusyError(InternalServerError):
    status_code = 503

    def __init__(self, message="Server is too busy! Try again later"):
        super().__init__(message)


class ErrorResult:
    def __init__(self, status_code: int, error_code: str, error_message: str):
        self.status_code = status_code
        self.error = {
            'code': error_code,
            'message': error_message
        }

    @staticmethod
    def from_exception(error: Exception):
        status_code = error.status_code if isinstance(error, UserError) else 500
        return ErrorResult(status_code, type(error).__name__, str(error))
