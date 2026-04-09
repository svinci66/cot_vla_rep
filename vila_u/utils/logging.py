import logging
import typing

if typing.TYPE_CHECKING:
    from loguru import Logger as LoguruLogger
    Logger = typing.Union[LoguruLogger, logging.Logger]
else:
    Logger = typing.Any

__all__ = ["logger"]


def __get_logger() -> Logger:
    try:
        from loguru import logger as loguru_logger

        return loguru_logger
    except ModuleNotFoundError:
        std_logger = logging.getLogger("vila_u")
        if not std_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
            )
            std_logger.addHandler(handler)
        std_logger.setLevel(logging.INFO)
        return std_logger


logger = __get_logger()
