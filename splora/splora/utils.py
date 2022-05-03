"""Utils of splora."""
import logging

LGR = logging.getLogger("GENERAL")
RefLGR = logging.getLogger("REFERENCES")


def setup_loggers(logname=None, refname=None, quiet=False, debug=False):
    """Set up loggers.

    Parameters
    ----------
    logname : str, optional
        Name of the log file, by default None
    refname : str, optional
        Name of the reference file, by default None
    quiet : bool, optional
        Whether the logger should run in quiet mode, by default False
    debug : bool, optional
        Whether the logger should run in debug mode, by default False
    """
    # Set up the general logger
    log_formatter = logging.Formatter(
        "%(asctime)s\t%(module)s.%(funcName)-12s\t%(levelname)-8s\t%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    stream_formatter = logging.Formatter(
        "%(levelname)-8s %(module)s:%(funcName)s:%(lineno)d %(message)s"
    )
    # set up general logging file and open it for writing
    if logname:
        log_handler = logging.FileHandler(logname)
        log_handler.setFormatter(log_formatter)
        LGR.addHandler(log_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    LGR.addHandler(stream_handler)

    if quiet:
        LGR.setLevel(logging.WARNING)
    elif debug:
        LGR.setLevel(logging.DEBUG)
    else:
        LGR.setLevel(logging.INFO)

    # Loggers for references
    text_formatter = logging.Formatter("%(message)s")

    if refname:
        ref_handler = logging.FileHandler(refname)
        ref_handler.setFormatter(text_formatter)
        RefLGR.setLevel(logging.INFO)
        RefLGR.addHandler(ref_handler)
        RefLGR.propagate = False


def teardown_loggers():
    """Remove logger handler."""
    for local_logger in (RefLGR, LGR):
        for handler in local_logger.handlers[:]:
            handler.close()
            local_logger.removeHandler(handler)
