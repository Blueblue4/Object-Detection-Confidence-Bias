import json
import logging
import pathlib
import datetime
import sys
from typing import Any, Union, Optional

import numpy as np
import pandas as pd


def latest_date(date_list: list[str], date_format: str = "%Y-%m-%d-%H-%M-%S") -> str:
    return sorted(date_list, key=lambda x: datetime.datetime.strptime(x, date_format))[
        -1
    ]


def append_dict_entries(appended_dict: dict, dict_to_append: dict, prefix: str = ""):
    appended_dict.update(
        {
            prefix + key: appended_dict[prefix + key] + [val]
            for key, val in dict_to_append.items()
        }
    )
    return appended_dict


def default_np(obj: Any):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()


def ensure_list(obj: Any):
    if isinstance(obj, list):
        return obj
    else:
        return [obj]


def save_dict(dict_to_save: dict, path: Union[str, pathlib.Path]):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict_to_save, open(path, "w"), default=default_np)


class DisableLoggerContext(object):
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


def save_dict_as_table(
    all_results: Union[dict, list],
    ignore_cols: Optional[Union[list[str], str]] = None,
    name: str = "default",
):
    logging.info("saving results")
    df_results = pd.DataFrame(all_results)
    df_results.to_pickle(f"{name}.pkl")
    if ignore_cols is not None:
        df_results = df_results[
            [
                col_name
                for col_name in df_results.columns
                if all([ig_name not in col_name for ig_name in ignore_cols])
            ]
        ]
    df_results.to_latex(f"{name}_table.tex", float_format="{:0.2f}".format)
    logging.info(df_results.to_latex(float_format="{:0.2f}".format))


def make_logger(name: str, log_exceptions: bool = True):
    logger = logging.getLogger(name)
    if log_exceptions:

        def log_except(exception_type, exception_value, exception_traceback):
            logger.exception(
                "Uncaught Exception!",
                exc_info=(exception_type, exception_value, exception_traceback),
            )

        logging.captureWarnings(True)
        sys.excepthook = log_except
    return logger
