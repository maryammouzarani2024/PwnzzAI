import os

_BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def _sqlite_uri(path: str) -> str:
    return "sqlite:///" + os.path.normpath(path).replace("\\", "/")


# Same directory as the main shop DB (pizza_shop.db): no extra folder required.
_DEFAULT_CATERING_SQL_URI = _sqlite_uri(os.path.join(_BASE_DIR, "catering_sql_lab.db"))
_CATERING_SQL_DATABASE_URI = os.environ.get("CATERING_SQL_DATABASE_URI", _DEFAULT_CATERING_SQL_URI)


class Config(object):

    # Main shop database (pizzas, users, comments, orders)
    SQLALCHEMY_DATABASE_URI = "sqlite:///pizza_shop.db"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Isolated SQLite DB for the catering agentic SQL / tool lab only (routing_flag)
    CATERING_SQL_DATABASE_URI = _CATERING_SQL_DATABASE_URI
    SQLALCHEMY_BINDS = {
        "catering_sql": _CATERING_SQL_DATABASE_URI,
    }

    # Secret key for session management
    SECRET_KEY = os.environ.get("SECRET_KEY") or "dev-secret-key-change-in-production"
