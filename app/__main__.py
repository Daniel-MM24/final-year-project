"""CLI entry point.

    python -m app                          # offline, 127.0.0.1:8000
    python -m app --port 9000
    python -m app --live                   # enables the Draft button
    python -m app --host 0.0.0.0 --token X # requires --token

The token is checked on every request via the ``X-Viewer-Token`` header or a
``?token=`` query parameter. It is *not* a security boundary (no HTTPS); it
is a speed bump so that "bind broadly" is never the default.
"""

from __future__ import annotations

import argparse
import sys

from .server import create_app

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m app", description=__doc__)
    p.add_argument("--host", default=DEFAULT_HOST,
                   help=f"bind address (default {DEFAULT_HOST})")
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help=f"TCP port (default {DEFAULT_PORT})")
    p.add_argument("--token", default=None,
                   help="required when --host is not 127.0.0.1")
    p.add_argument("--live", action="store_true",
                   help="enable the Draft button and call the real agent")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.host not in ("127.0.0.1", "localhost") and not args.token:
        print("refusing to bind a non-loopback address without --token",
              file=sys.stderr)
        return 2

    import uvicorn  # imported only when serving

    app = create_app(live=bool(args.live), token=args.token)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
