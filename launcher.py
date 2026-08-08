"""Minimal launcher for Black Diamond container entrypoint."""

from __future__ import annotations

import argparse
import http.server
import socketserver
import sys
from threading import Thread
from typing import Iterable

from black_diamond import __version__


class LauncherHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        print(format % args)


def serve(port: int) -> None:
    handler = LauncherHTTPRequestHandler
    with socketserver.TCPServer(("0.0.0.0", port), handler) as httpd:
        print(f"🚀 Black Diamond launcher running on http://0.0.0.0:{port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


def run_examples(names: Iterable[str]) -> None:
    for name in names:
        try:
            module_name = name.replace("-", "_")
            __import__(f"examples.{module_name}")
        except ModuleNotFoundError:
            print(f"Example '{name}' not found.")
            sys.exit(1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="black-diamond")
    subparsers = parser.add_subparsers(dest="command")

    start_parser = subparsers.add_parser("start", help="Start the Black Diamond launcher")
    start_parser.add_argument("--all", action="store_true", help="Start all launcher services")
    start_parser.add_argument("--backend", action="store_true", help="Start backend service only")
    start_parser.add_argument("--port", type=int, default=3000, help="Port for the service")

    subparsers.add_parser("version", help="Print current package version")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "start":
        port = args.port if not args.backend else 5000
        if args.all:
            port = 3000
        elif args.backend:
            port = 5000
        print(f"Starting Black Diamond launcher (version {__version__})")
        serve(port)
        return 0

    if args.command == "version":
        print(__version__)
        return 0

    print("No command specified. Use --help for available commands.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
