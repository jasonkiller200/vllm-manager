#!/usr/bin/env python3
"""Tunnel SSH through an HTTP proxy using CONNECT method."""
import sys
import socket
import select

PROXY_HOST = "192.168.6.119"
PROXY_PORT = 8080

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <host> <port>", file=sys.stderr)
        sys.exit(1)

    target_host = sys.argv[1]
    target_port = int(sys.argv[2])

    try:
        # Connect to proxy
        proxy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        proxy.settimeout(10)
        proxy.connect((PROXY_HOST, PROXY_PORT))
        proxy.settimeout(None)

        # Send HTTP CONNECT request
        connect_req = f"CONNECT {target_host}:{target_port} HTTP/1.1\r\nHost: {target_host}:{target_port}\r\n\r\n"
        proxy.send(connect_req.encode())

        # Read response
        response = b""
        while b"\r\n\r\n" not in response:
            chunk = proxy.recv(1024)
            if not chunk:
                print("Proxy connection closed", file=sys.stderr)
                sys.exit(1)
            response += chunk

        # Check response status
        first_line = response.split(b"\r\n")[0].decode()
        if "200" not in first_line:
            print(f"Proxy error: {first_line}", file=sys.stderr)
            sys.exit(1)

        # Bridge data between stdin/stdout and the socket
        stdin = sys.stdin.buffer
        stdout = sys.stdout.buffer

        while True:
            readable, _, _ = select.select([proxy, stdin], [], [])
            if proxy in readable:
                data = proxy.recv(4096)
                if not data:
                    break
                stdout.write(data)
                stdout.flush()
            if stdin in readable:
                data = stdin.read(4096)
                if not data:
                    break
                proxy.send(data)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
