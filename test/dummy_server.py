from http.server import SimpleHTTPRequestHandler, HTTPServer


class HelloHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"Hello, World!")


def run(server_class=HTTPServer, handler_class=HelloHandler, port=8000):
    server_address = ('192.168.10.114', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting httpd server on port {port}...')
    httpd.serve_forever()


if __name__ == '__main__':
    run()
