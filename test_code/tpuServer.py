from bottle import route, run

@route('/')
def index():
    return "<h1>Welcome to the TPU Server</h1><p>Object detection is running.</p>"

run(host='0.0.0.0', port=8081, debug=True, reloader=True)
