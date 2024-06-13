import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# now connect to the web server on port 80 - the normal http port
s.connect(("10.0.2.2", 30002))
# s.send(b"fx = get_sensor_fx()\n")
s.send(b"sync()\n")
s.send(b"textmsg('value')\n")
s.send(b"sync()\n")