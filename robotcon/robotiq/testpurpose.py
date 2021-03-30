import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# now connect to the web server on port 80 - the normal http port
s.connect(("10.2.0.50", 30002))
s.send("fx = get_sensor_fx()"+"\n")
s.send("sync()"+"\n")
s.send("textmsg('test')"+"\n")
s.send("sync()"+"\n")