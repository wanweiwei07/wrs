# Echo client program
import socket

HOST = "10.2.0.50" # The UR IP address
PORT = 30002 # UR secondary client
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

s.send("set_digital_out(1, False)"+"\n")
print(s.recv(1024))
# f = open ("rtq_cbseries_hand.script", "rb")   #Robotiq Gripper
# #f = open ("setzero.script", "rb")  #Robotiq FT sensor
#
# l = f.read(1024)
# while (l):
#     s.send(l)
#     l = f.read(1024)
# s.close()



