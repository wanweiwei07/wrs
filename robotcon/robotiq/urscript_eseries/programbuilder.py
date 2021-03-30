import logging

class ProgramBuilder(object):

    def __init__(self):
        self.logger = logging.getLogger("urx")
        self.complete_program = ""
        self.logger = False

    def loadprog(self, filename):
        # clear program
        self.complete_program = ""
        self.__file = open(filename, "r")
        partscript = self.__file.read(1024)
        while partscript:
            self.complete_program += partscript
            partscript = self.__file.read(1024)

    # def receiveftdata(self):
    #     import socket
    #     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #     # now connect to the web server on port 80 - the normal http port
    #     s.connect(("10.2.0.50", 63351))
    #     while True:
    #         print s.recv(1024)

    def ret_program_to_run(self):
        if(self.complete_program == ""):
            self.logger.debug("impedance control using ftsensor's program is empty")
            return ""
        return self.complete_program

if __name__=="__main__":
    import robotconn.ur3e as ur3ex

    pblder = ProgramBuilder()
    pblder.loadprog("skills.script")
    prog = pblder.ret_program_to_run()
    prog = prog.replace("parameter_massm", "10")
    prog = prog.replace("parameter_stroke", "0.035")

    # pblder.loadprog("peginholespiral.script")
    # pblder.loadprog("tst.script")
    rx = ur3ex.Ur3EUrx()
    rx.arm.send_program(prog)
    # wait until program runs
    while not rx.arm.is_program_running():
        pass