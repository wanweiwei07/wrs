class ProgramBuilder(object):

    def __init__(self):
        self.complete_program = ""

    def load_prog(self, filename):
        self.complete_program = ""
        self._file = open(filename, "r")
        partscript = self._file.read(1024)
        while partscript:
            self.complete_program += partscript
            partscript = self._file.read(1024)

    def get_program_to_run(self):
        if (self.complete_program == ""):
            self.logger.debug("The given script is empty!")
            return ""
        return self.complete_program
