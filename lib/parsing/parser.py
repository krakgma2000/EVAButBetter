class Parser:
    def __init__(self,data):
        self.data = data
    def parse(self,filename):
        raise NotImplementedError

    def to_file(self,data, filename):
        raise NotImplementedError