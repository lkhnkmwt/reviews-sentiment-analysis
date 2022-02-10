import os

class file_operations:
    def __init__(self,path):
        self.path=path

    def delete_if_exists(self):
        if os.path.exists(self.path):
            os.remove(self.path)
            return 1
        else:
            return 0

