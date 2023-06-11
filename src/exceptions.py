import sys
from src.logger import logging

def error_msg_detail(error, detail:sys):
    """
    Returns a message if error is encountered.
    """
    _, _, exc_tb = detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    error_msg = ("\nError occured at:\n"
                 "Script name: [{0}]\n"
                 "Line number: [{1}]\n"
                 "Error message: [{2}]").format(filename, exc_tb.tb_lineno, str(error))
    return error_msg
    
class CustomException(Exception):
    def __init__(self, error_msg, detail:sys):
        super().__init__(error_msg)
        self.error_msg = error_msg_detail(error_msg, detail)

    def __str__(self):
        return self.error_msg