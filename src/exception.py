import sys
from src.logger import logging

def error_message_detail(error,error_detail:sys):

    # Extracting the type, file name, line number and error message
    _,_,exc_tb=error_detail.exc_info() # This will give us the traceback object
    # exc_tb is a traceback object which contains the stack trace of the error
    # We can extract the file name, line number and error message from it

    file_name=exc_tb.tb_frame.f_code.co_filename # This will give us the file name where the error occurred
    # tb_frame is the frame object which contains the code object
    # co_filename is the attribute of the code object which contains the file name
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
     file_name,exc_tb.tb_lineno,str(error))
    # tb_lineno is the attribute of the traceback object which contains the line number where the error occurred
    # exc_info() returns a tuple of (type, value, traceback) where type is the type of the exception, value is the exception object and traceback is the traceback object

    return error_message

    

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys): 
        super().__init__(error_message)# This will call the constructor of the parent class Exception
        # We are passing the error message to the parent class constructor
        # We are also passing the error detail to the parent class constructor
        self.error_message=error_message_detail(error_message,error_detail=error_detail)
    
    def __str__(self): # This method is called when we print the object of this class
        # It will return the error message
        return self.error_message
    


        