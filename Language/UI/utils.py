import re
import os
import sys
import time
import datetime
import traceback
import numpy as np
import pandas as pd
from colorama import init, Fore, Style
from PySide6.QtCore import (QObject, QRunnable, QTimer, Signal, Slot,
                        Qt, QSize, QAbstractTableModel, QSortFilterProxyModel)
from PySide6.QtWidgets import (QMessageBox, QApplication, QStyle, QPushButton,
                             QDialog, QDialogButtonBox, QGridLayout, QLabel, QStyle)
from PySide6.QtGui import QIcon


init(autoreset=True, convert=False, strip=False)

VENV_PATH = os.path.abspath("C:/Users/Tom/miniconda3/envs/nlpenv/python.exe")
ERROR_PATH = os.path.abspath("/UI/ui_files/error_log.txt")


# ANSI escape character conversion to HTML for displaying rich text color in the console widget

TO_HTML = {
    # Foreground colors
    "\033[30m": "<font color=\"#FFAA00\">",  # Black -> Orange (replacement)
    "\033[31m": "<font color=\"#FF5555\">",  # Red -> Brighter Red
    "\033[32m": "<font color=\"#AAFF00\">",  # Green -> Brighter Green
    "\033[33m": "<font color=\"#FFFF00\">",  # Yellow -> Brighter Yellow
    "\033[34m": "<font color=\"#00AAFF\">",  # Blue -> Brighter Blue
    "\033[35m": "<font color=\"#AAAAFF\">",  # Magenta -> Light Purple (replacement)
    "\033[36m": "<font color=\"#00FFFF\">",  # Cyan -> Brighter Cyan
    "\033[37m": "<font color=\"#FFFFFF\">",  # White
    # Background colors
    "\033[40m": "<span style=\"background-color:black\">",       # Black
    "\033[41m": "<span style=\"background-color:red\">",         # Red
    "\033[42m": "<span style=\"background-color:green\">",       # Green
    "\033[43m": "<span style=\"background-color:yellow\">",      # Yellow
    "\033[44m": "<span style=\"background-color:blue\">",        # Blue
    "\033[45m": "<span style=\"background-color:magenta\">",     # Magenta
    "\033[46m": "<span style=\"background-color:cyan\">",        # Cyan
    "\033[47m": "<span style=\"background-color:white\">",       # White
    # Styles
    "\033[1m": "<b>",                        # Bold
    "\033[4m": "<u>",                        # Underline
    # Reset 
    "\033[0m": "</font></span>",             # Reset all
    "\033[39m": "</font>",                   # Reset foreground
    "\033[49m": "</span>",                   # Reset background
    # Special - custom codes to convert to html
    "<1>": "&nbsp;",                         # Insert 1 space
}

def formatted_time(start_time):
    """Formats processing time."""
    elapsed_time = time.time() - start_time
    # Convert to minutes and seconds
    minutes = int(elapsed_time // 60)
    seconds = float(elapsed_time % 60)
    return f"{Fore.CYAN}{minutes}{Fore.RESET} mins {Fore.CYAN}{seconds:.3f}{Fore.RESET} secs"

# def formated_time(start_time):
#     """Formats processing time."""
#     elapsed_time = time.time() - start_time
#     # Calculate hours, minutes and seconds
#     hours = int(elapsed_time // 3600)
#     minutes = int((elapsed_time % 3600) // 60)
#     seconds = elapsed_time % 60
#     return f"{Fore.CYAN}{hours}{Fore.RESET} hrs {Fore.CYAN}{minutes}{Fore.RESET} mins {Fore.CYAN}{seconds:.3f}{Fore.RESET} secs"

############################# Console Logger Class for Main Window #############################

class ConsoleLogger(QObject):
    """Logger used to redirect stdout (print) / stderr 
    and will append to the console QTextEdit widget"""
    def __init__(self, console):
        super(ConsoleLogger, self).__init__()
        self.console = console


    def convert_html(self, text):
        """Convert ANSI characters to HTML tags 
        to print color text to the console"""
        for ansi, html in TO_HTML.items():
            text = text.replace(ansi, html)
        return text


    def contains_html_tags(self, text):
        # Checks if the text contains HTML tags
        return bool(re.search(r'<[^<]+?>', text))


    def write(self, text):
        """Writes all redirected stdout/stderr messages 
        to the console widget"""
        rich_text = self.convert_html(text)
        cursor = self.console.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        if self.contains_html_tags(rich_text):
            rich_text = rich_text.replace('\n', '<br>')
            cursor.insertHtml(rich_text)
        else:
            cursor.insertText(rich_text)
        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()


    def flush(self):
        pass
        #sys.stdout.flush()

########################## Signals and Worker Classes for QThreadPool ##########################

class Signals(QObject):
    startProcess = Signal(str, list)
    startPlotModel = Signal(object, object, str)
    progress = Signal(int)
    update = Signal()
    exception = Signal()
    result = Signal(object)
    finished = Signal()
    enterPressed = Signal(str)


class Worker(QRunnable):
    """Thread worker for QThreadpool"""
    def __init__(self, func, *args, **kwargs):
        super(Worker, self).__init__()
        # Load the worker function and arguments
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.signals = Signals()


    @Slot()
    def run(self):
        try:  # Pass the function with arguments
            result = self.func(*self.args, **self.kwargs)
            if result is not None: # Check if the function produces a result and send
                self.signals.result.emit(result) 
        except Exception as e:
            # Format and log any uncaught exceptions in the worker thread
            exc_info = sys.exc_info()
            ExceptionLogger.log_exception(exc_info)
            # Trigger the error message box
            self.signals.exception.emit()
        finally:
            self.signals.finished.emit()


#################### Basic Pandas Model and Sort/Filter Model Classes ####################

class BasicPandasModel(QAbstractTableModel):
    """Abstract Table Model to handle Pandas Dataframe Objects for display in QTableView widget"""
    def __init__(self):
        super(BasicPandasModel, self).__init__()
        ################## Class Attribute Variables ##################
        self.errorMessage = ErrorMessageBox()
        self._data = pd.DataFrame()
        self._master_data = pd.DataFrame()
        # Pagination starting values
        self.rows_per_page = 100
        self.current_page = 0

    ########################## Class Methods #########################
        
    def loadData(self, df):
        self._data = df.copy()
        self._master_data = df
        self.original_index = df.index  # Save the original index


    def rowCount(self, index):
        return min(self.rows_per_page, len(self._data) - self.current_page * self.rows_per_page)


    def columnCount(self, index):
        return self._data.shape[1]


    def data(self, index, role):
        """How pandas data is translated and displayed in the model view"""
        # Adjusting row index to consider pagination
        row_idx = index.row() + self.current_page * self.rows_per_page
        value = self._data.iloc[row_idx, index.column()]
        # Data display types for the table model view
        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            if isinstance(value, datetime.datetime):
                if value.time() == datetime.time(0, 0):
                    return value.strftime("%Y-%m-%d") # Display just the date if no time exists in the format
                else:
                    return value.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(value, int):
                return int(value)
            elif isinstance(value, float):
                return float(value)
            else:
                return str(value)

        return None


    def setData(self, index, value, role):
        """Update the dataframe from user changes in the model view"""
        if role == Qt.ItemDataRole.EditRole:
            # Adjusting row index to consider pagination
            row_idx = index.row() + self.current_page * self.rows_per_page
            # Change the cell value to the translated value of the working dataframe
            self._data.iat[row_idx, index.column()] = value
            # Get the index column number and cell values to locate the same cell in the unfiltered master data
            index_idx = self._data.columns.get_loc('Index')
            index_value = self._data.iat[row_idx, index_idx]
            # Change the same cell in the master data based on the matching index column value of _data
            self._master_data.iat[index_value, index.column()] = value
            
            self.dataChanged.emit(index, index, [role])  # Emitting dataChanged signal
            return True
        return False


    def flags(self, index):
        return Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable


    def headerData(self, section, orientation, role):
        """Display header column names in the model view"""
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._data.columns[section])
            if orientation == Qt.Orientation.Vertical:
                return str(self._data.index[section])


    def sort(self, column, order):
        """Sort table by given column name or number"""
        self.layoutAboutToBeChanged.emit()

        if order == Qt.SortOrder.AscendingOrder:
            sort_order = Qt.SortOrder.AscendingOrder 
        else: 
            sort_order = Qt.SortOrder.DescendingOrder
        self._data.sort_values(by=self._data.columns[column], axis=0, ascending=sort_order == Qt.SortOrder.AscendingOrder, inplace=True)

        self.layoutChanged.emit()


    def searchData(self, column, operator, query):
        """Search dataframe columns based on operator type and query"""
        self.beginResetModel()
        self.current_page = 0  # Reset to the first page

        if self._data[column].dtype == object:
            if operator == 'contains':
                search_idxs = self._data[self._data[column].str.contains(query, case=False, na=False)].index
            elif operator == 'equals': 
                search_idxs = self._data[self._data[column].str.contains(rf'\b{query}\b', case=False, na=False)].index
            else:
                self.errorMessage.critical("Incorrect Data Type", "Greater or less than does not work in columns with text values.")
                return
        else:
            if query.isnumeric():
                if operator == 'contains' or operator == 'equals':
                    search_idxs = self._data[self._data[column] == float(query)].index
                elif operator == 'greater than':
                    search_idxs = self._data[self._data[column] >= float(query)].index
                elif operator == 'less than':
                    search_idxs = self._data[self._data[column] <= float(query)].index
            else:
                self.errorMessage.critical("Incorrect Data Type", "Must enter a number to search columns with numeric values.")
                return

        self._data = self._data.loc[search_idxs]

        self.endResetModel()


    def resetData(self):
        """Reset the dataframe back to its originally loaded state"""
        self.beginResetModel()
        self._data = self._master_data.loc[self.original_index]  # Reset to the original index
        self.current_page = 0  # Reset to the first page
        self.endResetModel()


    def prevPage(self):
        if self._data.empty:  # Check if the DataFrame is empty
            self.errorMessage.critical("Invalid Entry", "You cannot page an empty data table. Reset the table before pressing the page buttons.")
            return  # Exit the method early if DataFrame is empty
        self.beginResetModel()
        self.current_page = max(0, self.current_page - 1)  # Ensure it doesn’t go below 0
        self.endResetModel()


    def firstPage(self):
        if self._data.empty:  # Check if the DataFrame is empty
            self.errorMessage.critical("Invalid Entry", "You cannot page an empty data table. Reset the table before pressing the page buttons.")
            return  # Exit the method early if DataFrame is empty
        self.beginResetModel()
        self.current_page = 0  # Go to the first page
        self.endResetModel()


    def nextPage(self):
        if self._data.empty:  # Check if the DataFrame is empty
            self.errorMessage.critical("Invalid Entry", "You cannot page an empty data table. Reset the table before pressing the page buttons.")
            return  # Exit the method early if DataFrame is empty
        self.beginResetModel()
        total_pages = -(-len(self._data) // self.rows_per_page)  # Calculate total pages with ceiling division
        self.current_page = min(total_pages - 1, self.current_page + 1)  # Ensure it doesn’t go above total pages
        self.endResetModel()


    def lastPage(self):
        if self._data.empty:  # Check if the DataFrame is empty
            self.errorMessage.critical("Invalid Entry", "You cannot page an empty data table. Reset the table before pressing the page buttons.")
            return  # Exit the method early if DataFrame is empty
        self.beginResetModel()
        total_pages = -(-len(self._data) // self.rows_per_page)  # Calculate total pages with ceiling division
        self.current_page = max(0, total_pages - 1)  # Ensure it doesn’t go below 0
        self.endResetModel()


class SortFilterProxyModel(QSortFilterProxyModel):
    def __init__(self, *args, **kwargs):
        super(SortFilterProxyModel, self).__init__(*args, **kwargs)

    def lessThan(self, left, right):
        left_data = self.sourceModel().data(left, Qt.ItemDataRole.DisplayRole)
        right_data = self.sourceModel().data(right, Qt.ItemDataRole.DisplayRole)

        if left_data is None or right_data is None:
            return False

        try:
            left_data = float(left_data)
            right_data = float(right_data)
            return left_data < right_data
        except ValueError:
            # If conversion fails, fallback to string comparison
            return str(left_data) < str(right_data)

################################# Custom Button Manager Class #################################

class ButtonManager(QPushButton):
    """Button manager class to handle all disabled and enabled button behavior
    when a qprocess or long working thread is printing progress statements to
    the console widget display."""
    def __init__(self, *buttons):
        super(ButtonManager, self).__init__()
        self.buttons = buttons


    def disable_buttons(self, exclude=None):
        """Disable all buttons that can initiate separate threads 
        or processes while a thread or process is running, unless 
        they're in the 'exclude' list."""
        if exclude is None:
            exclude = []

        for btn in self.buttons:
            if btn not in exclude:
                btn.setDisabled(True)


    def enable_buttons(self, exclude=None):
        """Enable all buttons that can start new threads or processes
        and reset stylesheets back to default colors, unless 
        they're in the 'exclude' list."""
        if exclude is None:
            exclude = []

        for btn in self.buttons:
            if btn not in exclude:
                btn.setDisabled(False)


    # TODO: evenutally remove if no longer necessary. Keep for now until project is complete.           
    def disable_some_btns(self, include=[]):
        """Disable some buttons based on specific conditions"""

        for btn in self.buttons:
            if btn in include:
                btn.setDisabled(True)
 
   
    def enable_some_btns(self, include=[]):
        """Enable some buttons based on specific conditions"""

        for btn in self.buttons:
            if btn in include:
                btn.setDisabled(False)


############################ Confirm, Error, & Notice Message Boxes ############################

class ConfirmDialog(QDialog):
    """Custom confirm dialog class. 
    Construct the confirm message box with a custom title, message and buttons."""
    def __init__(self, title, message, button1='Ok', button2='Cancel', parent=None):
        super().__init__(parent)
        ################## Construct a Confirm Message Box ##################
        # Set the custom tile and message for each confirm dialog instance
        self.title = title
        self.message = message
        # Update the window title
        self.setWindowTitle(self.title)
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxQuestion)
        self.setWindowIcon(icon)  
        self.setStyleSheet("color: rgb(170, 255, 0);\n"
        "background-color: rgb(110, 110, 110);")

        # Construct custom Ok and Cancel buttons
        self.okButton = QPushButton(button1)
        ok_icon = QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
        self.okButton.setIcon(QIcon(ok_icon.pixmap(24, 24)))
        self.okButton.setStyleSheet("""
            QPushButton {
                color: white; 
                font-weight: bold;
                font-size: 11pt;
                background-color: rgb(80, 80, 80); 
                border: 2px outset rgb(50, 50, 50); 
                border-radius: 5px; 
                padding: 2px;
            }
            QPushButton:hover {
                background-color: rgb(90, 90, 90); 
                border: 2px outset rgb(60, 60, 60); 
            }
            QPushButton:pressed {
                color: rgb(170, 255, 0); 
                background-color: rgb(90, 90, 90); 
                border: 2px inset rgb(60, 60, 60); 
        }""")
        self.okButton.setMinimumSize(QSize(100, 25))

        self.cancelButton = QPushButton(button2)
        can_icon = QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DialogCancelButton)
        self.cancelButton.setIcon(QIcon(can_icon.pixmap(24, 24)))
        self.cancelButton.setStyleSheet("""
            QPushButton {
                color: white; 
                font-weight: bold;
                font-size: 11pt;
                background-color: rgb(80, 80, 80); 
                border: 2px outset rgb(50, 50, 50); 
                border-radius: 5px; 
                padding: 2px;
            }
            QPushButton:hover {
                background-color: rgb(90, 90, 90); 
                border: 2px outset rgb(60, 60, 60); 
            }
            QPushButton:pressed {
                color: rgb(255, 0, 0); 
                background-color: rgb(90, 90, 90); 
                border: 2px inset rgb(60, 60, 60); 
        }""")
        self.cancelButton.setMinimumSize(QSize(100, 25))

        self.buttonBox = QDialogButtonBox()
        # Add the custom buttons to the button box
        self.buttonBox.addButton(self.okButton, QDialogButtonBox.ButtonRole.AcceptRole)
        self.buttonBox.addButton(self.cancelButton, QDialogButtonBox.ButtonRole.RejectRole)
        self.buttonBox.setCenterButtons(True)
        self.buttonBox.setMinimumSize(250, 30)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.image = QLabel()
        pixmap = icon.pixmap(300, 300)
        self.image.setPixmap(pixmap.scaled(32, 32, Qt.AspectRatioMode.KeepAspectRatio))
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignRight)
        # Update the confirm message
        self.label = QLabel(self.message)
        self.label.setStyleSheet("color: rgb(170, 255, 0); font-size: 11pt; font-weight: bold;")
        self.label.setObjectName(u"message")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.layout = QGridLayout()
        self.layout.addWidget(self.image,0,0,2,1)
        self.layout.addWidget(self.label,0,1)
        self.layout.addWidget(self.buttonBox,1,1)
        self.setLayout(self.layout)


class ErrorMessageBox(QMessageBox):
    """Custom error message box class. 
    The critical method can take a custom title and message in the arguments when called."""
    def __init__(self, parent=None):
        super(ErrorMessageBox, self).__init__(parent)
        self.setStyleSheet("color: rgb(255, 80, 80);\n"
        "background-color: rgb(40, 40, 40); font-size: 11pt; font-weight: bold;")
        self.buttonBox = self.setStandardButtons(QMessageBox.StandardButton.Ok)

        okButton = self.button(QMessageBox.StandardButton.Ok)
        if okButton: 
            ok_icon = QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
            okButton.setIcon(QIcon(ok_icon.pixmap(24, 24)))
            okButton.setStyleSheet("""
            QPushButton {
                color: white; 
                font-weight: bold;
                font-size: 11pt;
                background-color: rgb(80, 80, 80); 
                border: 2px outset rgb(50, 50, 50); 
                border-radius: 5px; 
                padding: 2px;
            }
            QPushButton:hover {
                background-color: rgb(90, 90, 90); 
                border: 2px outset rgb(60, 60, 60); 
            }
            QPushButton:pressed {
                color: rgb(170, 255, 0); 
                background-color: rgb(90, 90, 90); 
                border: 2px inset rgb(60, 60, 60); 
            }""")


    @classmethod
    def critical(cls, title, message):
        """Set a custom title and message and display a critical message using a QMessageBox"""
        box = cls()
        box.setIcon(QMessageBox.Icon.Critical)
        icon = QApplication.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxCritical)
        box.setWindowIcon(QIcon(icon.pixmap(16, 16)))
        box.setWindowTitle(title)
        box.setText(message)
        box.exec()


class NoticeMessageBox(QMessageBox):
    """Custom Notice message box class.
    The notice method can take a custom title and message in the arguments when called."""
    def __init__(self, parent=None):
        super(NoticeMessageBox, self).__init__(parent)
        self.setStyleSheet("color: rgb(255, 170, 0);\n"
        "background-color: rgb(110, 110, 110); font-size: 11pt; font-weight: bold;")
        self.buttonBox = self.setStandardButtons(QMessageBox.StandardButton.Ok)

        okButton = self.button(QMessageBox.StandardButton.Ok)
        if okButton: 
            ok_icon = QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
            okButton.setIcon(QIcon(ok_icon.pixmap(24, 24)))
            okButton.setStyleSheet("""
            QPushButton {
                color: white; 
                font-weight: bold;
                font-size: 11pt;
                background-color: rgb(80, 80, 80); 
                border: 2px outset rgb(50, 50, 50); 
                border-radius: 5px; 
                padding: 2px;
            }
            QPushButton:hover {
                background-color: rgb(90, 90, 90); 
                border: 2px outset rgb(60, 60, 60); 
            }
            QPushButton:pressed {
                color: rgb(170, 255, 0); 
                background-color: rgb(90, 90, 90); 
                border: 2px inset rgb(60, 60, 60); 
            }""")


    @classmethod
    def notice(cls, title, message):
        """Set a custom title and message and display a notice message using a QMessageBox"""
        box = cls()
        box.setIcon(QMessageBox.Icon.Information)
        icon = QApplication.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation)
        box.setWindowIcon(QIcon(icon.pixmap(16, 16)))
        box.setWindowTitle(title)
        box.setText(message)
        box.exec()

################################### Excepttion Handler Class ###################################

class ExceptionLogger(QObject):
    """Exception logger class that will handle uncaught exceptions by redirecting
    the sys.exceptionhook to the exception_hoot method that will log exceptions and 
    display them on the console widget display. Log_qprocess_exception will handle 
    stderr message output and log those as well. A static method is used in the worker 
    class to log and display uncaught exceptions that occur in a worker thread."""

    ANSI_ESCAPE = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')

    def __init__(self):
        super(ExceptionLogger, self).__init__()
        # Error message box triggered on qprocess & uncaught exceptions
        self.errorMessage = ErrorMessageBox()
        # Redirect exception hook to capture all uncaught exceptions application wide
        sys.excepthook = self.exception_hook


    def exception_hook(self, exc_type, exc_value, exc_traceback):
        """Method handling uncaught exceptions application wide.
        It is triggered each time an uncaught exception occurs. 
        """
        if issubclass(exc_type, KeyboardInterrupt):
            # ignore keyboard interrupt to support console applications
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
        else:
            # Format the traceback
            traceback_details = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            seperator = '-'*97 +'\n'
            # Log the formatted traceback and save to error log file
            with open(ERROR_PATH, 'a') as file:
                file.write(f"Timestamp: {datetime.datetime.now()}\n")
                file.write(f"Uncaught Exception:\n{traceback_details}\n{seperator}")
                # Display on the console widget
            print(f"{Fore.RED}{seperator}Uncaught Exception:{Fore.RESET}\n{traceback_details}\n{Fore.RED}{seperator}{Fore.RESET}")
            # Trigger message box show
            self.errorMessage.critical("Uncaught Exception!", "Uncaught Exception! See console window for details.")


    def log_qprocess_exception(self, message):
        """Method to log uncaught exceptions that occur in a child process."""
        # Remove escape characters in the decoded traceback message
        message = self.ANSI_ESCAPE.sub('', message)
        seperator = '-'*97 +'\n'
        # Save to error log file
        with open(ERROR_PATH, 'a') as file:
            file.write(f"Timestamp: {datetime.datetime.now()}\n")
            file.write(f"QProcess Exception:\n{message}{seperator}")
        # Display on the console widget
        print(f"{Fore.RED}{seperator}QProcess Exception:{Fore.RESET}\n{message}\n{Fore.RED}{seperator}{Fore.RESET}")
        # Trigger the error message box
        self.errorMessage.critical("QProcess Exception!", "QProcess Exception! See console window for details.")


    @staticmethod
    def log_exception(exc_info):
        """Static method to log uncaught exceptions that occur in the threadpool."""
        # Format the traceback
        traceback_details = ''.join(traceback.format_exception(*exc_info))
        seperator = '-'*97 +'\n'
        # Log the formatted traceback and save to error log file
        with open(ERROR_PATH, 'a') as file:
            file.write(f"Timestamp: {datetime.datetime.now()}\n")
            file.write(f"Threadpool Exception:\n{traceback_details}\n{seperator}")
        # Display on the console widget
        print(f"{Fore.RED}{seperator}Threadpool Exception:{Fore.RESET}\n{traceback_details}\n{Fore.RED}{seperator}{Fore.RESET}")
