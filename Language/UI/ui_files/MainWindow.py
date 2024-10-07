# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MainWindow.ui'
##
## Created by: Qt User Interface Compiler version 6.7.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame,
    QGridLayout, QGroupBox, QLabel, QMainWindow,
    QMenu, QMenuBar, QPushButton, QSizePolicy,
    QStatusBar, QTextBrowser, QTextEdit, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(750, 850)
        MainWindow.setMinimumSize(QSize(750, 850))
        font = QFont()
        font.setPointSize(10)
        MainWindow.setFont(font)
        icon = QIcon()
        icon.addFile(u"./Language/UI/ui_files/icons8-chat-bot.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setStyleSheet(u"background-color: rgb(60, 60, 60);")
        self.actionLoad = QAction(MainWindow)
        self.actionLoad.setObjectName(u"actionLoad")
        self.actionLoad.setFont(font)
        self.actionQuit = QAction(MainWindow)
        self.actionQuit.setObjectName(u"actionQuit")
        self.actionQuit.setFont(font)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout_6 = QGridLayout(self.centralwidget)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.buttonFrame = QFrame(self.centralwidget)
        self.buttonFrame.setObjectName(u"buttonFrame")
        self.buttonFrame.setStyleSheet(u"QFrame {\n"
"    background-color: rgb(100, 100, 95);\n"
"    border-radius: 5px;\n"
"}")
        self.buttonFrame.setFrameShape(QFrame.Shape.WinPanel)
        self.buttonFrame.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout_2 = QGridLayout(self.buttonFrame)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.modelBox = QComboBox(self.buttonFrame)
        self.modelBox.setObjectName(u"modelBox")
        font1 = QFont()
        font1.setPointSize(11)
        font1.setBold(True)
        self.modelBox.setFont(font1)
        self.modelBox.setStyleSheet(u"QComboBox {\n"
"    border-radius: 5px;\n"
"    padding: 2px 10px;\n"
"    min-width: 6em;\n"
"	background-color: rgb(70, 70, 70);\n"
"	border: 2px outset rgb(50, 50, 50); \n"
"}\n"
"QComboBox:hover {\n"
"	background-color: rgb(80, 80, 80); \n"
"	border: 2px outset rgb(60, 60, 60); \n"
"}\n"
"QComboBox::drop-down:on {\n"
"    subcontrol-origin: padding;\n"
"    subcontrol-position: top right;\n"
"    width: 15px;\n"
"	border-left-width: 1px;\n"
"    border-left-color: rgb(50, 50, 50);\n"
"    border-left-style: solid; \n"
"    border-top-right-radius: 3px; \n"
"    border-bottom-right-radius: 3px;\n"
"}\n"
"QComboBox QAbstractItemView {\n"
"    border-radius: 5px;\n"
"    selection-background-color: rgb(80, 80, 80);\n"
"    background-color: rgb(70, 70, 70); \n"
"}")

        self.gridLayout_2.addWidget(self.modelBox, 0, 0, 1, 2)

        self.loadButton = QPushButton(self.buttonFrame)
        self.loadButton.setObjectName(u"loadButton")
        self.loadButton.setFont(font1)
        self.loadButton.setStyleSheet(u"QPushButton {\n"
"	color: white; \n"
"	font-weight: bold;\n"
"	font-size: 11pt;\n"
"	background-color: rgb(70, 70, 70); \n"
"	border: 2px outset rgb(50, 50, 50); \n"
"	border-radius: 5px; \n"
"	padding: 2px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(80, 80, 80); \n"
"	border: 2px outset rgb(60, 60, 60); \n"
"}\n"
"QPushButton:pressed {\n"
"	color: rgb(170, 255, 0); \n"
"	background-color: rgb(80, 80, 80); \n"
"	border: 2px inset rgb(60, 60, 60); \n"
"}")
        icon1 = QIcon()
        icon1.addFile(u"./Language/UI/ui_files/icons8-enter.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.loadButton.setIcon(icon1)

        self.gridLayout_2.addWidget(self.loadButton, 0, 2, 1, 2)

        self.genGroup = QGroupBox(self.buttonFrame)
        self.genGroup.setObjectName(u"genGroup")
        self.genGroup.setMaximumSize(QSize(16777215, 16777215))
        self.genGroup.setFont(font1)
        self.genGroup.setStyleSheet(u"QGroupBox {\n"
"	background-color: rgb(100,100,95);	\n"
"	border: 2px solid rgb(200,200,200); \n"
"	border-radius: 5px; \n"
"	margin-top: 3ex; \n"
"	padding: 2px;\n"
"}\n"
"QGroupBox::title {\n"
"	subcontrol-origin: margin;\n"
"	subcontrol-position: top center; \n"
"	padding: 0 1px; \n"
"	color: rgb(170, 255, 0);\n"
"}")
        self.gridLayout_5 = QGridLayout(self.genGroup)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.label1 = QLabel(self.genGroup)
        self.label1.setObjectName(u"label1")
        self.label1.setMinimumSize(QSize(75, 25))
        self.label1.setFont(font1)
        self.label1.setStyleSheet(u"color: rgb(255, 170, 0);")
        self.label1.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_5.addWidget(self.label1, 0, 0, 1, 1)

        self.timeLabel = QLabel(self.genGroup)
        self.timeLabel.setObjectName(u"timeLabel")
        self.timeLabel.setMinimumSize(QSize(75, 25))
        font2 = QFont()
        font2.setPointSize(11)
        font2.setBold(False)
        self.timeLabel.setFont(font2)
        self.timeLabel.setStyleSheet(u"color: rgb(0, 255, 255);")
        self.timeLabel.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_5.addWidget(self.timeLabel, 0, 1, 1, 1)

        self.label2 = QLabel(self.genGroup)
        self.label2.setObjectName(u"label2")
        self.label2.setMinimumSize(QSize(75, 25))
        self.label2.setFont(font1)
        self.label2.setStyleSheet(u"color: rgb(255, 170, 0);")
        self.label2.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_5.addWidget(self.label2, 1, 0, 1, 1)

        self.wordsLabel = QLabel(self.genGroup)
        self.wordsLabel.setObjectName(u"wordsLabel")
        self.wordsLabel.setMinimumSize(QSize(75, 25))
        self.wordsLabel.setFont(font2)
        self.wordsLabel.setStyleSheet(u"color: rgb(0, 255, 255);")
        self.wordsLabel.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_5.addWidget(self.wordsLabel, 1, 1, 1, 1)

        self.label3 = QLabel(self.genGroup)
        self.label3.setObjectName(u"label3")
        self.label3.setMinimumSize(QSize(75, 25))
        self.label3.setFont(font1)
        self.label3.setStyleSheet(u"color: rgb(255, 170, 0);")
        self.label3.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_5.addWidget(self.label3, 2, 0, 1, 1)

        self.scoreLabel = QLabel(self.genGroup)
        self.scoreLabel.setObjectName(u"scoreLabel")
        self.scoreLabel.setMinimumSize(QSize(75, 25))
        self.scoreLabel.setFont(font2)
        self.scoreLabel.setStyleSheet(u"color: rgb(0, 255, 255);")
        self.scoreLabel.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_5.addWidget(self.scoreLabel, 2, 1, 1, 1)


        self.gridLayout_2.addWidget(self.genGroup, 0, 4, 3, 1)

        self.tempBox = QComboBox(self.buttonFrame)
        self.tempBox.setObjectName(u"tempBox")
        font3 = QFont()
        font3.setPointSize(10)
        font3.setBold(True)
        self.tempBox.setFont(font3)
        self.tempBox.setStyleSheet(u"QComboBox {\n"
"    border-radius: 5px;\n"
"    padding: 2px 10px;\n"
"    min-width: 6em;\n"
"	background-color: rgb(70, 70, 70);\n"
"	border: 2px outset rgb(50, 50, 50); \n"
"}\n"
"QComboBox:hover {\n"
"	background-color: rgb(80, 80, 80); \n"
"	border: 2px outset rgb(60, 60, 60); \n"
"}\n"
"QComboBox::drop-down:on {\n"
"    subcontrol-origin: padding;\n"
"    subcontrol-position: top right;\n"
"    width: 15px;\n"
"	border-left-width: 1px;\n"
"    border-left-color: rgb(50, 50, 50);\n"
"    border-left-style: solid; \n"
"    border-top-right-radius: 3px; \n"
"    border-bottom-right-radius: 3px;\n"
"}\n"
"QComboBox QAbstractItemView {\n"
"    border-radius: 5px;\n"
"    selection-background-color: rgb(80, 80, 80);\n"
"    background-color: rgb(70, 70, 70); \n"
"}")

        self.gridLayout_2.addWidget(self.tempBox, 1, 0, 1, 1)

        self.topkBox = QComboBox(self.buttonFrame)
        self.topkBox.setObjectName(u"topkBox")
        self.topkBox.setFont(font3)
        self.topkBox.setStyleSheet(u"QComboBox {\n"
"    border-radius: 5px;\n"
"    padding: 2px 10px;\n"
"    min-width: 6em;\n"
"	background-color: rgb(70, 70, 70);\n"
"	border: 2px outset rgb(50, 50, 50); \n"
"}\n"
"QComboBox:hover {\n"
"	background-color: rgb(80, 80, 80); \n"
"	border: 2px outset rgb(60, 60, 60); \n"
"}\n"
"QComboBox::drop-down:on {\n"
"    subcontrol-origin: padding;\n"
"    subcontrol-position: top right;\n"
"    width: 15px;\n"
"	border-left-width: 1px;\n"
"    border-left-color: rgb(50, 50, 50);\n"
"    border-left-style: solid; \n"
"    border-top-right-radius: 3px; \n"
"    border-bottom-right-radius: 3px;\n"
"}\n"
"QComboBox QAbstractItemView {\n"
"    border-radius: 5px;\n"
"    selection-background-color: rgb(80, 80, 80);\n"
"    background-color: rgb(70, 70, 70); \n"
"}")

        self.gridLayout_2.addWidget(self.topkBox, 1, 1, 1, 1)

        self.toppBox = QComboBox(self.buttonFrame)
        self.toppBox.setObjectName(u"toppBox")
        self.toppBox.setFont(font3)
        self.toppBox.setStyleSheet(u"QComboBox {\n"
"    border-radius: 5px;\n"
"    padding: 2px 10px;\n"
"    min-width: 6em;\n"
"	background-color: rgb(70, 70, 70);\n"
"	border: 2px outset rgb(50, 50, 50); \n"
"}\n"
"QComboBox:hover {\n"
"	background-color: rgb(80, 80, 80); \n"
"	border: 2px outset rgb(60, 60, 60); \n"
"}\n"
"QComboBox::drop-down:on {\n"
"    subcontrol-origin: padding;\n"
"    subcontrol-position: top right;\n"
"    width: 15px;\n"
"	border-left-width: 1px;\n"
"    border-left-color: rgb(50, 50, 50);\n"
"    border-left-style: solid; \n"
"    border-top-right-radius: 3px; \n"
"    border-bottom-right-radius: 3px;\n"
"}\n"
"QComboBox QAbstractItemView {\n"
"    border-radius: 5px;\n"
"    selection-background-color: rgb(80, 80, 80);\n"
"    background-color: rgb(70, 70, 70); \n"
"}")

        self.gridLayout_2.addWidget(self.toppBox, 1, 2, 1, 1)

        self.tagCheck = QCheckBox(self.buttonFrame)
        self.tagCheck.setObjectName(u"tagCheck")
        self.tagCheck.setFont(font1)
        self.tagCheck.setStyleSheet(u"color: rgb(170, 255, 0);\n"
"background-color: rgb(100, 100, 95);")

        self.gridLayout_2.addWidget(self.tagCheck, 1, 3, 1, 1)

        self.inputGroup = QGroupBox(self.buttonFrame)
        self.inputGroup.setObjectName(u"inputGroup")
        self.inputGroup.setMaximumSize(QSize(16777215, 16777215))
        self.inputGroup.setFont(font1)
        self.inputGroup.setStyleSheet(u"QGroupBox {\n"
"	background-color: rgb(100,100,95);\n"
"	border: 2px solid rgb(200,200,200); \n"
"	border-radius: 5px; \n"
"	margin-top: 3ex; \n"
"	padding: 2px;\n"
"}\n"
"QGroupBox::title {\n"
"	subcontrol-origin: margin;\n"
"	subcontrol-position: top center; \n"
"	padding: 0 1px; \n"
"	color: rgb(170, 255, 0);\n"
"}")
        self.gridLayout_3 = QGridLayout(self.inputGroup)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.promptLabel = QLabel(self.inputGroup)
        self.promptLabel.setObjectName(u"promptLabel")
        self.promptLabel.setMinimumSize(QSize(0, 25))
        self.promptLabel.setMaximumSize(QSize(16777215, 16777215))
        self.promptLabel.setFont(font3)
        self.promptLabel.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.promptLabel.setFrameShadow(QFrame.Shadow.Plain)
        self.promptLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout_3.addWidget(self.promptLabel, 0, 0, 1, 1)


        self.gridLayout_2.addWidget(self.inputGroup, 2, 0, 1, 2)

        self.outputGroup = QGroupBox(self.buttonFrame)
        self.outputGroup.setObjectName(u"outputGroup")
        self.outputGroup.setMaximumSize(QSize(16777215, 16777215))
        self.outputGroup.setFont(font1)
        self.outputGroup.setStyleSheet(u"QGroupBox {\n"
"	background-color: rgb(100,100,95);\n"
"	border: 2px solid rgb(200,200,200); \n"
"	border-radius: 5px; \n"
"	margin-top: 3ex; \n"
"	padding: 2px;\n"
"}\n"
"QGroupBox::title {\n"
"	subcontrol-origin: margin;\n"
"	subcontrol-position: top center; \n"
"	padding: 0 1px; \n"
"	color: rgb(170, 255, 0);\n"
"}")
        self.gridLayout_4 = QGridLayout(self.outputGroup)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.respLabel = QLabel(self.outputGroup)
        self.respLabel.setObjectName(u"respLabel")
        self.respLabel.setMinimumSize(QSize(0, 25))
        self.respLabel.setFont(font3)
        self.respLabel.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.respLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout_4.addWidget(self.respLabel, 0, 0, 1, 1)


        self.gridLayout_2.addWidget(self.outputGroup, 2, 2, 1, 2)


        self.gridLayout_6.addWidget(self.buttonFrame, 0, 0, 1, 1)

        self.consoleFrame = QFrame(self.centralwidget)
        self.consoleFrame.setObjectName(u"consoleFrame")
        self.consoleFrame.setStyleSheet(u"QFrame {\n"
"    background-color: rgb(100, 100, 95);\n"
"    border-radius: 5px;\n"
"}")
        self.consoleFrame.setFrameShape(QFrame.Shape.WinPanel)
        self.consoleFrame.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout = QGridLayout(self.consoleFrame)
        self.gridLayout.setObjectName(u"gridLayout")
        self.console = QTextBrowser(self.consoleFrame)
        self.console.setObjectName(u"console")
        font4 = QFont()
        font4.setPointSize(12)
        self.console.setFont(font4)
        self.console.setStyleSheet(u"QTextBrowser {\n"
"	background-color: rgb(40, 40, 40); \n"
"	color: rgb(255, 255, 255); \n"
"	border: 2px inset rgb(80, 80, 80);\n"
"	border-radius: 5px;  \n"
"	padding: 2px;\n"
"}")

        self.gridLayout.addWidget(self.console, 0, 0, 1, 2)

        self.textBox = QTextEdit(self.consoleFrame)
        self.textBox.setObjectName(u"textBox")
        self.textBox.setMaximumSize(QSize(16777215, 50))
        font5 = QFont()
        font5.setPointSize(11)
        self.textBox.setFont(font5)
        self.textBox.setStyleSheet(u"QTextEdit {\n"
"	background-color: rgb(40, 40, 40); \n"
"	color: rgb(255, 255, 255); \n"
"	border: 2px inset rgb(80, 80, 80);\n"
"	border-radius: 5px;  \n"
"	padding: 2px;\n"
"}\n"
"QTextEdit:hover {\n"
"	border: 2px inset rgb(170, 255, 0); \n"
"}")

        self.gridLayout.addWidget(self.textBox, 1, 0, 1, 1)

        self.clearButton = QPushButton(self.consoleFrame)
        self.clearButton.setObjectName(u"clearButton")
        self.clearButton.setMinimumSize(QSize(125, 30))
        self.clearButton.setFont(font1)
        self.clearButton.setStyleSheet(u"QPushButton {\n"
"	color: white; \n"
"	font-weight: bold;\n"
"	font-size: 11pt;\n"
"	background-color: rgb(70, 70, 70); \n"
"	border: 2px outset rgb(50, 50, 50); \n"
"	border-radius: 5px; \n"
"	padding: 2px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(80, 80, 80); \n"
"	border: 2px outset rgb(60, 60, 60); \n"
"}\n"
"QPushButton:pressed {\n"
"	color: rgb(255, 0, 0); \n"
"	background-color: rgb(80, 80, 80); \n"
"	border: 2px inset rgb(60, 60, 60); \n"
"}")
        icon2 = QIcon()
        icon2.addFile(u"./Language/UI/ui_files/icons8-clear.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.clearButton.setIcon(icon2)

        self.gridLayout.addWidget(self.clearButton, 1, 1, 1, 1)


        self.gridLayout_6.addWidget(self.consoleFrame, 1, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 750, 23))
        self.menubar.setStyleSheet(u"QMenuBar {\n"
"	color: white;\n"
"	font-size: 11pt;\n"
"	font-weight: bold;\n"
"	background-color: rgb(60, 60, 60);\n"
"	spacing: 4px; /* spacing between menu bar items */\n"
"}\n"
"QMenuBar::item {\n"
"	padding: 1px 4px;\n"
"	background: transparent;\n"
"	border-radius: 2px;\n"
"}\n"
"QMenuBar::item:selected { /* when selected using mouse or keyboard */\n"
"	background: rgb(100, 100, 95);\n"
"}")
        self.menuMenu = QMenu(self.menubar)
        self.menuMenu.setObjectName(u"menuMenu")
        self.menuMenu.setStyleSheet(u"QMenu {\n"
"	color: white;\n"
"    background-color: rgb(100, 100, 95);\n"
"    margin: 2px; /* some spacing around the menu */\n"
"}\n"
"QMenu::item {\n"
"    padding: 2px 25px 2px 20px;\n"
"    border: 1px solid transparent; /* reserve space for selection border */\n"
"}\n"
"QMenu::item:selected {\n"
"    border-color: rgb(170, 255, 0);\n"
"    background: rgba(170, 255, 0, 50);\n"
"}")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuMenu.menuAction())
        self.menuMenu.addAction(self.actionLoad)
        self.menuMenu.addAction(self.actionQuit)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Chat Bot", None))
        self.actionLoad.setText(QCoreApplication.translate("MainWindow", u"Load Model", None))
        self.actionQuit.setText(QCoreApplication.translate("MainWindow", u"Quit", None))
#if QT_CONFIG(tooltip)
        self.modelBox.setToolTip(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" color:#aaff00;\">Select the type of GPT model to use</span></p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.modelBox.setPlaceholderText(QCoreApplication.translate("MainWindow", u"GPT Ultra Chat", None))
        self.loadButton.setText(QCoreApplication.translate("MainWindow", u"Load Model", None))
        self.genGroup.setTitle(QCoreApplication.translate("MainWindow", u"Generation", None))
        self.label1.setText(QCoreApplication.translate("MainWindow", u"Time: ", None))
        self.timeLabel.setText("")
        self.label2.setText(QCoreApplication.translate("MainWindow", u"# Words: ", None))
        self.wordsLabel.setText("")
        self.label3.setText(QCoreApplication.translate("MainWindow", u"Similarity: ", None))
        self.scoreLabel.setText("")
#if QT_CONFIG(tooltip)
        self.tempBox.setToolTip(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" color:#aaff00;\">Apply temperature - lower percentage value reduces randomness</span></p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.tempBox.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Temp", None))
#if QT_CONFIG(tooltip)
        self.topkBox.setToolTip(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" color:#aaff00;\">Generate the next word based on a limited sample of top words</span></p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.topkBox.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Top K", None))
#if QT_CONFIG(tooltip)
        self.toppBox.setToolTip(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" color:#aaff00;\">Generate the next word from a culmulative sampling based on a percentage</span></p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.toppBox.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Top P", None))
#if QT_CONFIG(tooltip)
        self.tagCheck.setToolTip(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" color:#aaff00;\">This will color named entities by type in the generated response.</span></p><p><span style=\" color:#aaff00;\">Blue = People, Green = Location, Purple = Organization</span></p><p><span style=\" color:#aaff00;\">Red = Global Political Entity, Cyan = Time, Yellow = Misc.</span></p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.tagCheck.setText(QCoreApplication.translate("MainWindow", u"  Tag Entities", None))
        self.inputGroup.setTitle(QCoreApplication.translate("MainWindow", u"Prompt Sentiment", None))
        self.promptLabel.setText("")
        self.outputGroup.setTitle(QCoreApplication.translate("MainWindow", u"Response Sentiment", None))
        self.respLabel.setText("")
        self.textBox.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Ask the chatbot a question and hit 'enter'", None))
        self.clearButton.setText(QCoreApplication.translate("MainWindow", u" Clear", None))
        self.menuMenu.setTitle(QCoreApplication.translate("MainWindow", u"Menu", None))
    # retranslateUi

