"""Error handler, displays the associated dialog box."""

from PyQt5.QtCore import (QtDebugMsg, QtWarningMsg, QtCriticalMsg,
                          QtFatalMsg, QtSystemMsg)
from PyQt5.QtWidgets import QMessageBox


#############################################################################
def errormsg(typerr, msgerr):
    """
       Handles and shows error messages.
       Set up by: QtCore.qInstallMessageHandler(errormessage)
    """
    if typerr == QtDebugMsg:
        print("DEBUG:\n{}\n".format(msgerr))

    elif typerr == QtWarningMsg:
        print("WARNING:\n{}\n".format(msgerr))
        QMessageBox.warning(None,
                            "Attention !",
                            "{}\n".format(msgerr))

    elif typerr in [QtCriticalMsg, QtFatalMsg, QtSystemMsg]:
        print("ERREUR CRITIQUE:\n{}\n".format(msgerr))
        QMessageBox.critical(None,
                             "ERREUR CRITIQUE:",
                             "{}\n".format(msgerr))
    else:
        print("INFO:\n{}\n".format(msgerr))
