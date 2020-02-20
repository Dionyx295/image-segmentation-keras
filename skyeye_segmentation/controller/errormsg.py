from PyQt5.QtCore import (QtDebugMsg, QtInfoMsg, QtWarningMsg, QtCriticalMsg,
                          QtFatalMsg, QtSystemMsg)
from PyQt5.QtWidgets import QMessageBox


#############################################################################
def errormsg(typerr, contexte, msgerr):
    """Permet d'afficher et/ou de neutraliser les messages d'erreur.
       Pour les messages critiques: affiche dans une fenêtre graphique.
       Mise en place par: QtCore.qInstallMessageHandler(messagederreur)
    """
    if typerr == QtDebugMsg:
        # exemple de désactivation d'un message
        # if "QWindowsFileSystemWatcherEngine: unknown message" in msgerr:
        #    return
        print("DEBUG:\n{}\n".format(msgerr))

    elif typerr == QtWarningMsg:
        # exemple de désactivation d'un message
        # if "QFont::setPixelSize: Pixel size <= 0 (0)" in msgerr:
        #    return
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
        # message retourné par QtCore.QtInfoMsg créé à partir de Qt 5.5
        print("INFO:\n{}\n".format(msgerr))