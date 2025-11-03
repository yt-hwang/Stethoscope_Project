import sys
from PyQt5 import QtWidgets
from UI.wearable_stetho_ui import WearableStethoUI

def main():
    app = QtWidgets.QApplication(sys.argv)
    ui = WearableStethoUI(); ui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
