from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as error:
        print(
            f"PySide6 is not installed. Run: pip install -e .\nDetails: {error}",
            file=sys.stderr,
        )
        return 1

    from .ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Deep Learning Studio")
    app.setOrganizationName("Fugu0141")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
