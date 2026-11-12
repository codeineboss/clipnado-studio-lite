if __name__ == "__main__":
    try:
        from clipnado.app import main
        main()
    except Exception:
        import os
        import traceback
        from pathlib import Path
        log = Path(os.environ.get("CLIPNADO_DATA_DIR", os.environ.get("LOCALAPPDATA", "."))) / "ClipnadoStudioLite" / "startup-error.log"
        try:
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(traceback.format_exc(), encoding="utf-8")
        except OSError:
            pass
        if os.name == "nt":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, f"Не удалось запустить Clipnado. Запустите diagnostics.cmd.\nДиагностика: {log}", "Clipnado", 16)
        else:
            raise
