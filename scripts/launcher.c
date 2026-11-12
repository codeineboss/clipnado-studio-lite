#define UNICODE
#define _UNICODE
#include <windows.h>
#include <wchar.h>

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE previous, PWSTR command, int show) {
    (void)instance; (void)previous; (void)command; (void)show;
    wchar_t root[32768], python[32768], args[32768];
    DWORD length = GetModuleFileNameW(NULL, root, 32768);
    if (!length || length >= 32000) return 1;
    wchar_t *slash = wcsrchr(root, L'\\');
    if (!slash) return 1;
    *slash = 0;
    _snwprintf(python, 32768, L"%ls\\runtime\\pythonw.exe", root);
    _snwprintf(args, 32768, L"\"%ls\" \"%ls\\run.py\"", python, root);
    STARTUPINFOW startup = {0};
    PROCESS_INFORMATION process = {0};
    startup.cb = sizeof(startup);
    if (!CreateProcessW(python, args, NULL, NULL, FALSE, 0, NULL, root, &startup, &process)) {
        MessageBoxW(NULL, L"Не удалось запустить Python. Полностью распакуйте архив и запустите diagnostics.cmd.", L"Clipnado Studio Lite", MB_ICONERROR);
        return 1;
    }
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return 0;
}
