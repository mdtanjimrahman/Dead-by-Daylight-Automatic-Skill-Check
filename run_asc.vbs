
Set WshShell = CreateObject("WScript.Shell")

WshShell.CurrentDirectory = "D:\dbd_autoSkillCheck"

WshShell.Run """D:\dbd_autoSkillCheck\.venv\Scripts\pythonw.exe"" ""D:\dbd_autoSkillCheck\great_patch.py""", 0, False

Set WshShell = Nothing
