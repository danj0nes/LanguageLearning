#Requires AutoHotkey v2.0

global lastActiveWin := 0

; Track last active non-GUI window
SetTimer TrackActiveWindow, 200

TrackActiveWindow() {
    global lastActiveWin
    win := WinGetID("A")
    if win && WinGetTitle(win) != " " {
        lastActiveWin := win
    }
}

; NORMAL window (has title bar now)
myGui := Gui("+AlwaysOnTop +ToolWindow", " ")
myGui.SetFont("s12")

chars := [
    ["é", "è", "ê", "ë"],
    ["à", "â"],
    ["î", "ï"],
    ["ô"],
    ["ù", "û", "ü"],
    ["ç"],
    ["œ", "æ"]
]

for row in chars {
    for c in row {
        btn := myGui.Add("Button", "w40 h30", c)
        btn.OnEvent("Click", MakeHandler(c))
    }
    myGui.Add("Text", "xm")
}

; Close behavior
myGui.OnEvent("Close", (*) => ExitApp())

myGui.Show()

MakeHandler(char) {
    return (*) => InsertChar(char)
}

InsertChar(char) {
    global lastActiveWin

    if lastActiveWin {
        WinActivate(lastActiveWin)
        Sleep 40
    }

    SendText(char)
}
