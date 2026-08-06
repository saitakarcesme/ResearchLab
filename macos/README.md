# ResearchLab macOS client

This native client is a lightweight `WKWebView` shell for the private Tailscale URL. It does
not start a backend, database, Ollama process, or research worker on the Mac.

Build the executable on Apple silicon with:

```sh
swiftc -O -framework AppKit -framework WebKit \
  -o ResearchLab.app/Contents/MacOS/ResearchLab ResearchLabApp/main.swift
```

Copy `ResearchLabApp/Info.plist` into `ResearchLab.app/Contents/Info.plist`, then ad-hoc sign
the bundle with `codesign --force --deep --sign - ResearchLab.app`.
