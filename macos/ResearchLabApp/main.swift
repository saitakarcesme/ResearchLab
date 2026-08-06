import AppKit
import Foundation
import WebKit

private let labURL = URL(string: "https://ibrahim.tailae27bb.ts.net/lab")!
private let healthURL = URL(string: "https://ibrahim.tailae27bb.ts.net/health")!
private let allowedHost = "ibrahim.tailae27bb.ts.net"

final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate, WKNavigationDelegate {
    private var window: NSWindow!
    private var webView: WKWebView!

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.applicationIconImage = makeApplicationIcon()

        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        configuration.applicationNameForUserAgent = "ResearchLabMac/1.1"
        configuration.preferences.javaScriptCanOpenWindowsAutomatically = false
        webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = self
        webView.allowsMagnification = true

        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1440, height: 900),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.title = "ResearchLab"
        window.titlebarAppearsTransparent = true
        window.backgroundColor = .black
        window.contentView = webView
        window.delegate = self
        window.center()
        window.setFrameAutosaveName("ResearchLabMainWindow")
        window.makeKeyAndOrderFront(nil)

        checkHealthAndLoad()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    func applicationDidResignActive(_ notification: Notification) {
        setWebContentActive(false)
    }

    func applicationDidBecomeActive(_ notification: Notification) {
        setWebContentActive(!window.isMiniaturized)
    }

    func applicationWillTerminate(_ notification: Notification) {
        webView.stopLoading()
        webView.navigationDelegate = nil
    }

    func windowDidMiniaturize(_ notification: Notification) {
        setWebContentActive(false)
    }

    func windowDidDeminiaturize(_ notification: Notification) {
        setWebContentActive(true)
    }

    private func setWebContentActive(_ active: Bool) {
        webView.evaluateJavaScript(
            "window.dispatchEvent(new CustomEvent('researchlab-visibility',{detail:\(active ? "true" : "false")}))"
        )
    }

    private func checkHealthAndLoad() {
        showStatus("Windows ResearchLab bağlantısı kontrol ediliyor…")

        var request = URLRequest(url: healthURL)
        request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        request.timeoutInterval = 12

        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            guard let self else { return }
            let httpStatus = (response as? HTTPURLResponse)?.statusCode
            let healthy: Bool
            if let data,
               let value = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                healthy = httpStatus == 200
                    && value["status"] as? String == "ok"
                    && value["execution_enabled"] as? Bool == true
            } else {
                healthy = false
            }

            DispatchQueue.main.async {
                if healthy {
                    self.webView.load(URLRequest(url: labURL, cachePolicy: .reloadIgnoringLocalCacheData))
                } else {
                    self.showConnectionError(error?.localizedDescription)
                }
            }
        }.resume()
    }

    private func showStatus(_ message: String) {
        let html = pageHTML(
            title: "ResearchLab",
            message: message,
            action: nil
        )
        webView.loadHTMLString(html, baseURL: nil)
    }

    private func showConnectionError(_ detail: String?) {
        let suffix = detail.map { "<small>\($0.htmlEscaped)</small>" } ?? ""
        let html = pageHTML(
            title: "ResearchLab hazır değil",
            message: "Tailscale bağlantısını ve Windows ResearchLab servisini kontrol edin.\(suffix)",
            action: "<a href=\"researchlab://retry\">Yeniden dene</a>"
        )
        webView.loadHTMLString(html, baseURL: nil)
    }

    private func pageHTML(title: String, message: String, action: String?) -> String {
        """
        <!doctype html>
        <meta charset="utf-8">
        <meta name="color-scheme" content="dark">
        <style>
          * { box-sizing: border-box; }
          body { margin: 0; min-height: 100vh; display: grid; place-items: center;
                 background: #080808; color: #f3f3f3; font: 16px -apple-system, BlinkMacSystemFont, sans-serif; }
          main { width: min(520px, calc(100vw - 48px)); padding: 36px; border: 1px solid #333;
                 border-radius: 22px; background: #111; text-align: center; }
          h1 { margin: 0 0 14px; font-size: 28px; }
          p { margin: 0; color: #aaa; line-height: 1.55; }
          small { display: block; margin-top: 10px; color: #777; }
          a { display: inline-block; margin-top: 24px; padding: 11px 18px; border-radius: 999px;
              background: #eee; color: #111; text-decoration: none; font-weight: 650; }
        </style>
        <main><h1>\(title)</h1><p>\(message)</p>\(action ?? "")</main>
        """
    }

    func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationAction: WKNavigationAction,
        decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
    ) {
        guard let url = navigationAction.request.url else {
            decisionHandler(.cancel)
            return
        }

        if url.scheme == "researchlab" && url.host == "retry" {
            decisionHandler(.cancel)
            checkHealthAndLoad()
            return
        }

        if url.host == nil || url.host == allowedHost {
            decisionHandler(.allow)
            return
        }

        NSWorkspace.shared.open(url)
        decisionHandler(.cancel)
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        showConnectionError(error.localizedDescription)
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        showConnectionError(error.localizedDescription)
    }

    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        showStatus("Görünüm güvenli biçimde yeniden başlatılıyor…")
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { [weak self] in
            self?.checkHealthAndLoad()
        }
    }

    private func makeApplicationIcon() -> NSImage {
        let size = NSSize(width: 512, height: 512)
        let image = NSImage(size: size)
        image.lockFocus()

        let background = NSBezierPath(roundedRect: NSRect(origin: .zero, size: size), xRadius: 112, yRadius: 112)
        NSColor(calibratedWhite: 0.055, alpha: 1).setFill()
        background.fill()

        let ring = NSBezierPath(roundedRect: NSRect(x: 38, y: 38, width: 436, height: 436), xRadius: 88, yRadius: 88)
        ring.lineWidth = 10
        NSColor(calibratedWhite: 0.88, alpha: 1).setStroke()
        ring.stroke()

        let attributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 236, weight: .semibold),
            .foregroundColor: NSColor.white
        ]
        let label = NSAttributedString(string: "R", attributes: attributes)
        let labelSize = label.size()
        label.draw(at: NSPoint(x: (512 - labelSize.width) / 2, y: (512 - labelSize.height) / 2 - 17))

        image.unlockFocus()
        return image
    }
}

private extension String {
    var htmlEscaped: String {
        replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
            .replacingOccurrences(of: "\"", with: "&quot;")
    }
}

let application = NSApplication.shared
let delegate = AppDelegate()
application.delegate = delegate
application.setActivationPolicy(.regular)
application.activate(ignoringOtherApps: true)
application.run()
