import threading, tkinter as tk
from tkinter import ttk, messagebox
import speedtest


class SpeedTestApp:
    def __init__(self, root):
        self.root = root
        root.title("Internet Speed Checker")
        root.geometry("280x140")
        root.resizable(0, 0)

        main_frame = ttk.Frame(root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=1)

        self.default_button_text = "Test Internet Speed"
        self.test_button = ttk.Button(main_frame, text=self.default_button_text, command=self.start_test)
        self.test_button.pack(expand=1)

    def _set_loading(self, is_loading: bool):
        def apply_state():
            if is_loading:
                self.test_button.config(text="Testing…", state=tk.DISABLED)
                self.root.configure(cursor="watch")
            else:
                self.test_button.config(text=self.default_button_text, state=tk.NORMAL)
                self.root.configure(cursor="")

        self.root.after(0, apply_state)

    def start_test(self):
        self._set_loading(True)
        threading.Thread(target=self.run_test, daemon=True).start()

    def run_test(self):
        try:
            st = speedtest.Speedtest()
            st.get_best_server()
            dl = st.download()
            up = st.upload(pre_allocate=False)
            ping = st.results.ping

            def show_results():
                messagebox.showinfo(
                    "Speed Test Results",
                    f"Download: {dl / 1e6:.2f} Mbps\nUpload: {up / 1e6:.2f} Mbps\nPing: {ping:.1f} ms",
                )

            self.root.after(0, show_results)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"An error occurred:\n{e}"))
        finally:
            self._set_loading(False)

if __name__ == "__main__":
    root = tk.Tk()
    SpeedTestApp(root)
    root.mainloop()