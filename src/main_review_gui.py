import tkinter as tk
from src.gui.video_review import VideoReviewApp


def main():
    root = tk.Tk()
    app = VideoReviewApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
