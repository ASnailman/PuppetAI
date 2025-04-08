import tkinter as tk

class PointOverlay:
    # member vars

    # functions
    def __init__(self):
        self.root = tk.Tk() # initialize window
        self.root.attributes('-topmost', True) # tkinter window stays on top
        self.root.attributes('-transparentcolor', 'white') # turn into an overlay by setting window to transparent
        self.root.overrideredirect(True)  # Remove window decorations
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}+0+0") # set window size to screen size
        
        self.canvas = tk.Canvas(self.root, width=screen_width, height=screen_height, bg='white', highlightthickness=0) # canvas to draw on
        self.canvas.pack() # places canvas on root window

        self.dot = self.canvas.create_oval(0, 0, 15, 15, fill='red', outline='') # create red dot 
          
    def update_dot(self, x, y):
        r = 7.5  # radius
        self.canvas.coords(self.dot, x - r, y - r, x + r, y + r) # update position of dot
        
    def run(self):
        self.root.mainloop() # main loop of tkinter, listens for events, blocks execution on main thread