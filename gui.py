###### gui.py ######
# ->

###### IMPORTS ######
from imports import *
###### END IMPORTS ######

px = py = 0

def gui_app():
    window = tk.Tk()
    tabControl = ttk.Notebook(window)

    for m in get_monitors():
        width = m.width
        height = m.height
        resolution = str(m.width) + 'x' + str(m.height)

    style=ttk.Style()
    style.theme_create('yummy',
                       settings={
                           "TNotebook": {
                               "configure": {
                                   "tabmargins": [10, 10, 10, 0],
                               },
                           },
                           "TNotebook.Tab": {
                                "configure": {
                                    "font": ('Aptos Extra Bold', 20),
                                    "background": '#0f0f0f',
                                    "foreground": 'white',
                                    "padding": 10,
                                },
                                "map": {
                                    "background": [
                                        ("selected", '#1f1f1f'),
                                        ('active', '#0f0f0f'),
                                        ],
                                    "foreground": [
                                        ('selected', 'white'),
                                        ('active', 'white')
                                        ],
                                },
                           },
                       }
                       )
    style.theme_use('yummy')

    window.geometry(resolution)
    window.state('zoomed')
    window.configure(bg='#101010')
    window.title("")

    frame_left = draw_frame_left(window, width, height)
    frame_left.pack(padx=0, pady=0, side=tk.LEFT)

    tab1 = draw_tab_1(tabControl, width, height)
    tab2 = tk.Frame(
        master=tabControl,
        width = width,
        height=height,
        bg='#1f1f1f',
    )
    tab3 = tk.Frame(
        master=tabControl,
        width = width,
        height=height,
        bg='#1f1f1f',
    )

    tabs = {
        "CT Scan Processing": tab1,
        "Nume tab 2": tab2,
        "Nume tab 3": tab3,
    }

    frame_top = draw_frame_top(tabs, tabControl, window, width, height)
    frame_top.pack(padx=0, pady=0, side=tk.TOP)

    window.mainloop()

def draw_frame_left(window, width, height):
    frame = tk.Frame(
        master=window,
        width=width//4,
        height=height,
        bg='#1f1f1f',
        highlightbackground='#2f2f2f',
        highlightthickness=2
    )
    frame.pack(expand=False, fill=BOTH)
    frame.pack_propagate(False)
    
    button = tk.Button(
        frame,
        text='Load CT Scan',
        font=('Aptos', 20),
        bg='#3f3f3f',
        fg='white',
        activebackground='#2f2f2f',
        activeforeground='white',
        width = width//5,
    )

    button.pack(padx=20, pady=20, side=tk.BOTTOM)

    return frame

def draw_tab_1(tabControl, width, height):
    tab = tk.Frame(
        master=tabControl,
        width = width,
        height=height,
        bg='#1f1f1f',
    )

    image = tk.Canvas(
        tab,
        width=width//2,
        height=height//2,
        )
    image.pack(padx=20, pady=20, side=tk.TOP, anchor=NW)
    image.bind("<Button 1>", get_mouse_click)
    File = askopenfilename(parent=tabControl, initialdir="./", title="Select image")
    original = PIL.Image.open(File)
    img = PIL.ImageTk.PhotoImage(original)
    image.img_ref = img
    image.create_image(0, 0, image=img, anchor="nw")

    return tab

def get_mouse_click(event):
    global px, py

    px = event.x
    py = event.y

def draw_frame_top(tabs, tabControl, window, width, height):
    for tab in tabs:
        tabControl.add(tabs[tab], text=tab)

    return tabControl
                
if __name__ == "__main__":
    gui_thread = Thread(target=gui_app, args=())
    gui_thread.start()
