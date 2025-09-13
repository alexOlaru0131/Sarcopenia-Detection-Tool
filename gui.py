###### gui.py ######
# ->

###### IMPORTS ######
from imports import *
###### END IMPORTS ######

for m in get_monitors():
    width = m.width
    height = m.height
    resolution = str(m.width) + 'x' + str(m.height)

p_xy = {
    "x": [],
    "y": [],
    "x, y": [],
    "point_drawn": [],
        }

zones = []
mouse_held = False

start = 0
end = 0

def_img = np.zeros((width//2, height//2, 3), dtype = np.uint8)
object_counter = 0

def gui_app():
    global width, height, resolution, def_img
    window = tk.Tk()
    tabControl = ttk.Notebook(window)

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
                                    "font": ('Aptos Extra Bold', 15),
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

    tab1, load_image = draw_tab_1(tabControl, width, height)

    frame_left = draw_frame_left(window, width, height, load_image)
    frame_left.pack(padx=0, pady=0, side=tk.LEFT)

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

def draw_frame_left(window, width, height, load_image):
    global button

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
        font=('Aptos', 15),
        bg='#3f3f3f',
        fg='white',
        activebackground='#2f2f2f',
        activeforeground='white',
        width = width//5,
        command=load_image,
    )

    button.pack(padx=20, pady=20, side=tk.BOTTOM)

    return frame

def draw_tab_1(tabControl, width, height):
    global def_img, start, end, p_xy

    start = time.time()

    tab = tk.Frame(tabControl, width=width, height=height, bg='#1f1f1f')

    image = tk.Canvas(tab, width=width//2, height=height//2)
    image.pack(padx=20, pady=20, side=tk.TOP, anchor=NW)

    def load_image():
        global def_img
        File = askopenfilename(parent=tab, initialdir="./", title="Select image")
        if not File:
            return
        original = PIL.Image.open(File)
        def_img = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)

        img = PIL.Image.fromarray(cv2.cvtColor(def_img, cv2.COLOR_BGR2RGB))
        tk_img = PIL.ImageTk.PhotoImage(img)
        image.img_ref = tk_img
        image.delete("all")
        image.create_image(0, 0, image=tk_img, anchor="nw")

    image.bind("<B1-Motion>", lambda e: mouse_click(e, image, def_img))
    image.bind("<ButtonRelease-1>", lambda e: mouse_release(e))


    return tab, load_image

def mouse_release(event):
    global p_xy, overlay, mouse_held
    overlay = np.zeros_like(def_img)

    zones.append(p_xy)

    p_xy = {
        "x": [],
        "y": [],
        "x, y": [],
        "point_drawn": [],
            }

### TO DO: FIX STUPID BUG WHERE THE COLOR INTENSITY IS SCALING
def draw_saved_polygons():
    global zones, def_img
    alpha = 0.3

    for zone in zones:
        xy = zone["x, y"]
        x = zone["x"]
        y = zone["y"]

        for i in range(len(x)-1):
            cv2.line(overlay, (x[i], y[i]), \
                    (x[i+1], y[i+1]), (0, 0, 255), 5)
        
        points = np.array(xy, np.int32)
        points = points.reshape((-1, 1, 2))

        if points.any():
            cv2.fillPoly(overlay, [points], color=(0, 0, 255))
        cv2.addWeighted(overlay, alpha, def_img, 1 - alpha, 0, def_img)

def mouse_click(event, image, def_img):
    global object_counter, start, mouse_held, overlay
    overlay = np.zeros_like(def_img)

    mouse_held = True

    px_i = event.x
    py_i = event.y

    p_xy["x"].append(px_i)
    p_xy["y"].append(py_i)
    p_xy["x, y"].append([px_i, py_i])
    p_xy["point_drawn"].append(False)

    cv2.circle(def_img, (px_i, py_i), 5, (0, 0, 255))

    draw_polygon(p_xy["x"], p_xy["y"])
    draw_saved_polygons()

    updated = PIL.Image.fromarray(cv2.cvtColor(def_img, cv2.COLOR_BGR2RGB))
    tk_img = PIL.ImageTk.PhotoImage(updated)
    image.img_ref = tk_img
    image.delete("all")
    image.create_image(0, 0, image=tk_img, anchor="nw")

def draw_polygon(points_x, points_y):
    global def_img, p_xy
    alpha = 0.3

    if len(points_x) > 2:
        for i in range(len(points_x)-1):
            if p_xy["point_drawn"][i] == False and \
               p_xy["point_drawn"][i+1] == False:
                cv2.line(overlay, (points_x[i], points_y[i]), \
                        (points_x[i+1], points_y[i+1]), (0, 0, 255), 5)
    
    valid_points = []
    for i in range(len(points_x)-1):
        if p_xy["point_drawn"][i] == False and \
           p_xy["point_drawn"][i+1] == False:
            valid_points.append(p_xy["x, y"])
            p_xy["point_drawn"][i] = True
            p_xy["point_drawn"][i+1] = True
    points = np.array(valid_points, np.int32)
    points = points.reshape((-1, 1, 2))

    if points.any():
        cv2.fillPoly(overlay, [points], color=(0, 0, 255))
    cv2.addWeighted(overlay, alpha, def_img, 1 - alpha, 0, def_img)

def draw_frame_top(tabs, tabControl, window, width, height):
    for tab in tabs:
        tabControl.add(tabs[tab], text=tab)

    return tabControl
                
if __name__ == "__main__":
    gui_thread = Thread(target=gui_app, args=())
    gui_thread.start()


