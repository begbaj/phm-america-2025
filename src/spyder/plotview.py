import os
import plotly.offline
import webbrowser  # <--- The standard library for opening URLs

def show_plotly(fig, browser="firefox"):
    # 1. Create the HTML file path
    file_path = os.path.abspath(os.path.join(os.getcwd(), "name.html"))
    
    # 2. Save the plot to HTML (don't open it yet)
    plotly.offline.plot(fig, filename=file_path, auto_open=False)

    # 3. Open it in your default browser (Chrome/Firefox/Edge)
    # This avoids the "EGL_BAD_CONTEXT" crash entirely because 
    # it doesn't use the embedded Qt browser.
    webbrowser.get(browser).open('file://' + file_path)