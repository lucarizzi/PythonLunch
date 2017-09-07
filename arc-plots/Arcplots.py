
# coding: utf-8

# In[4]:
from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter

import os,sys

from bokeh.plotting import figure, output_file, show, curdoc
from bokeh.models import Range1d, HoverTool, ColumnDataSource, CustomJS

from bokeh.layouts import widgetbox, layout
from bokeh.models.widgets import Button, RadioButtonGroup, Select, Slider, Dropdown, TextInput


# In[5]:
d = dict(one='1', two='2')
lampList = {
    'lamp1': dict(file="lamp1.dat", name="He"), 
    'lamp2': dict(file="lamp2.dat", name="Ar")}


directory = "arcplots"

sides = ["red","blue"]

gratings = {
    "150/7500": {
        "dispersion": 3,
        "range":12288,
        "id":"150/7500"
    },
    "300/5000": {
        "dispersion": 1.59,
        "range":6525,
        "id":"300/5000"
    },
    "400/8500": {
        "dispersion": 1.16,
        "range": 4762,
        "id":"400/8500"
    },
    "600/7500": {
        "dispersion": 0.80,
        "range": 3275,
        "id":"600/7500"
    },
    "600/5000": {
        "dispersion": 0.80,
        "range": 3275,
        "id":"600/5000"
    },
    "600/10000": {
        "dispersion": 0.80,
        "range":3275,
        "id":"600/10000"
    },
    "831/8200": {
        "dispersion": 0.58,
        "range": 2375,
        "id":"831/8200"
    }
}

# convert into a pandas data source

gratings = pd.Series(gratings)

# and make it an object that can be shared with the JavaScript layer

gratings_source = ColumnDataSource(data={'gratings':gratings})


for key in lampList.keys():
    file = os.path.join(directory,lampList[key]["file"])
    df = pd.read_csv(file,delim_whitespace=True)
    df.columns=['lambda','intensity']
    lampList[key]["lines"] = df

    


# In[17]:




# In[6]:

output_file('arcplot.html')
hover = HoverTool(tooltips=[
    ("lambda", "$x{00000.0}"),
])

p = figure(plot_width=600, plot_height=400)
p.add_tools(hover)


# In[7]:

source1 = ColumnDataSource(data={
    "x":lampList['lamp1']['lines']['lambda'],
    "y":gaussian_filter(lampList['lamp1']['lines']['intensity'],sigma=4)}, 
    id=lampList['lamp1']['name']
    )
source2 = ColumnDataSource(data={
    "x":lampList['lamp2']['lines']['lambda'],
    "y":gaussian_filter(lampList['lamp2']['lines']['intensity'],sigma=4)}, 
    id=lampList['lamp2']['name']
    )


# In[8]:

line1 = p.line("x","y", source=source1, color='firebrick', legend=lampList['lamp1']['name'])
line2 = p.line("x","y", source=source2, color='navy', legend=lampList['lamp2']['name'])


initial_grating_selection='600/10000'
initial_lambda_selection = 7500
initial_range = gratings[initial_grating_selection]['range']
initial_xmin = initial_lambda_selection-initial_range/2
initial_xmax = initial_lambda_selection+initial_range/2

p.x_range=Range1d(initial_xmin, initial_xmax)


# In[9]:

p.title.text = "LRIS Arcplot"
p.title.align = "right"
p.title.text_color = "orange"
p.title.text_font_size = "25px"


# In[10]:

p.background_fill_color = "beige"
p.background_fill_alpha = 0.5


# In[11]:

p.xaxis.axis_label = "Wavelength"
p.yaxis.axis_label = "Intensity"


# In[12]:

p.legend.click_policy="hide"


rangeBox = TextInput(title="Wavelength range", value=str(initial_range))
xminBox = TextInput(title="Minimum wavelength", value=str(initial_xmin))
xmaxBox = TextInput(title="Maximum Wavelength", value=str(initial_xmax))

select = Select(title="Gratings:",  value=initial_grating_selection, options=[str(x) for x in gratings.keys()])

update_xrange_slider = CustomJS(args=dict(x_range=p.x_range,  gratings=gratings_source, select=select, 
                                          rangeBox=rangeBox, xminBox= xminBox, xmaxBox= xmaxBox), code="""
var central = cb_obj.get("value");
var selected_grating = select.value;
var gratings_data = gratings.get('data')
var gratings = gratings_data['gratings']
var lookup = {};
for (var i=0, len = gratings.length; i < len; i++) {
    lookup[gratings[i].id] = gratings[i];
    }
var range = lookup[selected_grating.toString()].range;
var xmin = central-range/2;
var xmax = central+range/2;

x_range.set("start", xmin);
x_range.set("end", xmax);
rangeBox.set("value", range.toString());
xminBox.set("value", xmin.toString());
xmaxBox.set("value", xmax.toString());
""")

slider = Slider(start=3700,end=10000,value=7500, step=1, title="Central wavelength", callback=update_xrange_slider)

update_xrange_select = CustomJS(args=dict(x_range=p.x_range,  gratings=gratings_source, slider=slider,
                                          rangeBox=rangeBox, xminBox= xminBox, xmaxBox= xmaxBox), code="""
var central = slider.value;
var selected_grating = cb_obj.get("value");
var gratings_data = gratings.get('data')
var gratings = gratings_data['gratings']
var lookup = {};
for (var i=0, len = gratings.length; i < len; i++) {
    lookup[gratings[i].id] = gratings[i];
    }
var range = lookup[selected_grating.toString()].range;
var xmin = central-range/2;
var xmax = central+range/2;

x_range.set("start", xmin);
x_range.set("end", xmax);
rangeBox.set("value", range.toString());
xminBox.set("value", xmin.toString());
xmaxBox.set("value", xmax.toString());
""")





select.js_on_change("change", update_xrange_select)

# In[15]:
#def update_range(attr,old,new):
#    grating = str(select.value)
#    sys.stdout.write(grating)
#    range = float(gratings[grating]['range'])
#    xmin = float(slider.value-range/2)
#    xmax = float(slider.value+range/2)
#    rangeBox.value = str(range)
#    xminBox.value = str(xmin)
#    xmaxBox.value = str(xmax)
#    p.title.text=select.value


#select.on_change('value',update_range)


# In[18]:

l = layout([
    [select],
    [slider],
    [p],
    [rangeBox],
    [xminBox,xmaxBox]
])
show(l)

# In[20]:

curdoc().add_root(l)


# In[ ]:

#show(widgetbox(p,button_1, slider, button_group, select, button_2, width=300))


# In[19]:




