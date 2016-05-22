import plotly
import plotly.plotly as py
import pandas as pd
from plotly.widgets import GraphWidget

from IPython.html import widgets 
from IPython.display import display, clear_output
import numpy as np

py.sign_in('dswbtest', 'ci8iu7p6wi') #plotly API credentials

food = pd.read_csv("supply.csv")

# Definition of a function that defines the plot settings
def foodmap(year):
    year = str(year)
    # Properties of the data and how it is displayed
    data = [ dict(
            type = 'choropleth',
            locations = food['Country code'],
            z = food[year],
            text = food['Country name'],
            colorscale = [[0,"rgb(51,160,44)"],
                          [0.5,"rgb(255,255,51)"],
                          [1,"rgb(227,26,28)"]],
            opacity = 1,
            autocolorscale = False,
            reversescale = False,
            marker = dict(
                line = dict (
                    color = 'rgb(0,0,0)',
                    width = 0.5
                )
            ),
            colorbar = dict(
                autotick = True,
                title = 'kcal per capita'
            ),
        ) ]
    # Properties of the plot
    layout = dict(
        title = 'Food Supply (kcal per capita) in ' + str(year),
        geo = dict(
            showframe = False,
            showcoastlines = False,
            showcountry = True,
            countrycolor = "rgb(220, 0, 0)",
            coastlinecolor = "rgb(220, 0, 0)",
            landcolor = "rgb(220, 0, 0)",
            projection = dict(
                type = 'Mercator',
                scale = 1
            )
        )
    )

    fig = dict( data=data, layout=layout )
    url = py.plot( fig, validate=False, filename='d3-food-map' )
    return url

# Graph object
g = GraphWidget(foodmap(1961))

# Definition of a class that will update the graph object
class z_data:
    def __init__(self):
        self.z = food[str(int(1961))]
    
    def on_z_change(self, name, old_value, new_value):
        self.z = food[str(int(new_value))]
        self.title = "Food Supply (kcal per capita) in " + str(new_value)
        self.replot()
        
    def replot(self):
        g.restyle({ 'z': [self.z] })
        g.relayout({'title': self.title})

# Interactive object
edu_slider = widgets.IntSlider(min=1961,max=2011,value=1961,step=1)
edu_slider.description = 'Year'
edu_slider.value = 1961

z_state = z_data()
edu_slider.on_trait_change(z_state.on_z_change, 'value')

display(edu_slider)
display(g)